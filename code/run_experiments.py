import json
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score

from data_stream import CLASS_NAMES, StreamScaler, build_graph, load_flows, stream_windows
from models import FlowMLP, StreamingEGraphSAGE

SEED = 42
WINDOW_SIZE = 10000
MEMORY_WINDOWS = 2
HIDDEN_DIM = 64
LEARNING_RATE = 1e-3
OFFLINE_EPOCHS = 15
STREAM_WARMUP_STEPS = 60
STREAM_STEPS_PER_WINDOW = 30
REPLAY_PER_CLASS = 1000
RESULT_DIR = os.environ.get("KETQUA_DIR", "ketqua")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = len(CLASS_NAMES)


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def class_weights(counts):
    counts = np.asarray(counts, dtype=np.float64)
    weights = np.where(counts > 0, np.sqrt(counts.sum() / (NUM_CLASSES * np.maximum(counts, 1))), 1.0)
    return torch.as_tensor(weights, dtype=torch.float32, device=DEVICE)


def evaluate(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    benign_index = CLASS_NAMES.index("Benign")
    true_attack = y_true != benign_index
    pred_attack = y_pred != benign_index
    benign_total = max((~true_attack).sum(), 1)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(precision_score(y_true, y_pred, average="macro", labels=range(NUM_CLASSES), zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, average="macro", labels=range(NUM_CLASSES), zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", labels=range(NUM_CLASSES), zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", labels=range(NUM_CLASSES), zero_division=0)),
        "per_class_f1": [float(v) for v in f1_score(y_true, y_pred, average=None, labels=range(NUM_CLASSES), zero_division=0)],
        "detection_rate": float((pred_attack & true_attack).sum() / max(true_attack.sum(), 1)),
        "false_alarm_rate": float((pred_attack & ~true_attack).sum() / benign_total),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=range(NUM_CLASSES)).tolist(),
    }


def train_steps(model, optimizer, graph, mask, weights, steps):
    model.train()
    for _ in range(steps):
        optimizer.zero_grad()
        logits = model(graph)
        loss = F.cross_entropy(logits[mask], graph["y"][mask], weight=weights)
        loss.backward()
        optimizer.step()
    return float(loss.item())


@torch.no_grad()
def predict(model, graph):
    model.eval()
    return model(graph).argmax(dim=1)


def run_offline(flows):
    set_seed(SEED)
    rng = np.random.default_rng(SEED)
    is_train = rng.random(len(flows)) < 0.7
    flows = flows.assign(is_train=is_train)
    scaler = StreamScaler().fit(flows[flows.is_train])

    train_frame = flows[flows.is_train]
    test_frame = flows[~flows.is_train]
    x_train, y_train = scaler.transform(train_frame), train_frame["y"].values
    x_test, y_test = scaler.transform(test_frame), test_frame["y"].values

    results = {}
    timings = {}

    start = time.time()
    logistic = LogisticRegression(max_iter=1000, class_weight="balanced")
    logistic.fit(x_train, y_train)
    results["Logistic Regression"] = evaluate(y_test, logistic.predict(x_test))
    timings["Logistic Regression"] = time.time() - start

    start = time.time()
    forest = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=SEED, class_weight="balanced")
    forest.fit(x_train, y_train)
    results["Random Forest"] = evaluate(y_test, forest.predict(x_test))
    timings["Random Forest"] = time.time() - start

    graphs = []
    for _, window in stream_windows(flows, WINDOW_SIZE):
        graph = build_graph([window], scaler, DEVICE)
        graph["train_mask"] = torch.as_tensor(window["is_train"].values, device=DEVICE)
        graphs.append(graph)
    weights = class_weights(np.bincount(y_train, minlength=NUM_CLASSES))

    for name, model in [
        ("MLP", FlowMLP(len(scaler.mean), HIDDEN_DIM, NUM_CLASSES)),
        ("E-GraphSAGE", StreamingEGraphSAGE(len(scaler.mean), HIDDEN_DIM, NUM_CLASSES)),
    ]:
        set_seed(SEED)
        model = model.to(DEVICE)
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
        start = time.time()
        for _ in range(OFFLINE_EPOCHS):
            for index in np.random.permutation(len(graphs)):
                train_steps(model, optimizer, graphs[index], graphs[index]["train_mask"], weights, 1)
        true_values, pred_values = [], []
        for graph in graphs:
            test_mask = ~graph["train_mask"]
            pred_values.append(predict(model, graph)[test_mask].cpu().numpy())
            true_values.append(graph["y"][test_mask].cpu().numpy())
        results[name] = evaluate(np.concatenate(true_values), np.concatenate(pred_values))
        timings[name] = time.time() - start
        print(name, results[name]["macro_f1"], flush=True)

    for name in results:
        results[name]["train_seconds"] = timings[name]
    return results, {"train": int(is_train.sum()), "test": int((~is_train).sum())}


class ClassBalancedReplay:
    def __init__(self, capacity_per_class, seed):
        self.capacity = capacity_per_class
        self.rng = np.random.default_rng(seed)
        self.buffers = {}
        self.seen = {}

    def add(self, window):
        for label, group in window.groupby("y"):
            buffer = self.buffers.get(label)
            seen = self.seen.get(label, 0)
            rows = list(buffer.itertuples(index=False)) if buffer is not None else []
            for row in group.itertuples(index=False):
                seen += 1
                if len(rows) < self.capacity:
                    rows.append(row)
                else:
                    slot = self.rng.integers(0, seen)
                    if slot < self.capacity:
                        rows[slot] = row
            self.buffers[label] = pd.DataFrame(rows, columns=window.columns)
            self.seen[label] = seen

    def frame(self):
        return pd.concat(self.buffers.values(), ignore_index=True)


def run_streaming(flows, model_name, use_replay=True):
    set_seed(SEED)
    windows = list(stream_windows(flows, WINDOW_SIZE))
    scaler = StreamScaler().fit(windows[0][1])
    edge_dim = len(scaler.mean)
    if model_name == "Streaming E-GraphSAGE":
        model = StreamingEGraphSAGE(edge_dim, HIDDEN_DIM, NUM_CLASSES)
    else:
        model = FlowMLP(edge_dim, HIDDEN_DIM, NUM_CLASSES)
    model = model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    seen_counts = np.bincount(windows[0][1]["y"].values, minlength=NUM_CLASSES)
    warmup_graph = build_graph([windows[0][1]], scaler, DEVICE)
    all_edges = torch.ones(warmup_graph["num_edges"], dtype=torch.bool, device=DEVICE)
    train_steps(model, optimizer, warmup_graph, all_edges, class_weights(seen_counts), STREAM_WARMUP_STEPS)

    memory = [windows[0][1]]
    replay = ClassBalancedReplay(REPLAY_PER_CLASS, SEED)
    replay.add(windows[0][1])
    per_window = []
    true_values, pred_values = [], []
    for window_id, window in windows[1:]:
        memory = (memory + [window])[-MEMORY_WINDOWS:]
        if DEVICE.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        graph = build_graph(memory, scaler, DEVICE)
        current_mask = torch.zeros(graph["num_edges"], dtype=torch.bool, device=DEVICE)
        current_mask[-len(window):] = True
        prediction = predict(model, graph)[current_mask]
        if DEVICE.type == "cuda":
            torch.cuda.synchronize()
        latency = time.perf_counter() - start

        y_true = window["y"].values
        y_pred = prediction.cpu().numpy()
        true_values.append(y_true)
        pred_values.append(y_pred)
        per_window.append({
            "window": window_id,
            "flows": len(window),
            "nodes": graph["num_nodes"],
            "edges": graph["num_edges"],
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
            "latency_ms": latency * 1000,
            "throughput": len(window) / latency,
            "dominant_class": CLASS_NAMES[int(np.bincount(y_true).argmax())],
        })

        seen_counts = seen_counts + np.bincount(y_true, minlength=NUM_CLASSES)
        if use_replay:
            train_graph = build_graph(memory + [replay.frame()], scaler, DEVICE)
            replay.add(window)
        else:
            train_graph = graph
        all_edges = torch.ones(train_graph["num_edges"], dtype=torch.bool, device=DEVICE)
        train_steps(model, optimizer, train_graph, all_edges, class_weights(seen_counts), STREAM_STEPS_PER_WINDOW)

    summary = evaluate(np.concatenate(true_values), np.concatenate(pred_values))
    summary["mean_latency_ms"] = float(np.mean([w["latency_ms"] for w in per_window]))
    summary["mean_throughput"] = float(np.mean([w["throughput"] for w in per_window]))
    print(model_name, use_replay, summary["macro_f1"], summary["weighted_f1"], summary["false_alarm_rate"], summary["mean_latency_ms"], flush=True)
    return summary, per_window


def main():
    os.makedirs(RESULT_DIR, exist_ok=True)
    flows = load_flows()
    dataset_info = {
        "num_flows": len(flows),
        "num_src_ip": int(flows["IPV4_SRC_ADDR"].nunique()),
        "num_dst_ip": int(flows["IPV4_DST_ADDR"].nunique()),
        "num_endpoints": int(pd.unique(pd.concat([flows["src_node"], flows["dst_node"]])).size),
        "class_counts": {name: int((flows["Attack"] == name).sum()) for name in CLASS_NAMES},
        "num_windows": int(np.ceil(len(flows) / WINDOW_SIZE)),
        "window_class_counts": [
            np.bincount(window["y"].values, minlength=NUM_CLASSES).tolist()
            for _, window in stream_windows(flows, WINDOW_SIZE)
        ],
    }

    offline_results, split_info = run_offline(flows)
    dataset_info["split"] = split_info

    streaming_results = {}
    streaming_windows = {}
    for model_name, base_model, use_replay in [
        ("Streaming E-GraphSAGE + Replay", "Streaming E-GraphSAGE", True),
        ("Streaming E-GraphSAGE", "Streaming E-GraphSAGE", False),
        ("Streaming MLP + Replay", "Streaming MLP", True),
        ("Streaming MLP", "Streaming MLP", False),
    ]:
        summary, per_window = run_streaming(flows, base_model, use_replay)
        streaming_results[model_name] = summary
        streaming_windows[model_name] = per_window
        pd.DataFrame(per_window).to_csv(
            os.path.join(RESULT_DIR, model_name.replace(" + ", "_").replace(" ", "_").replace("-", "").lower() + "_windows.csv"), index=False
        )

    config = {
        "seed": SEED,
        "window_size": WINDOW_SIZE,
        "memory_windows": MEMORY_WINDOWS,
        "hidden_dim": HIDDEN_DIM,
        "learning_rate": LEARNING_RATE,
        "offline_epochs": OFFLINE_EPOCHS,
        "stream_warmup_steps": STREAM_WARMUP_STEPS,
        "stream_steps_per_window": STREAM_STEPS_PER_WINDOW,
        "replay_per_class": REPLAY_PER_CLASS,
        "device": str(DEVICE),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "",
    }
    with open(os.path.join(RESULT_DIR, "metrics.json"), "w", encoding="utf-8") as file:
        json.dump({
            "config": config,
            "dataset": dataset_info,
            "offline": offline_results,
            "streaming": streaming_results,
            "streaming_windows": streaming_windows,
        }, file, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
