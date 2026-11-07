import numpy as np
import pandas as pd
import torch

DATA_PATH = "data/NF-BoT-IoT.csv"
FEATURE_COLUMNS = [
    "PROTOCOL",
    "L7_PROTO",
    "IN_BYTES",
    "OUT_BYTES",
    "IN_PKTS",
    "OUT_PKTS",
    "TCP_FLAGS",
    "FLOW_DURATION_MILLISECONDS",
]
CLASS_NAMES = ["Benign", "DDoS", "DoS", "Reconnaissance", "Theft"]
LOG_COLUMNS = ["IN_BYTES", "OUT_BYTES", "IN_PKTS", "OUT_PKTS", "FLOW_DURATION_MILLISECONDS"]


def load_flows(path=DATA_PATH):
    flows = pd.read_csv(path)
    flows["src_node"] = flows["IPV4_SRC_ADDR"] + ":" + flows["L4_SRC_PORT"].astype(str)
    flows["dst_node"] = flows["IPV4_DST_ADDR"] + ":" + flows["L4_DST_PORT"].astype(str)
    flows["y"] = flows["Attack"].map({name: index for index, name in enumerate(CLASS_NAMES)})
    return flows


class StreamScaler:
    def __init__(self):
        self.mean = None
        self.std = None

    def transform_raw(self, frame):
        values = frame[FEATURE_COLUMNS].astype(np.float64).copy()
        for column in LOG_COLUMNS:
            values[column] = np.log1p(values[column].clip(lower=0))
        return values.values

    def fit(self, frame):
        values = self.transform_raw(frame)
        self.mean = values.mean(axis=0)
        self.std = values.std(axis=0) + 1e-6
        return self

    def transform(self, frame):
        return ((self.transform_raw(frame) - self.mean) / self.std).astype(np.float32)


def stream_windows(flows, window_size):
    for start in range(0, len(flows), window_size):
        yield start // window_size, flows.iloc[start:start + window_size]


def build_graph(window_frames, scaler, device):
    frame = pd.concat(window_frames, ignore_index=True)
    node_keys = pd.Index(pd.unique(pd.concat([frame["src_node"], frame["dst_node"]])))
    src = node_keys.get_indexer(frame["src_node"])
    dst = node_keys.get_indexer(frame["dst_node"])
    return {
        "num_nodes": len(node_keys),
        "src": torch.as_tensor(src, dtype=torch.long, device=device),
        "dst": torch.as_tensor(dst, dtype=torch.long, device=device),
        "edge_attr": torch.as_tensor(scaler.transform(frame), device=device),
        "y": torch.as_tensor(frame["y"].values, dtype=torch.long, device=device),
        "num_edges": len(frame),
    }
