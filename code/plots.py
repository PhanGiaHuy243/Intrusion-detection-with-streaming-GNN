import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from data_stream import CLASS_NAMES

RESULT_DIR = os.environ.get("KETQUA_DIR", "ketqua")
FIGURE_DIR = os.path.join(RESULT_DIR, "hinh")
COLORS = ["#2b6cb0", "#dd6b20", "#38a169", "#805ad5", "#c53030", "#718096"]
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Liberation Serif", "DejaVu Serif"], "font.size": 13, "figure.dpi": 150})


def save(figure, name):
    figure.tight_layout()
    figure.savefig(os.path.join(FIGURE_DIR, name), dpi=200, bbox_inches="tight")
    plt.close(figure)


def merged_dos_metrics(matrix):
    matrix = np.asarray(matrix)
    groups = [[0], [1, 2], [3], [4]]
    merged = np.array([[matrix[np.ix_(a, b)].sum() for b in groups] for a in groups])
    precision = np.diag(merged) / np.maximum(merged.sum(axis=0), 1)
    recall = np.diag(merged) / np.maximum(merged.sum(axis=1), 1)
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-9)
    support = merged.sum(axis=1)
    return {
        "accuracy": float(np.trace(merged) / merged.sum()),
        "macro_f1": float(f1.mean()),
        "weighted_f1": float((f1 * support).sum() / support.sum()),
        "per_class_f1": f1.tolist(),
    }


def plot_class_distribution(dataset):
    counts = [dataset["class_counts"][name] for name in CLASS_NAMES]
    figure, axis = plt.subplots(figsize=(7, 4))
    bars = axis.bar(CLASS_NAMES, counts, color=COLORS[:5])
    axis.set_yscale("log")
    axis.set_ylabel("Số luồng (thang log)")
    for bar, value in zip(bars, counts):
        axis.text(bar.get_x() + bar.get_width() / 2, value * 1.1, f"{value:,}", ha="center", fontsize=10)
    axis.set_ylim(top=max(counts) * 4)
    save(figure, "h_phanbo_lop.png")


def plot_window_drift(dataset):
    matrix = np.array(dataset["window_class_counts"], dtype=float)
    ratio = matrix / matrix.sum(axis=1, keepdims=True)
    figure, axis = plt.subplots(figsize=(9, 4))
    axis.stackplot(np.arange(len(ratio)), ratio.T, labels=CLASS_NAMES, colors=COLORS[:5], alpha=0.9)
    axis.set_xlabel("Chỉ số cửa sổ (mỗi cửa sổ 10.000 luồng)")
    axis.set_ylabel("Tỉ lệ lớp trong cửa sổ")
    axis.set_xlim(0, len(ratio) - 1)
    axis.set_ylim(0, 1)
    axis.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=5, frameon=False)
    save(figure, "h_troi_du_lieu.png")


def plot_offline_comparison(offline):
    names = list(offline.keys())
    metrics = [("accuracy", "Accuracy"), ("macro_f1", "Macro-F1"), ("weighted_f1", "Weighted-F1"), ("detection_rate", "Detection Rate")]
    figure, axis = plt.subplots(figsize=(9, 4.5))
    width = 0.2
    positions = np.arange(len(names))
    for index, (key, label) in enumerate(metrics):
        values = [offline[name][key] for name in names]
        bars = axis.bar(positions + (index - 1.5) * width, values, width, label=label, color=COLORS[index])
        for bar, value in zip(bars, values):
            axis.text(bar.get_x() + bar.get_width() / 2, value + 0.01, f"{value:.2f}", ha="center", fontsize=8)
    axis.set_xticks(positions)
    axis.set_xticklabels(names)
    axis.set_ylim(0, 1.12)
    axis.set_ylabel("Giá trị")
    axis.legend(ncol=4, loc="upper center", frameon=False)
    save(figure, "h_sosanh_offline.png")


def plot_per_class(offline, streaming):
    rows = {**offline, **{k: v for k, v in streaming.items() if "Replay" not in k}}
    figure, axis = plt.subplots(figsize=(10, 4.5))
    width = 0.8 / len(rows)
    positions = np.arange(len(CLASS_NAMES))
    for index, (name, result) in enumerate(rows.items()):
        axis.bar(positions + (index - (len(rows) - 1) / 2) * width, result["per_class_f1"], width, label=name, color=COLORS[index % len(COLORS)])
    axis.set_xticks(positions)
    axis.set_xticklabels(CLASS_NAMES)
    axis.set_ylabel("F1-score")
    axis.set_ylim(0, 1.05)
    axis.legend(ncol=3, fontsize=10.5, loc="upper left", frameon=False)
    save(figure, "h_f1_tung_lop.png")


def plot_confusion(result, name):
    matrix = np.array(result["confusion_matrix"], dtype=float)
    normalized = matrix / np.maximum(matrix.sum(axis=1, keepdims=True), 1)
    figure, axis = plt.subplots(figsize=(6.2, 5.2))
    image = axis.imshow(normalized, cmap="Blues", vmin=0, vmax=1)
    for row in range(len(CLASS_NAMES)):
        for column in range(len(CLASS_NAMES)):
            color = "white" if normalized[row, column] > 0.55 else "black"
            axis.text(column, row, f"{normalized[row, column]:.2f}\n({int(matrix[row, column])})", ha="center", va="center", fontsize=8, color=color)
    axis.set_xticks(range(len(CLASS_NAMES)))
    axis.set_yticks(range(len(CLASS_NAMES)))
    axis.set_xticklabels(CLASS_NAMES, rotation=30)
    axis.set_yticklabels(CLASS_NAMES)
    axis.set_xlabel("Nhãn dự đoán")
    axis.set_ylabel("Nhãn thực tế")
    figure.colorbar(image, ax=axis, fraction=0.046)
    save(figure, name)


def plot_stream_timeline(windows):
    figure, axis = plt.subplots(figsize=(9, 4.2))
    for index, (name, rows) in enumerate(windows.items()):
        axis.plot([r["window"] for r in rows], [r["weighted_f1"] for r in rows], marker="o", markersize=3, linewidth=1.3, label=name, color=COLORS[index])
    axis.set_xlabel("Chỉ số cửa sổ dữ liệu luồng")
    axis.set_ylabel("Weighted-F1 trên cửa sổ")
    axis.set_ylim(0, 1.05)
    axis.legend(ncol=2, fontsize=11, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.2))
    axis.grid(alpha=0.3)
    save(figure, "h_f1_theo_cuaso.png")


def plot_latency(windows):
    figure, (left, right) = plt.subplots(1, 2, figsize=(9.5, 4.2))
    for index, (name, rows) in enumerate(windows.items()):
        left.plot([r["window"] for r in rows], [r["latency_ms"] for r in rows], linewidth=1.2, label=name, color=COLORS[index])
        right.bar(index, np.mean([r["throughput"] for r in rows]), color=COLORS[index])
    left.set_xlabel("Chỉ số cửa sổ")
    left.set_ylabel("Độ trễ suy luận (ms)")
    left.legend(fontsize=9.5, frameon=False, loc="upper right")
    left.grid(alpha=0.3)
    right.set_xticks(range(len(windows)))
    right.set_xticklabels([n.replace("Streaming ", "").replace(" + ", "\n+ ") for n in windows], fontsize=10.5)
    right.set_ylabel("Thông lượng TB (luồng/giây)")
    save(figure, "h_do_tre.png")


def plot_graph_size(windows):
    rows = windows["Streaming E-GraphSAGE"]
    figure, axis = plt.subplots(figsize=(8.5, 3.8))
    axis.plot([r["window"] for r in rows], [r["nodes"] for r in rows], label="Số nút (IP:port)", color=COLORS[0])
    axis.plot([r["window"] for r in rows], [r["edges"] for r in rows], label="Số cạnh (luồng)", color=COLORS[1])
    axis.set_xlabel("Chỉ số cửa sổ")
    axis.set_ylabel("Kích thước đồ thị")
    axis.legend(frameon=False)
    axis.grid(alpha=0.3)
    save(figure, "h_kichthuoc_dothi.png")


def draw_box(axis, x, y, width, height, text, color):
    axis.add_patch(FancyBboxPatch((x, y), width, height, boxstyle="round,pad=0.02", facecolor=color, edgecolor="#2d3748", linewidth=1.2))
    axis.text(x + width / 2, y + height / 2, text, ha="center", va="center", fontsize=12.5)


def draw_arrow(axis, start, end):
    axis.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=14, color="#2d3748", linewidth=1.3))


def plot_architecture():
    figure, axis = plt.subplots(figsize=(10.5, 5.6))
    axis.set_xlim(0, 12)
    axis.set_ylim(0.1, 5.8)
    axis.axis("off")
    draw_box(axis, 0.1, 2.3, 2.0, 1.5, "Luồng\nNetFlow\nNF-BoT-IoT", "#ebf8ff")
    draw_box(axis, 2.5, 2.3, 2.0, 1.5, "Bộ phát\nmicro-batch\n10.000 flow", "#ebf8ff")
    draw_box(axis, 4.9, 2.3, 2.0, 1.5, "Tiền xử lý\nlog1p +\nchuẩn hoá", "#fefcbf")
    draw_box(axis, 7.3, 2.3, 2.0, 1.5, "Đồ thị động\nnút = IP:port\ncạnh = flow", "#fefcbf")
    draw_box(axis, 9.8, 3.7, 2.1, 1.5, "E-GraphSAGE\n2 tầng\nphân loại cạnh", "#e9d8fd")
    draw_box(axis, 9.8, 0.9, 2.1, 1.5, "Cảnh báo\n5 lớp\n+ độ trễ", "#fed7d7")
    draw_box(axis, 4.9, 0.25, 4.4, 1.4, "Học trực tuyến (test-then-train)\n+ replay cân bằng lớp", "#c6f6d5")
    draw_arrow(axis, (2.1, 3.05), (2.5, 3.05))
    draw_arrow(axis, (4.5, 3.05), (4.9, 3.05))
    draw_arrow(axis, (6.9, 3.05), (7.3, 3.05))
    draw_arrow(axis, (9.3, 3.4), (9.8, 4.3))
    draw_arrow(axis, (10.85, 3.7), (10.85, 2.4))
    draw_arrow(axis, (9.8, 1.0), (9.3, 0.7))
    draw_arrow(axis, (9.3, 1.4), (9.85, 3.7))
    axis.text(4.7, 5.4, "Khối dữ liệu luồng (Streaming)", fontsize=14, fontweight="bold", ha="center")
    axis.text(10.85, 5.4, "Khối mô hình GNN", fontsize=14, fontweight="bold", ha="center")
    save(figure, "h_mohinh_tongquat.png")


def main():
    os.makedirs(FIGURE_DIR, exist_ok=True)
    with open(os.path.join(RESULT_DIR, "metrics.json"), encoding="utf-8") as file:
        metrics = json.load(file)
    plot_architecture()
    plot_class_distribution(metrics["dataset"])
    plot_window_drift(metrics["dataset"])
    plot_offline_comparison(metrics["offline"])
    plot_per_class(metrics["offline"], metrics["streaming"])
    plot_confusion(metrics["offline"]["E-GraphSAGE"], "h_confusion_offline.png")
    plot_confusion(metrics["streaming"]["Streaming E-GraphSAGE"], "h_confusion_stream.png")
    plot_stream_timeline(metrics["streaming_windows"])
    plot_latency(metrics["streaming_windows"])
    plot_graph_size(metrics["streaming_windows"])
    merged = {
        group: {name: merged_dos_metrics(result["confusion_matrix"]) for name, result in metrics[group].items()}
        for group in ["offline", "streaming"]
    }
    with open(os.path.join(RESULT_DIR, "metrics_gop_dos.json"), "w", encoding="utf-8") as file:
        json.dump(merged, file, ensure_ascii=False, indent=2)
    for group, rows in merged.items():
        for name, values in rows.items():
            print(group, name, round(values["accuracy"], 4), round(values["macro_f1"], 4), round(values["weighted_f1"], 4))


if __name__ == "__main__":
    main()
