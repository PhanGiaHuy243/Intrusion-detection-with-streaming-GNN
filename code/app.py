import json
import os
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import streamlit as st
import torch
from sklearn.metrics import accuracy_score, f1_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_stream import CLASS_NAMES, StreamScaler, build_graph, load_flows, stream_windows
from models import FlowMLP, StreamingEGraphSAGE
from run_experiments import class_weights, predict, set_seed, train_steps

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
os.chdir(ROOT)
RESULT_DIR = "ketqua"
FIGURE_DIR = os.path.join(RESULT_DIR, "hinh")
RERUN_DIR = "ketqua_chay_lai"
DOC_FILES = [
    ("Báo cáo đồ án (PDF)", "baocao/BaoCao_BigData_StreamingGNN.pdf", "application/pdf"),
    ("Báo cáo đồ án (Word)", "baocao/BaoCao_BigData_StreamingGNN.docx", "application/vnd.openxmlformats-officedocument.wordprocessingml.document"),
    ("Bài báo (PDF)", "baocao/BaiBao_BigData_StreamingGNN.pdf", "application/pdf"),
    ("Bài báo (Word)", "baocao/BaiBao_BigData_StreamingGNN.docx", "application/vnd.openxmlformats-officedocument.wordprocessingml.document"),
]

st.set_page_config(page_title="Streaming GNN – Phát hiện xâm nhập", layout="wide")


@st.cache_data
def load_metrics(folder):
    path = os.path.join(folder, "metrics.json")
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as file:
        return json.load(file)


@st.cache_data
def load_flows_cached():
    return load_flows()


def fmt(value, digits=4):
    return f"{value:.{digits}f}".replace(".", ",")


def pct(value):
    return f"{value * 100:.2f}%".replace(".", ",")


def show_figure(name, caption, width=None):
    path = os.path.join(FIGURE_DIR, name)
    if os.path.exists(path):
        if width:
            st.image(path, caption=caption, width=width)
        else:
            st.image(path, caption=caption, use_container_width=True)


def result_table(results, streaming=False):
    rows = []
    for name, result in results.items():
        row = {
            "Mô hình": name,
            "Accuracy": fmt(result["accuracy"]),
            "Macro-F1": fmt(result["macro_f1"]),
            "Weighted-F1": fmt(result["weighted_f1"]),
            "Tỉ lệ phát hiện (DR)": pct(result["detection_rate"]),
            "Cảnh báo nhầm (FAR)": pct(result["false_alarm_rate"]),
        }
        if streaming:
            row["Độ trễ (ms)"] = fmt(result["mean_latency_ms"], 1)
            row["Luồng/giây"] = f"{int(result['mean_throughput']):,}".replace(",", ".")
        rows.append(row)
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)


def page_overview(metrics):
    st.title("Phát hiện xâm nhập mạng bằng GNN trên dữ liệu luồng")
    st.caption("Đồ án cuối kì – Giới thiệu về dữ liệu lớn · Intrusion Detection with Streaming GNN")
    dataset = metrics["dataset"]
    offline = metrics["offline"]["E-GraphSAGE"]
    stream = metrics["streaming"]["Streaming E-GraphSAGE"]
    columns = st.columns(4)
    columns[0].metric("Số luồng NetFlow", f"{dataset['num_flows']:,}".replace(",", "."))
    columns[1].metric("Macro-F1 E-GraphSAGE", fmt(offline["macro_f1"]))
    columns[2].metric("Weighted-F1 khi chạy luồng", fmt(stream["weighted_f1"]))
    columns[3].metric("Độ trễ / micro-batch", f"{fmt(stream['mean_latency_ms'], 1)} ms")
    st.markdown(
        "Hệ thống nhận các bản ghi **NetFlow** theo từng micro-batch 10.000 luồng, dựng **đồ thị động** "
        "(nút = IP:port, cạnh = luồng) trên cửa sổ trượt 2 micro-batch, dùng **E-GraphSAGE** để phân loại "
        "từng luồng thành *Benign, DDoS, DoS, Reconnaissance, Theft*, rồi **tự cập nhật trực tuyến** "
        "theo giao thức kiểm thử trước – huấn luyện sau."
    )
    show_figure("h_mohinh_tongquat.png", "Mô hình tổng quát của hệ thống")
    st.info("Dùng thanh bên trái để chuyển giữa các mục: Dữ liệu, Kết quả, Demo phát hiện theo luồng, Tài liệu.")


def page_data(metrics):
    st.header("Bộ dữ liệu NF-BoT-IoT")
    dataset = metrics["dataset"]
    counts = dataset["class_counts"]
    total = dataset["num_flows"]
    table = pd.DataFrame({
        "Lớp": CLASS_NAMES,
        "Số luồng": [f"{counts[name]:,}".replace(",", ".") for name in CLASS_NAMES],
        "Tỉ lệ": [pct(counts[name] / total) for name in CLASS_NAMES],
    })
    left, right = st.columns([1, 2])
    left.dataframe(table, hide_index=True, use_container_width=True)
    left.markdown(
        f"- **{dataset['num_src_ip']}** IP nguồn, **{dataset['num_dst_ip']}** IP đích\n"
        f"- **{dataset['num_endpoints']:,}** điểm cuối IP:port\n".replace(",", ".")
        + f"- **{dataset['num_windows']}** micro-batch × 10.000 luồng\n"
        "- 34,2% luồng có đặc trưng trùng nhưng khác nhãn (chủ yếu DoS/DDoS)"
    )
    with right:
        show_figure("h_phanbo_lop.png", "Phân bố số luồng theo lớp (thang log)")
    show_figure("h_troi_du_lieu.png", "Tỉ lệ các lớp theo từng micro-batch – hiện tượng trôi khái niệm")
    st.subheader("Xem thử 20 luồng đầu tiên")
    flows = load_flows_cached()
    st.dataframe(flows.drop(columns=["src_node", "dst_node", "y"]).head(20), use_container_width=True)


def page_results(metrics):
    st.header("Kết quả thực nghiệm")
    st.subheader("Thiết lập ngoại tuyến (chia ngẫu nhiên 70/30)")
    result_table(metrics["offline"])
    left, right = st.columns(2)
    with left:
        show_figure("h_sosanh_offline.png", "So sánh các mô hình")
    with right:
        show_figure("h_f1_tung_lop.png", "F1 từng lớp")
    show_figure("h_confusion_offline.png", "Ma trận nhầm lẫn E-GraphSAGE (ngoại tuyến)", width=560)
    st.subheader("Thiết lập luồng (kiểm thử trước – huấn luyện sau, 60 micro-batch)")
    result_table(metrics["streaming"], streaming=True)
    show_figure("h_f1_theo_cuaso.png", "Weighted-F1 theo từng micro-batch")
    left, right = st.columns(2)
    with left:
        show_figure("h_do_tre.png", "Độ trễ và thông lượng")
    with right:
        show_figure("h_kichthuoc_dothi.png", "Kích thước đồ thị động")


def streaming_demo(flows, window_size, steps, model_name, max_windows):
    set_seed(42)
    device = torch.device("cpu")
    windows = list(stream_windows(flows, window_size))[: max_windows + 1]
    scaler = StreamScaler().fit(windows[0][1])
    model_class = StreamingEGraphSAGE if model_name == "E-GraphSAGE" else FlowMLP
    model = model_class(len(scaler.mean), 64, len(CLASS_NAMES)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    seen = np.bincount(windows[0][1]["y"].values, minlength=len(CLASS_NAMES))
    graph = build_graph([windows[0][1]], scaler, device)
    train_steps(model, optimizer, graph, torch.ones(graph["num_edges"], dtype=torch.bool), class_weights(seen).cpu(), 60)
    memory = [windows[0][1]]
    for window_id, window in windows[1:]:
        memory = (memory + [window])[-2:]
        start = time.perf_counter()
        graph = build_graph(memory, scaler, device)
        prediction = predict(model, graph)[-len(window):].numpy()
        latency = (time.perf_counter() - start) * 1000
        truth = window["y"].values
        yield window_id, window, truth, prediction, latency, graph["num_nodes"]
        seen = seen + np.bincount(truth, minlength=len(CLASS_NAMES))
        train_steps(model, optimizer, graph, torch.ones(graph["num_edges"], dtype=torch.bool), class_weights(seen).cpu(), steps)


def page_demo():
    st.header("Demo: phát hiện xâm nhập theo luồng (chạy trực tiếp)")
    st.markdown(
        "Dữ liệu được phát lần lượt từng micro-batch như lưu lượng thật. Với mỗi micro-batch, mô hình "
        "**dự đoán trước**, hiển thị cảnh báo, rồi mới **học từ nhãn thật** để cập nhật."
    )
    controls = st.columns(4)
    model_name = controls[0].selectbox("Mô hình", ["E-GraphSAGE", "MLP (không dùng đồ thị)"])
    window_size = controls[1].selectbox("Kích thước micro-batch", [10000, 5000], index=0)
    steps = controls[2].slider("Số bước học mỗi micro-batch", 1, 30, 10)
    max_windows = controls[3].slider("Số micro-batch muốn chạy", 5, 60, 40)
    if not st.button("▶ Bắt đầu phát luồng", type="primary"):
        st.caption("Trên CPU, mỗi micro-batch mất khoảng 0,5–2 giây (gồm cả bước học).")
        return
    flows = load_flows_cached()
    status = st.empty()
    kpi = [column.empty() for column in st.columns(4)]
    chart = st.empty()
    alerts_box = st.empty()
    history = []
    all_truth, all_pred = [], []
    for window_id, window, truth, prediction, latency, nodes in streaming_demo(flows, window_size, steps, model_name, max_windows):
        all_truth.append(truth)
        all_pred.append(prediction)
        attacks = int((prediction != 0).sum())
        history.append({
            "Micro-batch": window_id,
            "Weighted-F1": f1_score(truth, prediction, average="weighted", zero_division=0),
            "Accuracy": accuracy_score(truth, prediction),
        })
        node_text = f"{nodes:,}".replace(",", ".")
        status.info(f"Đang xử lý micro-batch {window_id}/{max_windows} · đồ thị {node_text} nút · trễ {fmt(latency, 1)} ms")
        y_true = np.concatenate(all_truth)
        y_pred = np.concatenate(all_pred)
        kpi[0].metric("Luồng đã xử lý", f"{len(y_true):,}".replace(",", "."))
        kpi[1].metric("Cảnh báo tấn công (batch này)", f"{attacks:,}".replace(",", "."))
        kpi[2].metric("Weighted-F1 tích luỹ", fmt(f1_score(y_true, y_pred, average="weighted", zero_division=0)))
        kpi[3].metric("Độ trễ suy luận", f"{fmt(latency, 1)} ms")
        chart.line_chart(pd.DataFrame(history).set_index("Micro-batch"))
        sample = window.assign(**{"Dự đoán": [CLASS_NAMES[p] for p in prediction]})
        sample = sample[sample["Dự đoán"] != "Benign"].head(8)
        alerts_box.dataframe(
            sample[["IPV4_SRC_ADDR", "L4_SRC_PORT", "IPV4_DST_ADDR", "L4_DST_PORT", "IN_BYTES", "Attack", "Dự đoán"]]
            .rename(columns={"Attack": "Nhãn thật"}),
            hide_index=True, use_container_width=True,
        )
    status.success(f"Hoàn tất {max_windows} micro-batch.")
    y_true = np.concatenate(all_truth)
    y_pred = np.concatenate(all_pred)
    per_class = f1_score(y_true, y_pred, average=None, labels=range(len(CLASS_NAMES)), zero_division=0)
    st.subheader("F1 từng lớp sau khi chạy")
    st.dataframe(pd.DataFrame({"Lớp": CLASS_NAMES, "F1": [fmt(v) for v in per_class]}), hide_index=True)


def page_documents():
    st.header("Tài liệu đồ án")
    for label, path, mime in DOC_FILES:
        if os.path.exists(path):
            with open(path, "rb") as file:
                st.download_button(f"⬇ Tải {label}", file.read(), file_name=os.path.basename(path), mime=mime, key=path)
        else:
            st.warning(f"Không tìm thấy {path}")
    st.subheader("Mã nguồn chính")
    for name in ["models.py", "data_stream.py", "run_experiments.py"]:
        with st.expander(f"code/{name}"):
            with open(os.path.join("code", name), encoding="utf-8") as file:
                st.code(file.read(), language="python")


def page_rerun():
    st.header("Chạy lại toàn bộ thực nghiệm")
    st.markdown(
        "Chạy lại 4 mô hình ngoại tuyến và 4 cấu hình luồng trên toàn bộ 600.100 luồng, sau đó vẽ lại biểu đồ. "
        f"Kết quả lưu vào thư mục `{RERUN_DIR}/` (không ghi đè kết quả gốc của báo cáo). "
        "Trên CPU mất khoảng 5 phút."
    )
    if st.button("▶ Chạy lại thực nghiệm", type="primary"):
        environment = dict(os.environ, KETQUA_DIR=RERUN_DIR)
        log = st.empty()
        lines = []
        with st.spinner("Đang chạy..."):
            for script in ["code/run_experiments.py", "code/plots.py"]:
                process = subprocess.Popen([sys.executable, script], stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                           text=True, env=environment)
                for line in process.stdout:
                    if "warn" not in line.lower():
                        lines.append(line.rstrip())
                        log.code("\n".join(lines[-15:]))
                process.wait()
        load_metrics.clear()
        st.success("Hoàn tất.")
    rerun = load_metrics(RERUN_DIR)
    if rerun:
        st.subheader("Kết quả lần chạy lại gần nhất")
        result_table(rerun["offline"])
        result_table(rerun["streaming"], streaming=True)


PAGES = {
    "Tổng quan": page_overview,
    "Dữ liệu": page_data,
    "Kết quả thực nghiệm": page_results,
    "Demo phát hiện theo luồng": page_demo,
    "Tài liệu & mã nguồn": page_documents,
    "Chạy lại thực nghiệm": page_rerun,
}


def main():
    metrics = load_metrics(RESULT_DIR)
    st.sidebar.title("Streaming GNN IDS")
    choice = st.sidebar.radio("Mục", list(PAGES.keys()))
    st.sidebar.caption("Đồ án môn Giới thiệu về dữ liệu lớn")
    if choice in ("Tổng quan", "Dữ liệu", "Kết quả thực nghiệm") and metrics is None:
        st.error("Chưa có kết quả trong thư mục ketqua/. Hãy vào mục 'Chạy lại thực nghiệm'.")
        return
    page = PAGES[choice]
    if choice in ("Tổng quan", "Dữ liệu", "Kết quả thực nghiệm"):
        page(metrics)
    else:
        page()


main()
