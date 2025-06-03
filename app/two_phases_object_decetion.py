import gradio as gr
import cv2
import torch
import numpy as np
from ultralytics import YOLO

# 加载模型
model = YOLO("best.pt")  # 你的模型路径
SOILING_CLASSES = [5, 6, 7]
TRAFFIC_CLASSES = [0, 1, 2, 3, 4]
names = model.names


# 核心处理函数
def two_stage_detect(image):
    orig = image.copy()

    # 第1阶段检测
    results1 = model(orig)[0]
    masked = orig.copy()

    for box in results1.boxes:
        cls = int(box.cls.item())
        if cls in SOILING_CLASSES:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            cv2.rectangle(masked, (x1, y1), (x2, y2), (0, 0, 0), -1)  # 遮挡污渍

    # 第2阶段检测
    results2 = model(masked)[0]
    combined = orig.copy()

    # 第一阶段：红=污渍，橙=交通
    for box in results1.boxes:
        cls = int(box.cls.item())
        conf = float(box.conf.item())
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        color = (0, 0, 255) if cls in SOILING_CLASSES else (255, 165, 0)
        label = f"{names[cls]} {conf:.2f}"
        cv2.rectangle(combined, (x1, y1), (x2, y2), color, 2)
        cv2.putText(combined, label, (x1, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    # 第二阶段：绿色框为新发现的交通目标
    for box in results2.boxes:
        cls = int(box.cls.item())
        conf = float(box.conf.item())
        if cls in TRAFFIC_CLASSES:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            label = f"{names[cls]} {conf:.2f}"
            cv2.rectangle(combined, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(combined, label, (x1, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    return combined


# Gradio UI
demo = gr.Interface(
    fn=two_stage_detect,
    inputs=gr.Image(type="numpy", label="上传图像（带或不带污渍）"),
    outputs=gr.Image(type="numpy", label="检测结果（污渍+交通元素）"),
    title="YOLOv8 两阶段污渍与交通元素检测",
    description="支持同时检测图像中的污渍（红框）和交通目标（橙/绿框）。适用于 Woodscape 数据集中的 soiling 图像。"
)

demo.launch()
