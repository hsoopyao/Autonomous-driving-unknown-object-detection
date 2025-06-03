import gradio as gr
import cv2
import torch
import numpy as np
from PIL import Image
from torchvision import transforms, models
from ultralytics import YOLO

# 设备配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载 YOLO 模型
model = YOLO("best.pt")  # 替换为你的路径
names = model.names
SOILING_CLASSES = [5, 6, 7]  # 污渍类别

# 加载二分类模型（0: ID, 1: OOD）
cls_model = models.resnet18(pretrained=False)
cls_model.fc = torch.nn.Linear(cls_model.fc.in_features, 2)
cls_model.load_state_dict(torch.load("best_model.pth", map_location=device))
cls_model.to(device)
cls_model.eval()

# 图像预处理
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

def yolo_with_classification(image):
    orig = image.copy()
    masked = image.copy()
    candidates = []

    # 第1阶段 YOLO 检测（原图，检测污渍 + 交通）
    results1 = model(orig)[0]
    for box in results1.boxes:
        cls = int(box.cls.item())
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        if cls in SOILING_CLASSES:
            cv2.rectangle(masked, (x1, y1), (x2, y2), (0, 0, 0), -1)
        crop = orig[y1:y2, x1:x2]
        if crop.size != 0:
            candidates.append((crop, (x1, y1, x2, y2), cls))

    # 第2阶段 YOLO 检测（masked 图，检测被遮住的交通目标）
    results2 = model(masked)[0]
    for box in results2.boxes:
        cls = int(box.cls.item())
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        crop = orig[y1:y2, x1:x2]
        if crop.size != 0:
            candidates.append((crop, (x1, y1, x2, y2), cls))

    # 图像可视化
    vis = orig.copy()
    for crop, (x1, y1, x2, y2), cls_id in candidates:
        crop_pil = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        input_tensor = transform(crop_pil).unsqueeze(0).to(device)

        with torch.no_grad():
            output = cls_model(input_tensor)
            probs = torch.softmax(output, dim=1)
            conf, pred = torch.max(probs, dim=1)

        if pred.item() == 0:  # ID
            label = f"ID: {names[cls_id]} {conf.item():.2f}"
            color = (0, 255, 0)  # Green
        else:  # OOD
            label = f"OOD {conf.item():.2f}"
            color = (0, 0, 255)  # Red

        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
        cv2.putText(vis, label, (x1, max(y1 - 6, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    return vis


# Gradio UI
demo = gr.Interface(
    fn=yolo_with_classification,
    inputs=gr.Image(type="numpy", label="上传图像"),
    outputs=gr.Image(type="numpy", label="检测结果（ID + 类别 / OOD）"),
    title="YOLOv8 + 二分类模型：统一输出 ID（含类别）/ OOD",
    description="YOLO 两阶段检测所有目标，统一交给二分类模型识别是否为 OOD。ID 显示类别，OOD 只标注类型。"
)

demo.launch()
