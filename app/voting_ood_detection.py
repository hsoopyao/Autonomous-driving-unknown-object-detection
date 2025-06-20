import gradio as gr
import cv2
import torch
import numpy as np
from torchvision import transforms, models
from ultralytics import YOLO
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity

# 加载模型
model = YOLO("best.pt")  # 你的YOLO模型路径
cls_model = models.resnet18(pretrained=True)
cls_model.fc = torch.nn.Linear(cls_model.fc.in_features, 128)  # 提取特征用
cls_model.eval()

SOILING_CLASSES = [5, 6, 7]
TRAFFIC_CLASSES = [0, 1, 2, 3, 4]
names = model.names

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cls_model.to(device)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


def extract_feature(image_np):
    image_pil = Image.fromarray(cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB))
    input_tensor = transform(image_pil).unsqueeze(0).to(device)
    with torch.no_grad():
        feat = cls_model(input_tensor)
    return feat.squeeze().cpu().numpy()


def compute_rci_score(obj_feat, neigh_feat):
    sim = cosine_similarity([obj_feat], [neigh_feat])[0][0]
    return 1 - sim  # 不一致得分


def two_stage_detect(image):
    orig = image.copy()
    results1 = model(orig)[0]
    masked = orig.copy()

    candidate_boxes = []

    for box in results1.boxes:
        cls = int(box.cls.item())
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        if cls in SOILING_CLASSES:
            cv2.rectangle(masked, (x1, y1), (x2, y2), (0, 0, 0), -1)
        candidate_boxes.append((x1, y1, x2, y2, cls, float(box.conf.item())))

    results2 = model(masked)[0]
    for box in results2.boxes:
        cls = int(box.cls.item())
        if cls in TRAFFIC_CLASSES:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            candidate_boxes.append((x1, y1, x2, y2, cls, float(box.conf.item())))

    combined = orig.copy()

    for x1, y1, x2, y2, cls, conf in candidate_boxes:
        crop = orig[y1:y2, x1:x2]
        if crop.size == 0:
            continue
        obj_feat = extract_feature(crop)

        # 生成邻域区域
        h, w = orig.shape[:2]
        expand = 0.25
        x1n = max(0, int(x1 - (x2 - x1) * expand))
        y1n = max(0, int(y1 - (y2 - y1) * expand))
        x2n = min(w, int(x2 + (x2 - x1) * expand))
        y2n = min(h, int(y2 + (y2 - y1) * expand))
        neigh_crop = orig[y1n:y2n, x1n:x2n]
        neigh_feat = extract_feature(neigh_crop)

        rci_score = compute_rci_score(obj_feat, neigh_feat)
        is_ood = rci_score > 0.3

        if cls in SOILING_CLASSES:
            label = f"Soiling ({rci_score:.2f})" if is_ood else f"ID:{names[cls]}"
            color = (0, 0, 255) if is_ood else (0, 128, 255)
        else:
            label = f"Anomaly ({rci_score:.2f})" if is_ood else f"ID:{names[cls]}"
            color = (0, 165, 255) if is_ood else (0, 255, 0)

        cv2.rectangle(combined, (x1, y1), (x2, y2), color, 2)
        cv2.putText(combined, label, (x1, max(y1 - 4, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    return combined


# Gradio UI
demo = gr.Interface(
    fn=two_stage_detect,
    inputs=gr.Image(type="numpy", label="上传图像（带或不带污渍）"),
    outputs=gr.Image(type="numpy", label="检测结果（污渍+交通元素+RCI）"),
    title="YOLOv8 + RCI 区域对比一致性 OOD 检测",
    description="使用两阶段 YOLO 检测与 RCI 一致性评分，区分正常交通目标与污渍/异常目标。"
)

demo.launch()