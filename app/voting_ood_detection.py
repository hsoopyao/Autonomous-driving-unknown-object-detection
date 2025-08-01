import gradio as gr
import cv2
import torch
import numpy as np
from torchvision import transforms, models
from ultralytics import YOLO
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity

# 加载模型
model = YOLO("../models/yolov8/yolov0_train80/best.pt")
cls_model = models.resnet18(pretrained=False)
cls_model.fc = torch.nn.Linear(cls_model.fc.in_features, 2)
cls_model.load_state_dict(torch.load("../models/classification/result/classification_model.pth", map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu")))
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

TEMPERATURE = 1000
EPSILON = 0.001

RCI_LOW = 0.25
RCI_HIGH = 0.35
ODIN_THRESHOLD = -0.5
ENERGY_THRESHOLD = -12.0

ODIN_WEIGHT = 1.0
ENERGY_WEIGHT = 0.5
FUSED_SCORE_THRESHOLD = 1.2

def extract_feature(image_np):
    image_pil = Image.fromarray(cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB))
    input_tensor = transform(image_pil).unsqueeze(0).to(device)
    with torch.no_grad():
        feat = cls_model(input_tensor)
    return feat.squeeze().cpu().numpy(), input_tensor

def compute_rci_score(obj_feat, neigh_feat):
    sim = cosine_similarity([obj_feat], [neigh_feat])[0][0]
    return 1 - sim

def perturb_input(input_tensor):
    input_tensor.requires_grad = True
    output = cls_model(input_tensor)
    pred = output.argmax(dim=1)
    loss = torch.nn.CrossEntropyLoss()(output, pred)
    loss.backward()
    gradient = input_tensor.grad.data
    perturbed = input_tensor - EPSILON * torch.sign(gradient)
    return perturbed.detach()

def compute_uncertainty_scores(input_tensor):
    with torch.no_grad():
        logits = cls_model(input_tensor)
        energy_score = -torch.logsumexp(logits, dim=1).item()
    perturbed_tensor = perturb_input(input_tensor.clone())
    with torch.no_grad():
        logits_odin = cls_model(perturbed_tensor) / TEMPERATURE
        probs_odin = torch.softmax(logits_odin, dim=1)
        odin_score = -torch.max(probs_odin).item()
    return odin_score, energy_score

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
        obj_feat, input_tensor = extract_feature(crop)

        h, w = orig.shape[:2]
        expand = 0.25
        x1n = max(0, int(x1 - (x2 - x1) * expand))
        y1n = max(0, int(y1 - (y2 - y1) * expand))
        x2n = min(w, int(x2 + (x2 - x1) * expand))
        y2n = min(h, int(y2 + (y2 - y1) * expand))
        neigh_crop = orig[y1n:y2n, x1n:x2n]
        neigh_feat, _ = extract_feature(neigh_crop)

        rci_score = compute_rci_score(obj_feat, neigh_feat)

        if rci_score > RCI_HIGH:
            is_ood = True
            fused_score = rci_score
        elif rci_score < RCI_LOW:
            is_ood = False
            fused_score = rci_score
        else:
            odin_score, energy_score = compute_uncertainty_scores(input_tensor)
            fused_score = (
                ODIN_WEIGHT * max(0, odin_score - ODIN_THRESHOLD) +
                ENERGY_WEIGHT * max(0, energy_score - ENERGY_THRESHOLD)
            )
            is_ood = fused_score > FUSED_SCORE_THRESHOLD

        if cls in SOILING_CLASSES:
            label = f"Soiling OOD ({fused_score:.2f})" if is_ood else f"ID:{names[cls]}"
            color = (0, 0, 255) if is_ood else (0, 128, 255)
        else:
            label = f"Anomaly ({fused_score:.2f})" if is_ood else f"ID:{names[cls]}"
            color = (0, 165, 255) if is_ood else (0, 255, 0)

        cv2.rectangle(combined, (x1, y1), (x2, y2), color, 2)
        cv2.putText(combined, label, (x1, max(y1 - 4, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    return combined

# Gradio UI
demo = gr.Interface(
    fn=two_stage_detect,
    inputs=gr.Image(type="numpy", label="Upload Image (with or without lens soiling)"),
    outputs=gr.Image(type="numpy", label="Detection Result (Soiling + Traffic Elements + RCI-Based Fusion)"),
    title="YOLOv8 + RCI Primary + ODIN/Energy Assisted Soft Fusion for OOD Detection",
    description="RCI is the primary decision metric; ODIN and Energy estimators are conditionally activated only when the consistency score falls within the ambiguous interval [0.25, 0.35], enhancing robustness and accuracy."
)

demo.launch()