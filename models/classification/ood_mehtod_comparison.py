import os
import torch
import numpy as np
from PIL import Image
from torchvision import transforms, models
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import torch.nn.functional as F

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载分类模型（ID vs OOD）
model = models.resnet18(pretrained=False)
model.fc = torch.nn.Linear(model.fc.in_features, 2)
model.load_state_dict(torch.load("/kaggle/input/binary-model/best_model.pth", map_location=device))
model.to(device)
model.eval()

# 加载特征提取器（用于RCI）
feature_extractor = models.resnet18(pretrained=True)
feature_extractor.fc = torch.nn.Identity()
feature_extractor.to(device)
feature_extractor.eval()

# 图像预处理
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# 数据路径
id_dir = "/kaggle/input/binary-id-ood/ood_data/id"
ood_dir = "/kaggle/input/binary-id-ood/ood_data/ood"
TEMPERATURE = 1000
EPSILON = 0.001

# 分数记录
scores_msp, scores_odin, scores_energy, scores_rci, labels = [], [], [], [], []

# ODIN扰动函数
def perturb_input(input_tensor, epsilon=EPSILON):
    input_tensor.requires_grad = True
    output = model(input_tensor)
    pred = output.argmax(dim=1)
    loss = torch.nn.CrossEntropyLoss()(output, pred)
    loss.backward()
    gradient = input_tensor.grad.data
    perturbed = input_tensor - epsilon * torch.sign(gradient)
    return perturbed.detach()

# RCI计算函数
def compute_rci_score(img_pil, patch_size=64, margin=10):
    w, h = img_pil.size
    center_crop = img_pil.crop((
        w//2 - patch_size//2, h//2 - patch_size//2,
        w//2 + patch_size//2, h//2 + patch_size//2
    ))
    background_crop = img_pil.crop((
        margin, margin,
        margin + patch_size, margin + patch_size
    ))

    input_c = transform(center_crop).unsqueeze(0).to(device)
    input_b = transform(background_crop).unsqueeze(0).to(device)

    with torch.no_grad():
        feat_c = F.normalize(feature_extractor(input_c), dim=1)
        feat_b = F.normalize(feature_extractor(input_b), dim=1)

    cos_sim = F.cosine_similarity(feat_c, feat_b).item()
    return 1 - cos_sim  # 越大表示越 OOD

# 图像处理函数
def process_image(img_path, label):
    image = Image.open(img_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0).to(device)

    # MSP
    with torch.no_grad():
        logits = model(input_tensor)
        prob = torch.softmax(logits, dim=1)
        score_msp = -torch.max(prob).item()
        scores_msp.append(score_msp)

    # ODIN
    input_perturbed = perturb_input(input_tensor.clone())
    with torch.no_grad():
        logits_odin = model(input_perturbed) / TEMPERATURE
        prob_odin = torch.softmax(logits_odin, dim=1)
        score_odin = -torch.max(prob_odin).item()
        scores_odin.append(score_odin)

    # Energy
    with torch.no_grad():
        energy = -torch.logsumexp(logits, dim=1).item()
        scores_energy.append(energy)

    # RCI
    rci = compute_rci_score(image)
    scores_rci.append(rci)

    labels.append(label)

# 执行图像处理
for fname in os.listdir(id_dir):
    if fname.lower().endswith((".png", ".jpg", ".jpeg")):
        process_image(os.path.join(id_dir, fname), 0)

for fname in os.listdir(ood_dir):
    if fname.lower().endswith((".png", ".jpg", ".jpeg")):
        process_image(os.path.join(ood_dir, fname), 1)

# 转为 numpy
labels = np.array(labels)
msp = np.array(scores_msp)
odin = np.array(scores_odin)
energy = np.array(scores_energy)
rci = np.array(scores_rci)

# AUROC
fpr_msp, tpr_msp, _ = roc_curve(labels, msp)
fpr_odin, tpr_odin, _ = roc_curve(labels, odin)
fpr_energy, tpr_energy, _ = roc_curve(labels, energy)
fpr_rci, tpr_rci, _ = roc_curve(labels, rci)

auroc_msp = auc(fpr_msp, tpr_msp)
auroc_odin = auc(fpr_odin, tpr_odin)
auroc_energy = auc(fpr_energy, tpr_energy)
auroc_rci = auc(fpr_rci, tpr_rci)

# 绘图
plt.figure(figsize=(8, 6))
plt.plot(fpr_msp, tpr_msp, label=f"MSP (AUROC={auroc_msp:.3f})")
plt.plot(fpr_odin, tpr_odin, label=f"ODIN (AUROC={auroc_odin:.3f})")
plt.plot(fpr_energy, tpr_energy, label=f"Energy (AUROC={auroc_energy:.3f})")
plt.plot(fpr_rci, tpr_rci, label=f"RCI (AUROC={auroc_rci:.3f})")
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("AUROC Curve Comparison (MSP/ODIN/Energy/RCI)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("auroc_comparison_with_rci.png")
plt.show()
