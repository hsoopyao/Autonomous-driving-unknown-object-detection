import os
import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# --------------------------
# 设置设备
# --------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# --------------------------
# 图像预处理
# --------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# --------------------------
# 加载数据集
# --------------------------
data_dir = "/kaggle/input/binary-id-ood"
train_dataset = datasets.ImageFolder(root=data_dir, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# --------------------------
# 构建模型
# --------------------------
model = models.resnet18(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, 2)  # 二分类输出
model = model.to(device)

# --------------------------
# 设置损失函数和优化器
# --------------------------
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# 记录loss和acc
loss_history = []
acc_history = []

# --------------------------
# 训练模型
# --------------------------
epochs = 30
for epoch in range(epochs):
    model.train()
    total_loss = 0
    all_preds, all_labels = [], []

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        all_preds.extend(outputs.argmax(dim=1).cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(train_loader)
    acc = accuracy_score(all_labels, all_preds)
    loss_history.append(avg_loss)
    acc_history.append(acc)

    print(f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f} | Acc: {acc:.4f}")

# --------------------------
# 保存模型（建议保存整个模型，或同时保存结构）
# --------------------------
torch.save(model.state_dict(), "binary_stain_classifier.pt")
print("模型已保存为 binary_stain_classifier.pt")

# --------------------------
# 可视化 Loss 和 Accuracy
# --------------------------
plt.figure(figsize=(10, 4))

# Loss 曲线
plt.subplot(1, 2, 1)
plt.plot(range(1, epochs + 1), loss_history, label='Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.grid(True)

# Accuracy 曲线
plt.subplot(1, 2, 2)
plt.plot(range(1, epochs + 1), acc_history, label='Accuracy', color='green')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training Accuracy')
plt.grid(True)

plt.tight_layout()
plt.savefig('training_metrics.png')  # 保存为图片
plt.show()
