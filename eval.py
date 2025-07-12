import torch
import matplotlib.pyplot as plt
import numpy as np
import cv2
from timm import create_model
from datasets import LidarImageDetectionDataset
import torchvision.transforms as T
from models_mamba import VisionMamba

# === 设置设备 ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === 加载模型 ===
checkpoint_path = './output/test/checkpoint.pth'
model = create_model('test', pretrained=False, num_classes=1, img_size=(1280, 960))
checkpoint = torch.load(checkpoint_path, map_location='cpu')
model.load_state_dict(checkpoint['model'], strict=False)
model = model.to(device).eval()

# === 加载数据集 ===
dataset = LidarImageDetectionDataset(
    root='E:\\CCFA\\project\\dataset\\Dataset4Mamba\\val\\',
    transform=T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]),
    lidar_transform=T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.5], std=[0.5])
    ]),
    img_size=(1280, 960)
)

# === 工具函数 ===
def denormalize_img(tensor):
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    return (tensor * std[:, None, None] + mean[:, None, None]).clip(0, 1)

def visualize(img_tensor, pred_boxes, pred_scores, gt_boxes=None, threshold=0.5):
    img = denormalize_img(img_tensor.cpu()).permute(1, 2, 0).numpy()
    img = (img * 255).astype(np.uint8)

    h, w, _ = img.shape
    img_draw = img.copy()

    # 画预测框
    for box, score in zip(pred_boxes, pred_scores):
        if score < threshold:
            continue
        cx, cy, bw, bh = box
        cx, cy, bw, bh = int(cx * w), int(cy * h), int(bw * w), int(bh * h)
        x1, y1, x2, y2 = cx - bw // 2, cy - bh // 2, cx + bw // 2, cy + bh // 2
        cv2.rectangle(img_draw, (x1, y1), (x2, y2), (0, 255, 0), 2)  # 绿色：预测框
        cv2.putText(img_draw, f"{score:.2f}", (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

    # 画GT框
    if gt_boxes is not None:
        for box in gt_boxes:
            cx, cy, bw, bh = box
            cx, cy, bw, bh = int(cx * w), int(cy * h), int(bw * w), int(bh * h)
            x1, y1, x2, y2 = cx - bw // 2, cy - bh // 2, cx + bw // 2, cy + bh // 2
            cv2.rectangle(img_draw, (x1, y1), (x2, y2), (255, 0, 0), 2)  # 红色：GT框

    plt.figure(figsize=(12, 8))
    plt.imshow(img_draw)
    plt.axis('off')
    plt.show()

# === 开始评估 ===
n_samples = 5
for idx in range(n_samples):
    img, lidar_img, target = dataset[idx]
    img_input = img.unsqueeze(0).to(device)
    lidar_input = lidar_img.unsqueeze(0).to(device)

    with torch.no_grad():
        predictions = model(img_input, lidar_input)  # 只返回 predictions

    pred_logits = predictions['pred_logits'][0]  # (num_queries, num_classes+1)
    pred_boxes = predictions['pred_boxes'][0]    # (num_queries, 4)

    pred_scores = pred_logits.softmax(-1)[:, 0]  # 因为只有1类（0类是目标）
    pred_labels = pred_scores > 0.5  # 置信度阈值筛选

    print(f"\n🧩 Sample {idx}")
    for i in range(pred_logits.shape[0]):
        print(f"Query {i:02d} | Score: {pred_scores[i]:.3f} | Box: {pred_boxes[i].cpu().numpy()}")

    # 只保留分数高的
    selected_boxes = pred_boxes[pred_labels]
    selected_scores = pred_scores[pred_labels]

    visualize(img, selected_boxes.cpu().numpy(), selected_scores.cpu().numpy(), gt_boxes=target['boxes'].cpu().numpy())
