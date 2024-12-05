import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader, random_split
import os
from PIL import Image
import numpy as np

# Đường dẫn đến dữ liệu
data_dir = "handModel/output_dataset/train"  # Đường dẫn tới thư mục train

# Cấu hình thiết bị
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Tăng cường dữ liệu (Data Augmentation)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),  # Lật ngang
    transforms.RandomRotation(15),          # Xoay nhẹ
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),  # Thay đổi màu
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Dataset tùy chỉnh
class HandDirectionDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.data = []
        
        # Duyệt qua từng thư mục góc quay
        for angle_folder in os.listdir(root_dir):
            angle_path = os.path.join(root_dir, angle_folder)
            if not os.path.isdir(angle_path):
                continue
            try:
                angle = float(angle_folder.split('_')[0])
                for img_name in os.listdir(angle_path):
                    img_path = os.path.join(angle_path, img_name)
                    if os.path.isfile(img_path):
                        self.data.append((img_path, angle))
            except ValueError:
                continue
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img_path, angle = self.data[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        
        # Chuyển góc quay thành chỉ số
        label_angle = int(angle / 22.5) % 16  # Chia thành 16 góc quay
        return image, label_angle

# Tạo dataset và chia thành train/validation
full_data = HandDirectionDataset(root_dir=data_dir, transform=transform)
train_size = int(0.8 * len(full_data))
val_size = len(full_data) - train_size
train_data, val_data = random_split(full_data, [train_size, val_size])

train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
val_loader = DataLoader(val_data, batch_size=32, shuffle=False)

# Số lớp cho góc quay
num_angles = 16

# Định nghĩa mô hình
class AngleOnlyModel(nn.Module):
    def __init__(self, num_angles):
        super(AngleOnlyModel, self).__init__()
        # Sử dụng ResNet-50 với trọng số tiền huấn luyện
        self.base_model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        in_features = self.base_model.fc.in_features
        self.base_model.fc = nn.Sequential(
            nn.Dropout(0.5),  # Dropout để tránh overfitting
            nn.Linear(in_features, num_angles)
        )

    def forward(self, x):
        return self.base_model(x)

# Khởi tạo mô hình
model = AngleOnlyModel(num_angles=num_angles).to(device)

# Circular Smooth Label (CSL)
def create_circular_smooth_label(num_classes, angle, window_size=1):
    label = np.zeros(num_classes)
    angle_idx = int(angle / 360 * num_classes)
    for offset in range(-window_size, window_size + 1):
        idx = (angle_idx + offset) % num_classes
        label[idx] = 1.0 - abs(offset) / (window_size + 1)
    return label / label.sum()

# Hàm mất mát
class CSLoss(nn.Module):
    def __init__(self):
        super(CSLoss, self).__init__()

    def forward(self, outputs, targets):
        return -torch.mean(targets * torch.log_softmax(outputs, dim=1))

criterion = CSLoss()
optimizer = optim.AdamW(model.parameters(), lr=0.001)  # AdamW Optimizer
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)  # Giảm LR sau mỗi 5 epoch

# Huấn luyện mô hình
num_epochs = 20
best_val_loss = float('inf')
patience = 5
epochs_no_improve = 0

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    # Huấn luyện
    for inputs, labels_angle in train_loader:
        inputs, labels_angle = inputs.to(device), labels_angle.to(device)
        angles = [label * 22.5 for label in labels_angle]
        labels_angle_csl = torch.tensor(
            [create_circular_smooth_label(num_angles, angle) for angle in angles],
            dtype=torch.float32).to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels_angle_csl)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    # Kiểm tra trên tập validation
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for inputs, labels_angle in val_loader:
            inputs, labels_angle = inputs.to(device), labels_angle.to(device)
            angles = [label * 22.5 for label in labels_angle]
            labels_angle_csl = torch.tensor(
                [create_circular_smooth_label(num_angles, angle) for angle in angles],
                dtype=torch.float32).to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels_angle_csl)
            val_loss += loss.item()

    val_loss /= len(val_loader)
    print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {running_loss/len(train_loader):.4f}, Val Loss: {val_loss:.4f}")

    # Early Stopping
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        epochs_no_improve = 0
        torch.save(model.state_dict(), "best_hand_angle_model.pth")  # Lưu mô hình tốt nhất
    else:
        epochs_no_improve += 1
        if epochs_no_improve == patience:
            print("Early stopping triggered.")
            break

    scheduler.step()  # Cập nhật learning rate

print("Huấn luyện hoàn tất. Mô hình tốt nhất đã được lưu.")
