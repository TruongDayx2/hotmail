import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
from PIL import Image

# Đường dẫn đến output dataset đã chia
data_dir = "output_dataset/train"  # Đường dẫn tới thư mục train

# Cấu hình thiết bị (GPU nếu có sẵn)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Chuẩn bị tăng cường dữ liệu
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomApply([
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.GaussianBlur(kernel_size=(5, 5), sigma=(0.1, 2.0))
    ], p=0.5),
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Dataset tùy chỉnh
class ObjectDirectionDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.data = []
        
        # Duyệt qua từng thư mục loại đối tượng
        for object_class in os.listdir(root_dir):
            object_class_path = os.path.join(root_dir, object_class)
            
            # Duyệt qua từng thư mục góc quay
            for angle_folder in os.listdir(object_class_path):
                angle_path = os.path.join(object_class_path, angle_folder)
                angle = float(angle_folder.split('_')[0])  # Lấy giá trị góc từ tên thư mục
                
                # Duyệt qua từng ảnh trong thư mục góc quay
                for img_name in os.listdir(angle_path):
                    img_path = os.path.join(angle_path, img_name)
                    self.data.append((img_path, object_class, angle))
        
        # Tạo danh sách nhãn loại đối tượng
        self.classes = sorted(list(set([item[1] for item in self.data])))
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img_path, object_class, angle = self.data[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        
        # Chuyển loại đối tượng thành chỉ số
        label_class = self.class_to_idx[object_class]
        
        # Chuyển góc quay thành chỉ số
        label_angle = int(angle / 22.5) % 16  # Chia thành 16 góc quay (mỗi góc cách nhau 22.5 độ)
        
        return image, label_class, label_angle

# Tạo dataset và dataloader
train_data = ObjectDirectionDataset(root_dir=data_dir, transform=transform)
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)

# Số lớp cho loại đối tượng và góc quay
num_classes = len(train_data.classes)  # Loại đối tượng
num_angles = 16  # 16 góc quay, mỗi góc cách nhau 22.5 độ

# Định nghĩa mô hình hai đầu ra với trọng số đã huấn luyện trước đó
class DualOutputModel(nn.Module):
    def __init__(self, num_classes, num_angles):
        super(DualOutputModel, self).__init__()
        # Khởi tạo mô hình ResNet18 mà không tải lớp fully connected
        self.base_model = models.resnet18(weights=None)
        
        # Tải trọng số từ object_classification_model.pth, bỏ qua lớp fully connected
        checkpoint = torch.load("object_classification_model.pth")
        filtered_checkpoint = {k: v for k, v in checkpoint.items() if not k.startswith("fc.")}
        self.base_model.load_state_dict(filtered_checkpoint, strict=False)

        in_features = self.base_model.fc.in_features
        self.base_model.fc = nn.Identity()  # Loại bỏ lớp fully connected ban đầu

        # Đầu ra cho loại đối tượng
        self.fc_class = nn.Linear(in_features, num_classes)
        
        # Đầu ra cho góc quay
        self.fc_angle = nn.Linear(in_features, num_angles)

    def forward(self, x):
        x = self.base_model(x)
        class_output = self.fc_class(x)
        angle_output = self.fc_angle(x)
        return class_output, angle_output

# Khởi tạo mô hình và chuyển sang thiết bị
model = DualOutputModel(num_classes=num_classes, num_angles=num_angles).to(device)

# Tạo Circular Smooth Label (CSL) cho đầu ra góc quay
def create_circular_smooth_label(num_classes, angle, window_size=1):
    label = np.zeros(num_classes)
    angle_idx = int(angle / 360 * num_classes)
    for offset in range(-window_size, window_size + 1):
        idx = (angle_idx + offset) % num_classes
        label[idx] = 1.0 - abs(offset) / (window_size + 1)
    return label / label.sum()

# Định nghĩa hàm mất mát
class CSLoss(nn.Module):
    def __init__(self):
        super(CSLoss, self).__init__()

    def forward(self, outputs, targets):
        return -torch.mean(targets * torch.log_softmax(outputs, dim=1))

criterion_class = nn.CrossEntropyLoss()  # Mất mát cho loại đối tượng
criterion_angle = CSLoss()  # Mất mát cho góc quay với CSL

optimizer = optim.Adam(model.parameters(), lr=0.001)
num_epochs = 10

# Huấn luyện mô hình hai đầu ra
for epoch in range(num_epochs):
    model.train()
    running_loss_class = 0.0
    running_loss_angle = 0.0

    for inputs, labels_class, labels_angle in train_loader:
        inputs = inputs.to(device)
        labels_class = labels_class.to(device)
        
        # Chuyển đổi nhãn góc quay thành CSL
        angles = [label * 22.5 for label in labels_angle]
        labels_angle_csl = torch.tensor([create_circular_smooth_label(num_angles, angle) for angle in angles], dtype=torch.float32).to(device)
        
        optimizer.zero_grad()
        
        # Dự đoán loại đối tượng và góc quay
        outputs_class, outputs_angle = model(inputs)
        
        # Tính toán mất mát
        loss_class = criterion_class(outputs_class, labels_class)
        loss_angle = criterion_angle(outputs_angle, labels_angle_csl)
        loss = loss_class + loss_angle  # Tổng hợp hai loại mất mát
        
        # Backward và tối ưu hóa
        loss.backward()
        optimizer.step()
        
        running_loss_class += loss_class.item()
        running_loss_angle += loss_angle.item()

    print(f"Epoch [{epoch+1}/{num_epochs}], Class Loss: {running_loss_class/len(train_loader):.4f}, Angle Loss: {running_loss_angle/len(train_loader):.4f}")

# Lưu mô hình hai đầu ra đã huấn luyện
torch.save(model.state_dict(), "dual_output_model.pth")
print("Mô hình hai đầu ra đã được lưu thành công dưới dạng 'dual_output_model.pth'")
