import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
import os

# Cấu hình thiết bị
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Dataset tùy chỉnh
class MultiObjectDirectionDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.data = []
        self.classes = []

        for object_class in os.listdir(root_dir):
            object_class_path = os.path.join(root_dir, object_class)
            if not os.path.isdir(object_class_path):
                continue
            self.classes.append(object_class)
            for angle_folder in os.listdir(object_class_path):
                angle_path = os.path.join(object_class_path, angle_folder)
                if not os.path.isdir(angle_path):
                    continue
                try:
                    angle = float(angle_folder.split('_')[0])
                    for img_name in os.listdir(angle_path):
                        img_path = os.path.join(angle_path, img_name)
                        if os.path.isfile(img_path):
                            self.data.append((img_path, object_class, angle))
                except ValueError:
                    continue

        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, object_class, angle = self.data[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        label_class = self.class_to_idx[object_class]
        label_angle = int(angle / 22.5) % 16
        return image, label_class, label_angle


# Chuẩn bị dữ liệu
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
data_dir = "output_dataset/train"
dataset = MultiObjectDirectionDataset(root_dir=data_dir, transform=transform)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_data, val_data = random_split(dataset, [train_size, val_size])
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
val_loader = DataLoader(val_data, batch_size=32, shuffle=False)

# Định nghĩa mô hình
class DualOutputModel(nn.Module):
    def __init__(self, num_classes, num_angles):
        super(DualOutputModel, self).__init__()
        self.base_model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        in_features = self.base_model.fc.in_features
        self.base_model.fc = nn.Identity()
        self.fc_class = nn.Linear(in_features, num_classes)
        self.fc_angle = nn.Linear(in_features, num_angles)

    def forward(self, x):
        features = self.base_model(x)
        class_output = self.fc_class(features)
        angle_output = self.fc_angle(features)
        return class_output, angle_output


num_classes = len(dataset.classes)
num_angles = 16
model = DualOutputModel(num_classes, num_angles).to(device)

# Huấn luyện
criterion_class = nn.CrossEntropyLoss()
criterion_angle = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=0.001)

num_epochs = 20
for epoch in range(num_epochs):
    model.train()
    for inputs, labels_class, labels_angle in train_loader:
        inputs, labels_class, labels_angle = inputs.to(device), labels_class.to(device), labels_angle.to(device)
        optimizer.zero_grad()
        class_outputs, angle_outputs = model(inputs)
        loss_class = criterion_class(class_outputs, labels_class)
        loss_angle = criterion_angle(angle_outputs, labels_angle)
        loss = loss_class + loss_angle
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch + 1} completed.")

# Lưu mô hình
torch.save(model.state_dict(), "dual_output_model.pth")
