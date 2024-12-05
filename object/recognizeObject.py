import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torchvision.models import resnet18, ResNet18_Weights
from torch.utils.data import DataLoader
import os

import ssl
ssl._create_default_https_context = ssl._create_unverified_context


# Thiết lập thiết bị (GPU nếu có sẵn)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Đường dẫn đến dataset đã chia
data_dir = "output_dataset"  # Cập nhật đường dẫn đến output dataset của bạn

# Chuẩn bị tăng cường dữ liệu (data augmentation)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Load dữ liệu từ các thư mục train và test
train_data = datasets.ImageFolder(root=os.path.join(data_dir, "train"), transform=transform)
test_data = datasets.ImageFolder(root=os.path.join(data_dir, "test"), transform=transform)

train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

# Khởi tạo mô hình ResNet18 với trọng số ImageNet
model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
num_classes = len(train_data.classes)  # Số lượng lớp đối tượng
model.fc = nn.Linear(model.fc.in_features, num_classes)
model = model.to(device)

# Định nghĩa hàm mất mát và tối ưu hóa
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Huấn luyện mô hình
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}")

# Lưu mô hình đã huấn luyện
torch.save(model.state_dict(), "object_classification_model.pth")
print("Đã lưu mô hình nhận diện loại đối tượng dưới dạng 'object_classification_model.pth'")
