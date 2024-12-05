import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import numpy as np
import os

# Cấu hình thiết bị (sử dụng GPU nếu có sẵn)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Đường dẫn tới file mô hình đã huấn luyện
model_path = "dual_output_model.pth"

# Đường dẫn tới thư mục train
data_dir = "output_dataset/train"

# Hàm tự động lấy class_labels từ thư mục train
def get_class_labels(data_dir):
    class_labels = [name for name in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, name))]
    return sorted(class_labels)

# Các lớp đối tượng và số góc quay
class_labels = get_class_labels(data_dir) # Cập nhật danh sách nhãn loại đối tượng theo dataset của bạn
num_classes = len(class_labels)
num_angles = 16  # Số góc quay đã sử dụng trong huấn luyện (mỗi góc cách nhau 22.5 độ)

# Định nghĩa lại mô hình DualOutputModel
class DualOutputModel(nn.Module):
    def __init__(self, num_classes, num_angles):
        super(DualOutputModel, self).__init__()
        self.base_model = models.resnet18(weights=None)
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

# Tải mô hình và chuyển sang thiết bị
model = DualOutputModel(num_classes=num_classes, num_angles=num_angles).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# Chuẩn bị phép biến đổi ảnh giống như khi huấn luyện
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Hàm để dự đoán loại đối tượng và góc quay
def predict(image_path):
    # Tải và biến đổi ảnh
    image = Image.open(image_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0).to(device)

    # Dự đoán
    with torch.no_grad():
        class_output, angle_output = model(input_tensor)
        
        # Loại đối tượng
        _, predicted_class = torch.max(class_output, 1)
        predicted_class_label = class_labels[predicted_class.item()]
        
        # Góc quay
        _, predicted_angle = torch.max(angle_output, 1)
        predicted_angle_degree = predicted_angle.item() * 22.5  # Chuyển đổi sang độ

    return predicted_class_label, predicted_angle_degree

# Đường dẫn tới ảnh kiểm thử
image_path = "testDirection.PNG"  # Thay bằng đường dẫn tới ảnh mà bạn muốn kiểm thử

# Thực hiện dự đoán và hiển thị kết quả
predicted_class_label, predicted_angle_degree = predict(image_path)
print(f"Loại đối tượng dự đoán: {predicted_class_label}")
print(f"Góc quay dự đoán: {predicted_angle_degree:.1f} độ")
