import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import numpy as np

# Đường dẫn đến mô hình đã lưu và ảnh đầu vào
model_path = "best_hand_angle_model.pth"  # Tệp mô hình đã lưu
image_path = "292_Test.jpg"     # Đường dẫn đến ảnh cần kiểm tra

# Cấu hình thiết bị (GPU nếu có sẵn)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Tiền xử lý ảnh (giống với khi huấn luyện)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Định nghĩa lại mô hình (giống với phần huấn luyện)
class AngleOnlyModel(nn.Module):
    def __init__(self, num_angles):
        super(AngleOnlyModel, self).__init__()
        self.base_model = models.resnet50(weights=None)
        in_features = self.base_model.fc.in_features
        self.base_model.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_features, num_angles)
        )

    def forward(self, x):
        return self.base_model(x)

# Khởi tạo mô hình và tải trọng số đã huấn luyện
num_angles = 16
model = AngleOnlyModel(num_angles=num_angles).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()  # Chuyển sang chế độ đánh giá

# Hàm dự đoán góc quay từ đầu ra của mô hình
def predict_angle(output, num_angles):
    _, predicted_idx = torch.max(output, 1)
    predicted_angle = predicted_idx.item() * (360 / num_angles)
    return predicted_angle

# Hàm xử lý và dự đoán cho một ảnh
def process_and_predict(image_path):
    try:
        # Mở và tiền xử lý ảnh
        image = Image.open(image_path).convert("RGB")
        processed_image = transform(image).unsqueeze(0).to(device)  # Thêm batch dimension

        # Dự đoán
        with torch.no_grad():
            output = model(processed_image)
            predicted_angle = predict_angle(output, num_angles)

        print(f"Predicted Angle for the image '{image_path}': {predicted_angle:.2f} degrees")
        return predicted_angle
    except Exception as e:
        print(f"Error processing the image '{image_path}': {e}")

# Kiểm tra với một ảnh
process_and_predict(image_path)
