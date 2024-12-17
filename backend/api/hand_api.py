import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import io
import uvicorn
import os

# Khởi tạo ứng dụng FastAPI
app = FastAPI()

# Thêm middleware để hỗ trợ CORS (cho phép ReactJS truy cập API)
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],  # Cho phép tất cả các domain. Bạn có thể giới hạn ReactJS URL tại đây.
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# Đường dẫn đến mô hình đã lưu
current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, "../model/best_hand_angle_model.pth")

# Cấu hình thiết bị
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Tiền xử lý ảnh (giống với khi huấn luyện)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Định nghĩa mô hình (giống với khi huấn luyện)
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

# Số lớp góc quay
num_angles = 16

# Khởi tạo mô hình và tải trọng số đã huấn luyện
model = AngleOnlyModel(num_angles=num_angles).to(device)
model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
model.eval()  # Chuyển sang chế độ đánh giá

# Hàm dự đoán góc quay từ đầu ra của mô hình
def predict_angle(output, num_angles):
    _, predicted_idx = torch.max(output, 1)
    predicted_angle = predicted_idx.item() * (360 / num_angles)
    return predicted_angle

# Endpoint API
@app.post("/predict/")
async def predict(image: UploadFile = File(...)):
    try:
        # Đọc ảnh từ file upload
        image_data = await image.read()
        pil_image = Image.open(io.BytesIO(image_data)).convert("RGB")

        # Tiền xử lý ảnh
        processed_image = transform(pil_image).unsqueeze(0).to(device)  # Thêm batch dimension

        # Dự đoán
        with torch.no_grad():
            output = model(processed_image)
            predicted_angle = predict_angle(output, num_angles)

        # Trả kết quả
        return {"angle": predicted_angle}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
