import torch
from torchvision import transforms
from torchvision.models import resnet18
import torch.nn as nn
from PIL import Image
import os


# Đường dẫn tới thư mục train
data_dir = "output_dataset/train"

# Hàm tự động lấy class_labels từ thư mục train
def get_class_labels(data_dir):
    class_labels = [name for name in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, name))]
    return sorted(class_labels)

class_labels = get_class_labels(data_dir)
num_classes = len(class_labels)  # Đặt số lớp bằng số lượng nhãn tìm được
print("Danh sách nhãn tự động:", class_labels)

# Cấu hình thiết bị (sử dụng GPU nếu có sẵn)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Định nghĩa lại mô hình ResNet18 và tải trọng số đã huấn luyện
model = resnet18(weights=None)  # Không tải trọng số từ ImageNet
model.fc = nn.Linear(model.fc.in_features, num_classes)  # Đảm bảo số lớp đúng với mô hình đã huấn luyện
model.load_state_dict(torch.load("object_classification_model.pth", map_location=device), strict=False)
model = model.to(device)
model.eval()

# Chuẩn bị phép biến đổi ảnh giống như khi huấn luyện
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Đường dẫn tới ảnh kiểm thử
image_path = "testObject.PNG"  # Thay đường dẫn này bằng ảnh mà bạn muốn kiểm thử

# Tải và biến đổi ảnh
image = Image.open(image_path).convert("RGB")  # Chuyển đổi ảnh sang RGB để đảm bảo có 3 kênh
input_tensor = transform(image).unsqueeze(0).to(device)

# Thực hiện dự đoán
with torch.no_grad():
    outputs = model(input_tensor)
    _, predicted = torch.max(outputs, 1)
    predicted_label = class_labels[predicted.item()]

# Hiển thị kết quả dự đoán
print(f"Loại đối tượng dự đoán: {predicted_label}")
