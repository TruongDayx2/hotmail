import os
import shutil
from sklearn.model_selection import train_test_split

# Đường dẫn gốc tới dataset của bạn
original_data_dir = "dataset0"
output_data_dir = "output_dataset"  # Đường dẫn sẽ lưu dataset đã chia thành train/test

# Tỷ lệ dữ liệu train/test
train_ratio = 0.8

# Tạo thư mục train và test trong output_data_dir
for phase in ["train", "test"]:
    for obj in os.listdir(original_data_dir):
        obj_dir = os.path.join(original_data_dir, obj)
        if os.path.isdir(obj_dir):
            for angle in os.listdir(obj_dir):
                angle_dir = os.path.join(obj_dir, angle)
                if os.path.isdir(angle_dir):
                    os.makedirs(os.path.join(output_data_dir, phase, obj, angle), exist_ok=True)

# Phân chia dữ liệu vào train/test
for obj in os.listdir(original_data_dir):
    obj_dir = os.path.join(original_data_dir, obj)
    if os.path.isdir(obj_dir):
        for angle in os.listdir(obj_dir):
            angle_dir = os.path.join(obj_dir, angle)
            if os.path.isdir(angle_dir):
                # Lấy tất cả ảnh trong thư mục góc độ hiện tại
                images = [img for img in os.listdir(angle_dir) if img.endswith((".PNG", ".jpg", ".jpeg"))]

                # Kiểm tra nếu thư mục không có ảnh
                if len(images) == 0:
                    print(f"Thư mục {angle_dir} không có ảnh nào, bỏ qua.")
                    continue

                # Phân chia train/test
                train_images, test_images = train_test_split(images, train_size=train_ratio, random_state=42)
                
                # Copy ảnh vào thư mục train
                for img in train_images:
                    src_path = os.path.join(angle_dir, img)
                    dest_path = os.path.join(output_data_dir, "train", obj, angle, img)
                    shutil.copy2(src_path, dest_path)
                
                # Copy ảnh vào thư mục test
                for img in test_images:
                    src_path = os.path.join(angle_dir, img)
                    dest_path = os.path.join(output_data_dir, "test", obj, angle, img)
                    shutil.copy2(src_path, dest_path)

print("Dataset đã được chia thành công!")
