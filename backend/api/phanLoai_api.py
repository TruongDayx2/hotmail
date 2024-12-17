from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.responses import FileResponse
import os
import shutil
import logging

# Khởi tạo ứng dụng FastAPI
app = FastAPI()

# Cấu hình logging
logging.basicConfig(level=logging.DEBUG)

# Định nghĩa mô hình dữ liệu cho từng API
class ListImagesRequest(BaseModel):
    folder_path: str

class FolderRequest(BaseModel):
    folder_path: str  # Đường dẫn thư mục chứa ảnh
    image_name: str   # Tên file ảnh (chỉ dùng trong API di chuyển ảnh)
    object_name: str  # Tên object (chỉ dùng trong API di chuyển ảnh)
    angle_degree: str # Góc a_degree (chỉ dùng trong API di chuyển ảnh)
    folder_add: str # Thư mục đích

class FolderImg(BaseModel):
    folder_path: str  # Đường dẫn thư mục chứa ảnh

# API: Trả danh sách hình ảnh từ thư mục động
@app.post("/list-images/")
async def list_images(request: ListImagesRequest):
    logging.debug(f"Received request: {request}")
    folder_path = request.folder_path

    # Kiểm tra thư mục có tồn tại không
    if not os.path.exists(folder_path):
        raise HTTPException(status_code=404, detail=f"Folder not found: {folder_path}")

    # Lấy danh sách các file ảnh (chỉ các file hợp lệ)
    images = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
    return {"images": images}

@app.post("/list-objects")
async def list_object(request: ListImagesRequest):
    folder_path = request.folder_path
    # Kiểm tra thư mục có tồn tại không
    if not os.path.exists(folder_path):
        raise HTTPException(status_code=404, detail=f"Folder not found: {folder_path}")
    # Lấy danh sách thư mục con trong folder_path
    subdirectories = [
        d for d in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, d))
    ]

    # Trả về danh sách thư mục
    return {"folders": subdirectories}

# API: Di chuyển hình ảnh đến thư mục đích
@app.post("/move-image/")
async def move_image(request: FolderRequest):
    logging.debug(f"Received request: {request}")
    source_folder = request.folder_path
    image_name = request.image_name
    object_name = request.object_name
    angle_degree = request.angle_degree
    folder_add = request.folder_add

    # Kiểm tra thư mục nguồn
    if not os.path.exists(source_folder):
        raise HTTPException(status_code=404, detail=f"Source folder not found: {source_folder}")

    # Kiểm tra thư mục đích
    if not os.path.exists(folder_add):
        raise HTTPException(status_code=404, detail=f"Source folder add not found: {folder_add}")


    # Đường dẫn file nguồn
    source_path = os.path.join(source_folder, image_name)
    print("source_path",source_path)
    if not os.path.exists(source_path):
        raise HTTPException(status_code=404, detail=f"Image not found: {source_path}")

    # Đường dẫn đích
    target_folder = os.path.join(folder_add, object_name, angle_degree + "_degree")
    target_path = os.path.join(target_folder, image_name)
    print("target_folder",target_folder)
    print("target_path",target_path)

    # Tạo thư mục đích nếu chưa tồn tại
    os.makedirs(target_folder, exist_ok=True)

    # Di chuyển file
    shutil.move(source_path, target_path)

    # Trả kết quả
    return {"message": f"Image '{image_name}' moved to '{target_folder}' successfully."}
    

# API: Trả về hình ảnh theo tên
@app.get("/images/{image_name}")
async def get_image(image_name: str,folder_path: str):
    # Đường dẫn file nguồn
    if not os.path.exists(folder_path):
        raise HTTPException(status_code=404, detail=f"Image not found: {folder_path}")

    # Đường dẫn đầy đủ đến file
    image_path = os.path.join(folder_path, image_name)

    # Kiểm tra file tồn tại
    if not os.path.exists(image_path):
        raise HTTPException(status_code=404, detail=f"Image '{image_name}' not found.")

    # Trả về file ảnh
    return FileResponse(image_path)