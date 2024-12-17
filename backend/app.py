from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.hand_api import app as hand_api
# from api.test_direction import app as test_direction_api
from api.phanLoai_api import app as phanLoai_api
# from api.test_hand import app as test_hand_api

# Khởi tạo ứng dụng FastAPI chính
app = FastAPI()

# Thêm middleware CORS (cho phép ReactJS truy cập)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Chỉ định URL của ứng dụng ReactJS (nếu cần)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Tích hợp các API con
app.mount("/hand", hand_api)  # Mount API tại đường dẫn /hand
# app.mount("/direction", test_direction_api)  # Mount API tại đường dẫn /direction
app.mount("/phanLoai", phanLoai_api)  # Mount API tại đường dẫn /tool


# Chạy ứng dụng
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
    # pip install -r requirements.txt
    # uvicorn app:app --reload --host 0.0.0.0 --port 8000

