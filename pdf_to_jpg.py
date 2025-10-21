from pdf2image import convert_from_path
import os
import time

poppler_path = r"C:\Users\ADMIN\Downloads\vintern_api\poppler-25.07.0\poppler-25.07.0\Library\bin"
pdf_path = "C:/Users/khait/OneDrive/Desktop/AI_KIOSK/ocr_form/pdf/Bieu_mau.pdf"

# Thiết lập thư mục xuất (ngay cạnh file script)
output_folder = os.path.dirname(os.path.abspath(__file__))

# Tăng tốc: dùng đa luồng, xuất trực tiếp bằng Poppler (không load vào PIL), dùng pdftocairo
start_time = time.time()
saved_paths = convert_from_path(
    pdf_path,
    dpi=300,  # giảm xuống 200 nếu cần tốc độ hơn
    poppler_path=poppler_path,
    output_folder=output_folder,
    output_file="page",  # sẽ tạo file dạng page-1.png, page-2.png, ...
    fmt="png",
    thread_count=os.cpu_count() or 1,
    paths_only=True,
    use_pdftocairo=True,
)

for p in saved_paths:
    print(f"Đã lưu: {os.path.basename(p)}")

end_time = time.time()
print(f"Time taken: {end_time - start_time} seconds")