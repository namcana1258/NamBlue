import cv2 as cv
import easyocr

# 1. KHỞI TẠO MODEL EASYOCR MỘT LẦN DUY NHẤT Ở ĐÂY
print("Đang tải model EasyOCR, vui lòng đợi...")
reader = easyocr.Reader(['en'], gpu=True)

def crop_plate(img, x, y, w, h, pad=10):
    x1 = max(x - pad, 0)
    y1 = max(y - pad, 0)
    x2 = min(x + w + pad, img.shape[1])
    y2 = min(y + h + pad, img.shape[0])
    return img[y1:y2, x1:x2]

def preprocess_plate(plate_img):
    if plate_img.size == 0:
        return plate_img
    h, w = plate_img.shape[:2]
    scale = 60.0 / h if h < 60 else 1.0
    resize = cv.resize(plate_img, None, fx=scale, fy=scale, interpolation=cv.INTER_CUBIC)
    gray = cv.cvtColor(resize, cv.COLOR_BGR2GRAY)
    _, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    denoised = cv.fastNlMeansDenoising(binary, h=10)
    return denoised

# 2. ĐỌC LUỒNG VIDEO (Thay tên file mp4 tương ứng)
video_path = 'baigiuxe.mp4'  # Thay đổi thành 'plate2.mp4' nếu muốn
cap = cv.VideoCapture(video_path)

if not cap.isOpened():
    print("Không thể mở được video!")
    exit()

# 3. VÒNG LẶP XỬ LÝ TỪNG KHUNG HÌNH
while True:
    ret, frame = cap.read()
    
    # Nếu hết video thì thoát vòng lặp
    if not ret:
        print("Đã phát hết video.")
        break

    # Thu nhỏ frame lại một chút để máy xử lý nhanh hơn (tùy chọn)
    # frame = cv.resize(frame, (1024, 768))

    # --- Các bước tiền xử lý ảnh ---
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    _, binary = cv.threshold(gray, 90, 255, cv.THRESH_BINARY)
    denoise = cv.GaussianBlur(binary, (5, 5), 0)
    edges = cv.Canny(denoise, 20, 200)

    kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
    closed = cv.morphologyEx(edges, cv.MORPH_CLOSE, kernel)

    contours, _ = cv.findContours(closed.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    img_size = frame.shape[0] * frame.shape[1] 

    # --- Lọc các khung hình khả nghi ---
    for cnt in contours:
        x, y, w, h = cv.boundingRect(cnt)
        aspect_ratio = w / h
        area_ratio = (w * h) / img_size
        
        # Nới rộng aspect_ratio xuống 1.2 để nhận diện cả biển số xe máy vuông (trong baigiuxe.mp4)
        if (1.2 < aspect_ratio < 6.0) and (0.005 < area_ratio < 0.15):
            
            # Cắt và tối ưu ảnh biển số
            plate_img = crop_plate(frame, x, y, w, h)
            plate_ready = preprocess_plate(plate_img)
            
            if plate_ready.size == 0:
                continue
                
            # Đưa qua AI đọc chữ
            result = reader.readtext(plate_ready, allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789', detail=1)
            
            for (bbox, text, conf) in result:
                # Lọc bỏ nhiễu: Chỉ lấy kết quả AI tự tin > 40%
                if conf > 0.4:
                    # Khoanh vùng biển số bằng hình chữ nhật xanh lá
                    cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    
                    # In kết quả Text lên video bằng màu đỏ
                    cv.putText(frame, f"{text} ({conf:.2f})", (x, y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    
                    print(f"Phát hiện biển số: {text} - Tỉ lệ chính xác: {conf:.2f}")

    # 4. HIỂN THỊ VIDEO
    cv.imshow('Nhan dien bien so xe', frame)

    # Nhấn phím 'q' trên bàn phím để tắt video giữa chừng
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

# Dọn dẹp bộ nhớ RAM và GPU
cap.release()
cv.destroyAllWindows()