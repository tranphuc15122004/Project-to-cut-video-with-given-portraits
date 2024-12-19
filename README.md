# Chương trình cắt video theo người với ảnh chân dung cho trước

## 📖 Mô tả
Với một video đầu vào và ảnh chân dụng được cung cấp, chương trình thực hiện cắt nhưng đoạn có chứa mục tiêu. Đầu ra của chương trình là một video tổng hợp tất cả những đoạn có chứa đối tượng yêu cầu.

Chương trình được xây dựng gồm 2 giai đoạn chính: <br>

- **Giai đoạn 1**: duyệt qua video lần thứ nhất để thu thập các thông tin về hình dạng của đối tượng dựa trên khuôn mặt (Chương trình sẽ sử dụng các kỹ thuật nhận diện khuôn mặt để xác định vị trí và hình dạng của đối tượng trong từng khung hình).

- **Giai đoạn 2**: duyệt qua video lần thứ hai để cắt và ghép các đoạn video có chứa đối tượng yêu cầu dựa trên thông tin thu thập được từ giai đoạn 1. Kết quả là một video tổng hợp các đoạn có chứa đối tượng mục tiêu.

## 🛠️ Công nghệ và Thư viện Sử dụng

### 📦 **Xử lý Deep Learning**

- **PyTorch**: Framework học sâu mạnh mẽ, cung cấp cơ sở để xây dựng và triển khai các mô hình học máy. Trong chương trình này, PyTorch hỗ trợ cả xử lý khuôn mặt và nhận dạng người.

- **TorchReID**: Một công cụ tối ưu cho bài toán nhận diện lại (Re-Identification), sử dụng để trích xuất vector đặc trưng cho nhận diện người. (Trong chương trình sử dụng pretrained model **osnet_x1_0**)

- **FaceNet PyTorch (MTCNN & InceptionResnetV1)**:

  - **MTCNN**: Công cụ phát hiện khuôn mặt đa nhiệm nhanh chóng và chính xác.
  - **InceptionResnetV1**: Mạng nơ-ron tiên tiến được huấn luyện trước trên tập dữ liệu VGGFace2, chuyên dùng để tạo embedding cho khuôn mặt.


### 📷 **Phát hiện và xử lý hình ảnh**

- **YOLOv8n (You Only Look Once, phiên bản nhẹ)**: Phiên bản nhỏ gọn của mô hình YOLO, được tối ưu hóa để phát hiện đối tượng nhanh chóng và hiệu quả.

- **OpenCV**: Thư viện phổ biến cho xử lý hình ảnh và video. Trong chương trình này, OpenCV được sử dụng để đọc, hiển thị và xử lý dữ liệu hình ảnh.

- **Pillow (PIL)**: Được sử dụng để xử lý các định dạng ảnh trước khi đưa vào các mô hình học sâu.


### 🧠 **Tính toán và So sánh**

- **NumPy**: Thư viện tính toán số học nhanh và hiệu quả, hỗ trợ xử lý mảng dữ liệu lớn trong chương trình.

- **SciPy (Spatial Distance)**: Module `cosine` được sử dụng để tính khoảng cách cosine giữa các embedding, giúp so sánh độ tương đồng của khuôn mặt và người.

- **Time**: Thư viện Python tích hợp, được dùng để đo thời gian thực thi trong các giai đoạn của chương trình.


### ⚙️ **Cấu trúc Chương trình**
1. **Nhận diện Khuôn mặt**:
   - Phát hiện khuôn mặt trong hình ảnh bằng **MTCNN**.
   - Trích xuất vector đặc trưng khuôn mặt bằng **InceptionResnetV1**.
   - So sánh độ tương đồng giữa các khuôn mặt bằng khoảng cách cosine.

2. **Nhận diện Người**:
   - Phát hiện người trong khung hình với **YOLOv8**.
   - Trích xuất đặc trưng người bằng **TorchReID**.
   - So sánh các đặc trưng để phân biệt hoặc nhận diện lại người.

### ⚡ **Hiệu suất và Tăng tốc**
- **CUDA/GPU**: Tăng tốc quá trình xử lý thông qua GPU, giảm thời gian thực thi cho các mô hình học sâu như TorchReID, MTCNN, và YOLO. 

## 🧪 Kiểm thử

- Bạn có thể kiểm thử chương trình qua [liên kết này](https://husteduvn-my.sharepoint.com/:f:/g/personal/phuc_td224891_sis_hust_edu_vn/EoOOFX1PLLxItFRFVfAUo8MBAuNlmYEe48g48c_oSArGrw?e=IcqCRC).
