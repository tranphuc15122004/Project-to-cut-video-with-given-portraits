import cv2
import torch
from PIL import Image
import numpy as np
from facenet_pytorch import MTCNN, InceptionResnetV1
from torchvision import transforms
from scipy.spatial.distance import cosine
import time

# Initialize model vggface2
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
face_extraction_model = InceptionResnetV1(pretrained= 'vggface2' ,device=device).eval()
face_detection_model = MTCNN(keep_all= True , device= device)

# Detect face in an image
def face_detecting(img) -> list:
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    boxes, probabilities = face_detection_model.detect(img_rgb)
    face_crops = []

    if boxes is None:
        print("Không thể xác định được khuôn mặt trong ảnh.")
    else:
        for box, prob in zip(boxes, probabilities):
            x1, y1, x2, y2 = map(int, box)
            # Cắt khuôn mặt ra từ ảnh
            face_crop = img[y1:y2, x1:x2]
            face_crops.append(face_crop)
        print(f"Đã phát hiện {len(boxes)} khuôn mặt.")
    return face_crops


# Preprocess image for face feature extraction
# Đầu vào là một ảnh khuôn mặt đọc bằng cv2 (numpy array)
# Đầu ra là một tensor 
def preprocess_image(face_image) -> torch.Tensor:
    # Kiểm tra xem đầu vào có hợp lệ không
    if face_image is None:
        print("Không thể đọc được ảnh khuôn mặt. Đầu vào là None.")
        return None
    if not isinstance(face_image, np.ndarray):
        print("Đầu vào của hàm Phân tích khuôn mặt không phải là mảng NumPy.")
        return None
    if face_image.size == 0:
        print("Ảnh đầu vào hàm Phân tích khuôn mặt không chứa dữ liệu.")
        return None

    try:
        # Chuyển đổi ảnh sang RGB
        img = Image.fromarray(cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB))
    except cv2.error as e:
        print(f"Lỗi trong quá trình chuyển đổi màu: {e}")
        return None

    # Thiết lập các phép biến đổi ảnh
    transform = transforms.Compose([
        transforms.Resize((160, 160)),  # Resize ảnh
        transforms.ToTensor(),         # Chuyển ảnh sang Tensor
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Chuẩn hóa giá trị
    ])

    try:
        # Biến đổi ảnh
        img_tensor = transform(img).unsqueeze(0).to(device)  # Thêm batch dimension và đưa lên thiết bị
    except Exception as e:
        print(f"Lỗi trong quá trình biến đổi ảnh: {e}")
        return None

    return img_tensor

# feature extraction
# Đầu vào là một mảng các tensor
# Đầu ra: 1 mảng các numpy array 
def face_embedding(img) -> 'np.array':
    processed =  preprocess_image(img)
    if processed is None:
        return None
    with torch.no_grad():
        embedding = face_extraction_model(processed)
        embedding = embedding.cpu().numpy()
    return embedding


def compare_faces(embedding1, embedding2, threshold=0.6):
    embedding1 = embedding1[0]
    embedding2 = embedding2[0]
    score = cosine(embedding1, embedding2)
    if score <= threshold:
        print('=> face is a Match (%.3f <= %.3f)' % (score, threshold))
        return True
    else:
        print('=> face is NOT a Match (%.3f > %.3f)' % (score, threshold))
        return False

def test1():
    begin_time = time.time()
    img_paths = [r'assets\\input\\messi1.jpg',
                r'assets\\input\\messi2.jpg',
                r'assets\\input\\messi3.jpg',
                r'assets\\input\\messi4.jpg',
                r'assets\\input\\messi5.jpg',
                r'assets\\input\\messi6.jpg',
                r'assets\\input\\messi7.jpg',]
    face_tensors = []
    for path in img_paths:
        _, faces = face_detecting(path)
        if faces:
            face_tensors.append(preprocess_image(faces[0])) 

    if face_tensors:
        batched_tensors = torch.cat(face_tensors, dim=0)  
        embeddings = face_embedding(batched_tensors)  
        mean_feature = np.mean(embeddings, axis=0)
    else:
        print("Không có khuôn mặt nào được phát hiện.")
    
    img , face = face_detecting(r'assets\\input\\messi_test.jpg')
    processed_img =  preprocess_image(face[0])
    embedded = face_embedding(processed_img)
    embedded = embedded.squeeze()
    compare_faces(embedded , mean_feature)
    print(time.time() - begin_time)
    
def test2():
    begin_time = time.time()
    img1 , face1 = face_detecting(r'assets\\input\\messi_test.jpg')
    processed_img1 =  preprocess_image(face1[0])
    embedded1 = face_embedding(processed_img1)
    embedded1 =    embedded1.squeeze()
    
    img2 , face2 = face_detecting(r'assets\\input\\messi6.jpg')
    processed_img2 =  preprocess_image(face2[0])
    embedded2 = face_embedding(processed_img2)
    embedded2 = embedded2.squeeze()
    compare_faces (embedded1 , embedded2)
    print(time.time() - begin_time)
    
def test3():
    begin = time.time()
    img = cv2.imread(r'assets\\input\\messi_test.jpg')
    face1 = face_detecting(img)
    embedded1 = face_embedding(face1)
    print(embedded1)
    print(embedded1[0].shape)    
    print(time.time() - begin)

if __name__ == "__main__":
    test3()