import cv2
import torch
from PIL import Image
import numpy as np
from facenet_pytorch import MTCNN, InceptionResnetV1
from torchvision import transforms
from scipy.spatial.distance import cosine
import time
from typing import Tuple

# Initialize model vggface2
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
face_extraction_model = InceptionResnetV1(pretrained= 'vggface2' ,device=device).eval()
face_detection_model = MTCNN(keep_all= True , device= device)

# Detect face in an image
def face_detecting(img , rate , face_detection_model) -> list:
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    boxes, probabilities = face_detection_model.detect(img_rgb)
    results = []
    back_up = []
    img_area = img.shape[0] * img.shape[1]  # Diện tích ảnh

    if boxes is None:
        print("Không thể xác định được khuôn mặt trong ảnh.")
    else:
        for box, prob in zip(boxes, probabilities):
            if prob > 0.7:
                x1, y1, x2, y2 = map(int, box)
                # Tính diện tích bounding box
                face_area = (x2 - x1) * (y2 - y1)
                face_area_ratio = face_area / img_area
                face_crop = img[y1:y2, x1:x2]    
                back_up.append(face_crop)
                # Chỉ giữ các khuôn mặt có tỷ lệ < 0.2
                if face_area_ratio < rate:
                    face_crop = img[y1:y2, x1:x2]
                    results.append(face_crop)
        print(f"Đã phát hiện {len(results)} khuôn mặt có tỷ lệ nhỏ hơn 20% cơ thể.")

    return results or back_up

def target_face_detect(img , target_embed , face_detection_model) -> bool:
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    boxes, probabilities = face_detection_model.detect(img_rgb)
    check = False
    if boxes is None or len(boxes) ==0 :
        print("Không thể xác định được khuôn mặt trong ảnh.")
        return check
    else:
        try:
            faces = face_detecting(img , face_detection_model)
        except: return False
        if faces is not None:
            for face in faces:
                try:
                    face_embed = face_embedding(face , face_extraction_model)
                except:
                    return False
                if face_embed is None:
                    continue
                face_embed = face_embed[0]
                try:
                    check = compare_faces(face_embed , target_embed , 0.5)
                except: return False
                if check: 
                    print("Đã phát hiện đối tượng trong ảnh.")
                    return check
                
    """ else:
        for box, prob in zip(boxes, probabilities):
            if prob >= 0.5:
                x1, y1, x2, y2 = map(int, box)
                crop = img[y1:y2 , x1:x2]
                crop_embed = face_embedding(crop)
                if crop_embed is None:
                    continue
                if compare_faces(crop_embed , target_embed , 0.4):
                    print("Đã phát hiện đối tượng trong ảnh.")
                    return True """
    print('Không có đối tượng ở trong ảnh')
    return check

def face_detecting_coordinates(img ,face_detection_model) -> list:
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    boxes, probabilities = face_detection_model.detect(img_rgb)
    
    # Kiểm tra nếu không phát hiện được khuôn mặt nào
    if boxes is None or len(boxes) == 0:
        return []
    face_coordinates = []
    
    # Lấy tất cả các bounding boxes và độ tin cậy của chúng
    for box, prob in zip(boxes, probabilities):
        x1, y1, x2, y2 = map(int, box)
        face_coordinates.append((x1, y1, x2, y2, prob))  # Thêm độ tin cậy vào tuple

    # Sắp xếp các bounding box theo độ tin cậy giảm dần
    face_coordinates.sort(key=lambda x: x[4], reverse=True)    
    return face_coordinates

def face_of_person(person_img , target_embed):
    img_rgb = cv2.cvtColor(person_img, cv2.COLOR_BGR2RGB)
    boxes, probabilities = face_detection_model.detect(img_rgb)
    
    if boxes is None or len(boxes) ==0 :
        print("Không thể xác định được khuôn mặt trong ảnh.")
        return False
    else:
        for box, prob in zip(boxes, probabilities):
            if prob >= 0.7:
                x1, y1, x2, y2 = map(int, box)
                crop = person_img[y1:y2 , x1:x2]
                crop_embed = face_embedding(crop)
                if crop_embed is None:
                    continue
                if compare_faces(crop_embed , target_embed , 0.5):
                    print("Đã phát hiện đối tượng trong ảnh.")
                    return True
    print('Không có đối tượng ở trong ảnh')
    return False

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
def face_embedding(img , face_extraction_model) -> 'np.array':
    processed =  preprocess_image(img)
    if processed is None:
        return None
    with torch.no_grad():
        embedding = face_extraction_model(processed)
        embedding = embedding.cpu().numpy()
    return embedding


def compare_faces(embedding1, embedding2, threshold=0.6):
    embedding1 = embedding1.reshape(512)
    embedding2 = embedding2.reshape(512)
    score = cosine(embedding1, embedding2)
    if score <= threshold:
        print('=> face is a Match (%.3f <= %.3f)' % (score, threshold))
        return True
    else:
        print('=> face is NOT a Match (%.3f > %.3f)' % (score, threshold))
        return False

def test1():
    target_img = cv2.imread(r'assets/input/messi4.jpg')
    target_face = face_detecting(target_img)[0]
    target_embed = face_embedding(target_face)
    
    img = cv2.imread(r'assets/input/messi5.jpg')
    print (face_detecting_coordinates(img , target_embed))

if __name__ == "__main__":
    test1()