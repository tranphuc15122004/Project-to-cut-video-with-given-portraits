import torchreid 
import torch
from torchvision import transforms 
from scipy.spatial.distance import cosine
from PIL import Image
from ultralytics import YOLO
import cv2
import numpy as np

device= 'cuda' if torch.cuda.is_available() else 'cpu'
extraction_model = torchreid.utils.FeatureExtractor(
    model_name= 'osnet_x1_0',
    device= device
)
detection_model = YOLO('yolov8n.pt')
detection_model.to(device)

def person_detecting(image) -> list['np.array']:
    person_crop = []
    result= detection_model(image , conf = 0.5, device = device)
    for res in result:
        for box in res.boxes:
            classID = int(box.cls[0])
            if classID == 0:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                tmp = image [y1:y2 , x1:x2]
                person_crop.append(tmp)
    return person_crop


# Trả về Tensor với kích thước đầu vào chuẩn để chuẩn bị cho quá trình trích xuất đặc trưng
def preprocess(person_image) -> torch.Tensor:
    # Kiểm tra tính hợp lệ của đầu vào
    if person_image is None:
        print("Không thể đọc được ảnh Người đầu vào. Đầu vào là None.")
        return None
    if not isinstance(person_image, np.ndarray):
        print("Đầu vào của hàm Phân tích Người không phải là mảng NumPy.")
        return None
    if person_image.size == 0:
        print("Ảnh đầu vào của hàm Phân tích Người không chứa dữ liệu.")
        return None

    try:
        # Chuyển đổi ảnh sang RGB
        img = Image.fromarray(cv2.cvtColor(person_image, cv2.COLOR_BGR2RGB))
    except cv2.error as e:
        print(f"Lỗi trong quá trình chuyển đổi màu: {e}")
        return None

    # Thiết lập các phép biến đổi ảnh
    transform = transforms.Compose([
        transforms.Resize((256, 128)),  # Resize ảnh về kích thước 256x128
        transforms.ToTensor(),         # Chuyển ảnh sang Tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Chuẩn hóa giá trị
    ])

    try:
        # Biến đổi ảnh
        res = transform(img).unsqueeze(0).to(device)  # Thêm batch dimension và đưa lên thiết bị
    except Exception as e:
        print(f"Lỗi trong quá trình biến đổi ảnh: {e}")
        return None

    return res

# Trả về embedding matrix của đối tượng dưới dạng numpy array dạng (1,512)
def person_embedding(person_image) -> np.array:
    processed = preprocess(person_image)
    with torch.no_grad():
            embedding = extraction_model(processed)
            embedding = embedding.cpu().numpy()
    return embedding

def compare_people(embedding1, embedding2, threshold=0.6):
    embedding1 = embedding1[0]
    embedding2 = embedding2[0]
    score = cosine(embedding1, embedding2)
    if score <= threshold:
        print('=> These picture belong to the same person (%.3f <= %.3f)' % (score, threshold))
        return True
    else:
        print('=> These picture do NOT belong to the same person (%.3f > %.3f)' % (score, threshold))
        return False

def test():
    img = cv2.imread(r'assets\\input\\messi1.jpg')
    person_img = person_detecting(img)
    embed = person_embedding(person_img)
    embed = embed[0]
    print(embed)
    
    img1 = cv2.imread(r'assets\\input\\messi3.jpg')
    person_img1 = person_detecting(img1)
    embed1 = person_embedding(person_img1)
    embed1 = embed1[0]
    compare_people(embed , embed1 , 0.5) 
    
if __name__ == "__main__":
    test()