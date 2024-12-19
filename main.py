import cv2
from ultralytics import YOLO
from person_module.person_process import  person_embedding, compare_people
import torch
from face_module.face_process import face_detecting , face_embedding , compare_faces
import numpy as np
import torchreid 
from facenet_pytorch import MTCNN, InceptionResnetV1
import os
from phase1_test import phase1
from phase2_test import phase2
from scipy.spatial.distance import cosine
import time
device = 'gpu' if torch.cuda.is_available() else 'cpu'
face_extraction_model = InceptionResnetV1(pretrained= 'vggface2' ,device=device).eval()
face_detection_model = MTCNN(keep_all= True , device= device)
person_extraction_model = torchreid.utils.FeatureExtractor(
    model_name= 'osnet_x1_0',
    device= device
)
person_detection_model = YOLO('yolov8n.pt')
person_detection_model.to(device)

# PATH
input_path = r'assets\\input\\neymar_testcase.mp4'
output_folder = r'assets\\output' ; output_filename = r'result.mp4'
output_path = os.path.join(output_folder , output_filename)


# Các khai báo của đối tượng
target_img = cv2.imread(r'assets/input/messi4.jpg')
target_face = face_detecting(target_img , 1  ,face_detection_model)
if len(target_face) == 1:
    target_face = target_face[0]
target_face_embed = face_embedding(target_face , face_extraction_model)[0]
target_gesture_embed_log = []
average_target_embed = None


if __name__ == "__main__":
    begin_time = time.time()
    phase1(input_path  , target_face_embed , target_gesture_embed_log , face_extraction_model , face_detection_model , person_detection_model, person_extraction_model)
    print ('############### KẾT THÚC PHASE I #################')
    print('Thời gian thực hiện chương trình: ' , time.time() - begin_time) 
    average_target_embed = np.mean(target_gesture_embed_log , axis= 0)
    
    max_bound = 0.0
    for gesture_embed in target_gesture_embed_log:
        temp = cosine(average_target_embed , gesture_embed)
        max_bound = max(max_bound , temp)
    max_bound = round(max_bound +0.05 , 1)
    max_bound = max(max_bound , 0.3)
    
    phase2_begin_time = time.time()
    phase2(input_path, output_path  ,max_bound , average_target_embed , target_face_embed , face_extraction_model , face_detection_model , person_detection_model ,person_extraction_model)
    print ('############### KẾT THÚC PHASE II #################')
    print('Thời gian thực hiện chương trình: ' , time.time() - begin_time)
    print('Thời gian thực hiện phase 2: ' , time.time() -phase2_begin_time)