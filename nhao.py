import numpy as np
import cv2
from face_module.face_process import face_detecting , face_embedding ,compare_faces
from person_module.person_process import  person_embedding, compare_people , person_detecting
from scipy.spatial.distance import cosine
from facenet_pytorch import MTCNN, InceptionResnetV1
import torchreid
import torch


device = 'gpu' if torch.cuda.is_available() else 'cpu'
face_extraction_model = InceptionResnetV1(pretrained= 'vggface2' ,device=device).eval()
face_detection_model = MTCNN(keep_all= True , device= device)



target_img1 = cv2.imread(r'assets/input/messi4.jpg')
target_face1 = face_detecting(target_img1  ,face_detection_model)
if len(target_face1) == 1:
    target_face = target_face1[0]
target_face_embed1 = face_embedding(target_face , face_extraction_model)[0]


target_img = cv2.imread(r'assets\\input\\Screenshot 2024-12-19 170731.png')
target_face = face_detecting(target_img  ,face_detection_model)
for face in target_face:
    emb = face_embedding(face , face_extraction_model)
    compare_faces(target_face_embed1 ,emb)
