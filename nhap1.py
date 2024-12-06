import cv2
from ultralytics  import YOLO
import torch
import numpy as np
import os
from person_module.person_process import person_detecting, person_embedding, compare_people
from face_module.face_process import face_embedding , face_detecting ,compare_faces

target_img = cv2.imread(r'assets/input/messi11.jpg')
target_face = person_detecting(target_img)
if len(target_face) > 0:
    target_face = target_face[0]
    
target_embed = person_embedding(target_face)

target_img1 = cv2.imread(r'assets/input/messi9.jpg')
target_face1 = person_detecting(target_img1)
if len(target_face1) > 0:
    target_face1 = target_face1[0]
    
target_embed1 = person_embedding(target_face1)
target_embed = target_embed.flatten()
print(target_embed)
print(target_embed.shape)

print(type(target_embed))
#compare_people(target_embed , target_embed1 , 0.5)
