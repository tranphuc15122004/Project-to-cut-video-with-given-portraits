from face_module.face_process import face_detecting , face_embedding
from person_module.person_process import person_detecting , person_embedding
import numpy as np
import torch
import cv2

class Person:
    per_img = None , 
    id  = None
    def __init__(self , id , per_img):
        self.id = id
        self.person_img = per_img
        self.face_img = face_detecting(per_img)[0]
        self.person_embed = person_embedding(per_img)[0]
        self.face_embed = face_embedding(self.face_img)[0]
    
    
img = cv2.imread(r'assets\\input\\messi1.jpg')
person = person_detecting(img)[0]
Nguoi = Person(1 , person)

cv2.imshow('dd' ,Nguoi.face_img)
cv2.waitKey(0)
cv2.destroyAllWindows()