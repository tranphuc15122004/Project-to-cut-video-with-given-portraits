import numpy as np
import cv2
from face_module.face_process import face_detecting , face_embedding ,compare_faces
from person_module.person_process import  person_embedding, compare_people , person_detecting
from scipy.spatial.distance import cosine


loaded_data2 = np.load("target_gesture_embed_log.npy", allow_pickle=True)
tmp2 = np.mean(loaded_data2 , axis=0)
result = 0.0
for i in range(len(loaded_data2)):
    temp = cosine(tmp2 , loaded_data2[i])
    result = max(result , temp)

print(result)
print(round(result , 1))
