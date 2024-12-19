from typing import Tuple
import cv2
from ultralytics  import YOLO
import torch
import numpy as np
import os
import tensorflow as tf
from face_module.face_process import face_embedding , face_detecting , target_face_detect, compare_faces
from person_module.person_process import  person_embedding , compare_people , calculate_iou_opencv
import time
from scipy.spatial.distance import cosine
import torchreid 
from facenet_pytorch import MTCNN, InceptionResnetV1

def find_target(frame, last_bbox ,output_video , max_bound ,average_target_embed  , token , target_face_embed  , face_extraction_model , face_detection_model , person_detection_model ,person_extraction_model):
    face_check = False
    gesture_check = False
    target_bbox = None

    # xác định người và kiểm tra nếu người đó có trong frame
    result = person_detection_model(frame , conf = 0.5)
    if not result or len(result) == 0:
        return target_bbox , token
    person_images = []
    
    for res in result:
        for box in res.boxes:
            classID = int(box.cls[0])
            if classID == 0:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())    
                person_img = frame[y1:y2 , x1:x2] 
                person_images.append(person_img)         
                if (last_bbox is not None) and (len(last_bbox) == 4) and isinstance(last_bbox , Tuple):
                    if token == 0:
                        try:
                            face_check = target_face_detect(person_img , target_face_embed , face_detection_model)
                        except: return None , token
                        print('Check FACE')
                        if face_check:
                            token = token + 6
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0 , 255 , 0), 6)
                            target_bbox = (x1, y1, x2, y2)
                            print('TOKEN ADDING')
                            break
                    else:
                        iou = calculate_iou_opencv(last_bbox , (x1, y1, x2, y2))
                        print(iou)
                        if (iou > 0.7):
                            gesture_check = True
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0 , 255 , 0), 6)
                            target_bbox = (x1, y1, x2, y2)
                            if token > 0: token = token - 1
                            break
                else:
                    try:
                        faces = face_detecting(person_img, 1 , face_detection_model)
                    except: return None , token
                    if faces is not None:
                        for face in faces:
                            try:
                                face_embed = face_embedding(face , face_extraction_model)
                            except: return None , token
                            if face_embed is None:
                                continue
                            face_embed = face_embed[0]
                            try:
                                check = compare_faces(face_embed , target_face_embed , 0.5)
                            except: return None , token
                            if check: 
                                face_check = True
                                token = 6
                                target_bbox = (x1, y1, x2, y2)
                                if (target_bbox is not None) and (len(target_bbox) == 4) and isinstance(target_bbox , Tuple):
                                    x1 , y1 , x2 , y2 = target_bbox
                                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0 , 255 , 0), 6)
                                print('TOKEN ADDING')
                                break
        if face_check or gesture_check:  break
    
    if not (face_check or gesture_check):
        best_fit_score = max_bound
        for person_img in person_images:
            try:
                person_embed = person_embedding(person_img , person_extraction_model)[0]
                gesture_check , tmp_score = compare_people(person_embed ,average_target_embed , 0.5)
            except: return None , token
            
            if tmp_score < best_fit_score:
                target_bbox = (x1 , y1 , x2 , y2)
                best_fit_score = tmp_score
            if face_check or gesture_check:  break

    if output_video.isOpened() and (face_check  or gesture_check):
        output_video.write(frame)
        if (target_bbox is not None) and (len(target_bbox) == 4) and isinstance(target_bbox , Tuple):
            x1 , y1 , x2 , y2 = target_bbox
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0 , 255 , 0), 6)
            
    return target_bbox , token


def phase2(input_path , output_path  , max_bound ,  average_target_embed ,target_face_embed , face_extraction_model , face_detection_model , person_detection_model ,person_extraction_model ): 
    fourcc = cv2.VideoWriter.fourcc(*'mp4v')
    input_video = cv2.VideoCapture(input_path) 
    output_video = cv2.VideoWriter(output_path, fourcc, input_video.get(cv2.CAP_PROP_FPS), 
                                    (int(input_video.get(cv2.CAP_PROP_FRAME_WIDTH)), int(input_video.get(cv2.CAP_PROP_FRAME_HEIGHT))))
    person_bbox = None
    frame_count = 0
    frame_shift_threshold = int(input_video.get(cv2.CAP_PROP_FPS) / 6)
    token = 0
    while True:
        ret , frame = input_video.read()
        if ret == False:
            print('Cannot read video')
            break
        else:
            if frame_count >= frame_shift_threshold:
                tmp , token_tmp = find_target(frame , person_bbox , output_video , max_bound , average_target_embed  , token ,target_face_embed , face_extraction_model , face_detection_model , person_detection_model ,person_extraction_model)
                person_bbox = tmp 
                token = token_tmp
                print(tmp)
                if(person_bbox is not None) and (len(person_bbox) == 4) and isinstance(person_bbox , Tuple):
                    x1 , y1 , x2 , y2 = person_bbox
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0 , 255 , 0), 6)
                frame_count = 0
            else:
                if (person_bbox is not None) and output_video.isOpened() :
                    output_video.write(frame)
                    if (len(person_bbox) == 4) and isinstance(person_bbox , Tuple):
                        x1 , y1 , x2 , y2 = person_bbox
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0 , 255 , 0), 6)
                frame_count = frame_count +1 

    input_video.release()
    output_video.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    device = 'gpu' if torch.cuda.is_available() else 'cpu'
    face_extraction_model = InceptionResnetV1(pretrained= 'vggface2' ,device=device).eval()
    face_detection_model = MTCNN(keep_all= True , device= device)
    person_extraction_model = torchreid.utils.FeatureExtractor(
        model_name= 'osnet_x1_0',
        device= device
    )
    person_detection_model = YOLO('yolov8n.pt')
    person_detection_model.to(device)
    
    
    input_path = r'assets\\input\\lionel-messi-vs-atletico-madrid-2018-19-home-4k-uhd-english-commentary-1080-publer.io.mp4'
    output_folder = r'assets/output' ; output_filename = r'messi_test11.mp4'
    output_path = os.path.join(output_folder , output_filename)
    fourcc = cv2.VideoWriter.fourcc(*'mp4v')
    
    
    # Các khai báo của đối tượng
    target_img = cv2.imread(r'assets/input/messi4.jpg')
    target_face = face_detecting(target_img , 1 , face_detection_model)
    if len(target_face) == 1:
        target_face = target_face[0]
    target_face_embed = face_embedding(target_face , face_extraction_model)
    loaded_data = np.load("target_gesture_embed_log_test.npy", allow_pickle=True)
    average_target_embed = np.mean(loaded_data , axis=0)
    max_bound = 0.0
    
    for gesture_embed in loaded_data:
        temp = cosine(average_target_embed , gesture_embed)
        max_bound = max(max_bound ,temp)
    max_bound = round(max_bound +0.05, 1)
    max_bound = max(max_bound, 0.3)
    
    
    begin_time = time.time()
    phase2(input_path, output_path  ,max_bound , average_target_embed , target_face_embed , face_extraction_model , face_detection_model , person_detection_model ,person_extraction_model)
    print('Thời gian: ', time.time() - begin_time)
