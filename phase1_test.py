from typing import Tuple
import cv2
from ultralytics  import YOLO
import torch
import numpy as np
import time
import os
from person_module.person_process import person_embedding , calculate_iou_opencv
from face_module.face_process import face_embedding , face_detecting ,compare_faces ,face_detecting_coordinates 
import torchreid 
from facenet_pytorch import MTCNN, InceptionResnetV1

    
# cải tiến cái này
def find_target( frame , last_bbox , output_video , target_face_embed , target_gesture_embed_log , face_extraction_model , face_detection_model  , person_detection_model, person_extraction_model) -> Tuple:
    check = False
    target_bbox = None 
    # xác định người và kiểm tra nếu người đó có trong frame
    result = person_detection_model(frame , conf = 0.5)
    if not result or len(result) == 0:
        return target_bbox
    
    for res in result:
        if not res.boxes or len(res.boxes) == 0:
            continue 
        for box in res.boxes:
            classID = int(box.cls[0])
            
            # Xét các đối tượng là con người trong frame
            if classID == 0:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

                if (last_bbox is not None) and (len(last_bbox) == 4) and isinstance(last_bbox , Tuple) :
                    iou = calculate_iou_opencv(last_bbox , (x1, y1, x2, y2))
                    print(iou)
                    if (iou > 0.7):
                        check = True
                        person_img = frame[y1:y2 , x1:x2]
                        try:
                            person_embed = person_embedding(person_img ,person_extraction_model)[0]
                        except: return None
                        target_gesture_embed_log.append(person_embed)
                        target_bbox = (x1, y1, x2, y2)
                        break
                
                person_img = frame[y1:y2 , x1:x2]
                try:
                    faces = face_detecting(person_img , 0.1 , face_detection_model)
                except: return None
                if faces is not None:
                    for face in faces:
                        try:
                            face_embed = face_embedding(face , face_extraction_model)
                        except:
                            return None
                        if face_embed is None:
                            continue
                        face_embed = face_embed[0]
                        try:
                            check = compare_faces(face_embed , target_face_embed , 0.5)
                        except: return None
                        if check is True: 
                            try:
                                person_embed = person_embedding(person_img , person_extraction_model)[0]
                            except: return None
                            target_gesture_embed_log.append(person_embed)
                            target_bbox = (x1, y1, x2, y2)
                            break
                if check is True:  break

        if check is True:  break

    if output_video.isOpened() and check == True:
        if (target_bbox is not None) and (len(target_bbox) == 4) and isinstance(target_bbox , Tuple):
            x1 , y1 , x2 , y2 = target_bbox
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0 , 255 , 0), 2)
        output_video.write(frame)
    return target_bbox

def phase1(input_path  , target_face_embed , target_gesture_embed_log , face_extraction_model , face_detection_model,  person_detection_model, person_extraction_model):
    fourcc = cv2.VideoWriter.fourcc(*'mp4v')
    input_video = cv2.VideoCapture(input_path)
    output_folder = r'assets/output' ; output_filename = r'phase1.mp4'
    output_path = os.path.join(output_folder , output_filename)
    output_video = cv2.VideoWriter(output_path, fourcc, input_video.get(cv2.CAP_PROP_FPS), 
                                    (int(input_video.get(cv2.CAP_PROP_FRAME_WIDTH)), int(input_video.get(cv2.CAP_PROP_FRAME_HEIGHT))))

    frame_count = 0
    person_bbox = None
    frame_shift_threshold = int(input_video.get(cv2.CAP_PROP_FPS) / 4)

    try:
        while True:
            ret , frame = input_video.read()
            if ret == False:
                print('Cannot read video')
                break
            else:
                if frame_count > frame_shift_threshold:
                    tmp = find_target(frame ,person_bbox , output_video , target_face_embed , target_gesture_embed_log , face_extraction_model , face_detection_model,  person_detection_model, person_extraction_model)
                    person_bbox = tmp
                    print(tmp)
                    frame_count = 0
                else:
                    if (person_bbox is not None) and output_video.isOpened() :
                        output_video.write(frame)
                        if (len(person_bbox) == 4) and isinstance(person_bbox , Tuple):
                            x1 , y1 , x2 , y2 = person_bbox
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0 , 255 , 0), 2)
                    frame_count = frame_count + 1
    finally:
        np.save("target_gesture_embed_log.npy", target_gesture_embed_log)

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
    
    
    begin_time = time.time()
    input_path =     r'assets\\input\\messi_testcase.mp4'
    target_img = cv2.imread(r'assets/input/messi4.jpg')
    target_face = face_detecting(target_img , 1 , face_detection_model)
    if len(target_face) == 1:
        target_face = target_face[0]
    target_face_embed = face_embedding(target_face)[0]
    target_gesture_embed_log = []
    average_target_embed = None
    
    phase1(input_path , target_face_embed , target_gesture_embed_log , face_extraction_model , face_detection_model , person_detection_model, person_extraction_model)
    print('Thời gian: ' ,time.time() - begin_time)  
