import cv2
from ultralytics import YOLO
from person_module.person_process import  person_embedding, compare_people
import torch
from face_module.face_process import face_detecting , face_embedding , compare_faces
import numpy as np
import os
device = 'gpu' if torch.cuda.is_available() else 'cpu'
detection_model = YOLO('yolov8n.pt').to(device= device)
input_video = cv2.VideoCapture(r'assets/input/messi_testcase.mp4')

target_img = cv2.imread(r'assets/input/messi4.jpg')
target_face = face_detecting(target_img)
if len(target_face) == 1:
    target_face = target_face[0]
target_face_embed = face_embedding(target_face)
target_gesture_embed_log = []
average_target_embed = None

fourcc = cv2.VideoWriter.fourcc(*'mp4v')

# phase thu thập dữ liệu về dáng dựa trên khuôn mặt
def phase1():
    output_folder = r'assets\\output' ; output_filename = r'messi_testcase_phase1.mp4'
    output_path = os.path.join(output_folder , output_filename)
    fourcc = cv2.VideoWriter.fourcc(*'mp4v')
    output_video = cv2.VideoWriter(output_path, fourcc, input_video.get(cv2.CAP_PROP_FPS), 
                                    (int(input_video.get(cv2.CAP_PROP_FRAME_WIDTH)), int(input_video.get(cv2.CAP_PROP_FRAME_HEIGHT))))

    while True:
        ret , frame = input_video.read()
        if ret == False:
            print('Cannot read video')
            break
        check = False
            
        # xác định người và kiểm tra nếu người đó có trong frame
        result = detection_model(frame , conf = 0.5)
        for res in result:
            for box in res.boxes:
                classID = int(box.cls[0])
                
                # Xét các đối tượng là con người trong frame
                if classID == 0:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    confidence_score = box.conf.item()  # Extract the value as a float
                    cv2.putText(frame, f'{confidence_score:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                    person_img = frame[y1:y2 , x1:x2]
                    person_embed = person_embedding(person_img)
                    person_embed = person_embed[0]
                    
                    faces = face_detecting(person_img)
                    if faces is not None:
                        for face in faces:
                            face_embed = face_embedding(face)
                            if face_embed is None:
                                continue
                            check = compare_faces(face_embed , target_face_embed)
                            if check is True: 
                                target_gesture_embed_log.append(person_embed)
                                break
                    if check is True:  break
                        
        if output_video.isOpened() and check == True:
            output_video.write(frame)
            

    input_video.release()
    output_video.release()
    cv2.destroyAllWindows()
    
# Phase phân tích video
def phase2():
    pass

if __name__ == "__main__":
    phase1()
    print ('############### KẾT THÚC PHASE I #################')
    np.save("target_gesture_embed_log.npy", target_gesture_embed_log)
    average_target_embed = np.mean(target_gesture_embed_log , axis= 0)
    phase2()