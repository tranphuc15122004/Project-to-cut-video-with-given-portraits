import cv2
from ultralytics  import YOLO
import torch
import numpy as np
import os
from face_module.face_process import face_embedding , face_detecting,  compare_faces
from person_module.person_process import person_detecting, person_embedding , compare_people


device = 'gpu' if torch.cuda.is_available() else 'cpu'
detection_model = YOLO('yolov8n.pt').to(device= device)
input_video = cv2.VideoCapture(r'assets\\input\\neymar_testcase.mp4')

target_img = cv2.imread(r'assets/input/neymar2.jpg')
target_gesture = (target_img)
if len(target_gesture) == 1:
    target_gesture = target_gesture[0]
target_embed = person_embedding(target_gesture)

output_folder = r'assets\\output' ; output_filename = r'test_result2.mp4'
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
            if classID == 0:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                confidence_score = box.conf.item()  # Extract the value as a float
                cv2.putText(frame, f'{confidence_score:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                person_img = frame[y1:y2 , x1:x2]
                person_embed = person_embedding(person_img)
                check = compare_people(target_embed , person_embed)
            if check is True: break
                
    if output_video.isOpened() and check == True:
        output_video.write(frame)

input_video.release()
output_video.release()
cv2.destroyAllWindows()