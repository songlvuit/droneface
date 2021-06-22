# import the necessary libs
import numpy as np
import argparse
import time
import cv2
import os
import sys
from PIL import Image

cwd = os.getcwd()
print("cwd", cwd)

sys.path.append(cwd + "/" + "libs")
sys.path.append(cwd + "/" + "libs/insight_face/InsightFace_Pytorch")
sys.path.append(cwd + "/" + "libs/insight_face")
print("path sys: ")
for line in sys.path:
    print(line)

from libs import yolo_detection
from libs.insight_face.InsightFace_Pytorch import face_verify

#yolo_detection.test_import()

net, ln = yolo_detection.init_net()
print("TEST - yolo detection NET initted")

conf = face_verify.init_config()
mtcnn = face_verify.init_MTCNN()
learner = face_verify.init_learner(conf)
targets, names = face_verify.init_facebank(conf=conf, learner=learner, mtcnn=mtcnn)


# on single input image
def detection(image):
    print("START TEST FACE DETECTION RECOG")
    
    draw_image = image.copy()

    boxes = yolo_detection.detect_bboxes(net, ln, image)
    
    faces = []
    for i in range(len(boxes)):
        # extract the bounding box coordinates
        (x, y) = (boxes[i][0], boxes[i][1])
        (w, h) = (boxes[i][2], boxes[i][3])

        # crop and save face
        DELTA_y = int(0.1 * h)
        DELTA_x = int(0.2 * w)
		#crop_face = image[y:y+h,x:x+w].copy()  
        crop_face = image[y-DELTA_y*2:y+h+DELTA_y,x-DELTA_x:x+w+DELTA_x].copy()

        #cv2.imwrite("test/output/output_crop/crop_face_{}.png".format(str(i)), crop_face)
        try:
            # crop_face = mtcnn.align(crop_face)
            pillow_image = Image.fromarray(cv2.cvtColor(crop_face, cv2.COLOR_BGR2RGB))
            face = pillow_image.resize((112,112))
            faces.append(face)
        except:
            pass
    face_ids = range(len(faces))
    min_face_id, min_face_score = face_verify.verify_faces(conf=conf, learner=learner, targets=targets, faces=faces, face_ids=face_ids)
    
    (x, y) = (boxes[min_face_id][0], boxes[min_face_id][1])
    (w, h) = (boxes[min_face_id][2], boxes[min_face_id][3])
    cropface = draw_image[y:y+h, x:x+w]
    cv2.rectangle(draw_image, (x, y), (x + w, y + h), (0,255,0), 3)
    
    return draw_image, cropface 