import cv2
import math
import argparse
import sys
import os
from os import path
from shutil import move, copy
from pathlib import Path, PureWindowsPath

faceProto="opencv_face_detector.pbtxt"
faceModel="opencv_face_detector_uint8.pb"
ageProto="age_deploy.prototxt"
ageModel="age_net.caffemodel"
genderProto="gender_deploy.prototxt"
genderModel="gender_net.caffemodel"

MODEL_MEAN_VALUES=(78.4263377603, 87.7689143744, 114.895847746)
ageList=['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
genderList=['Male','Female']

faceNet=cv2.dnn.readNet(faceModel,faceProto)
ageNet=cv2.dnn.readNet(ageModel,ageProto)
genderNet=cv2.dnn.readNet(genderModel,genderProto)

def highlightFace(net, frame, conf_threshold=0.7):
    frameOpencvDnn=frame.copy()
    frameHeight=frameOpencvDnn.shape[0]
    frameWidth=frameOpencvDnn.shape[1]
    blob=cv2.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [104, 117, 123], True, False)

    net.setInput(blob)
    detections=net.forward()
    faceBoxes=[]
    for i in range(detections.shape[2]):
        confidence=detections[0,0,i,2]
        if confidence>conf_threshold:
            x1=int(detections[0,0,i,3]*frameWidth)
            y1=int(detections[0,0,i,4]*frameHeight)
            x2=int(detections[0,0,i,5]*frameWidth)
            y2=int(detections[0,0,i,6]*frameHeight)
            faceBoxes.append([x1,y1,x2,y2])
            cv2.rectangle(frameOpencvDnn, (x1,y1), (x2,y2), (0,255,0), int(round(frameHeight/150)), 8)
    return frameOpencvDnn,faceBoxes


def AgeGender(frame):
    if not frame is None:
        resultImg,faceBoxes=highlightFace(faceNet,frame)

        if not faceBoxes is None:
            for faceBox in faceBoxes:
                face=frame[max(0,faceBox[1]-padding):
                           min(faceBox[3]+padding,frame.shape[0]-1),max(0,faceBox[0]-padding)
                           :min(faceBox[2]+padding, frame.shape[1]-1)]
                if not face is None and face.size > 64:
                    blob=cv2.dnn.blobFromImage(face, 1.0, (227,227), MODEL_MEAN_VALUES, swapRB=False)
                    genderNet.setInput(blob)
                    genderPreds=genderNet.forward()
                    gender=genderList[genderPreds[0].argmax()]

                    ageNet.setInput(blob)
                    agePreds=ageNet.forward()
                    age=ageList[agePreds[0].argmax()]
        else:
            print("No face detected")

    return gender, age


target_dir = os.getcwd()
target_dir = os.path.join(target_dir, 'data_src')
match_path = os.path.join(target_dir, 'match')
#file_name = sys.argv[1]
file_name = input()
image = file_name
print(f'{file_name}')

video=cv2.VideoCapture(image if image else 0)
padding=20

hasFrame,frame=video.read()

if hasFrame:
    gender, age = AgeGender(frame)
    print(f'{gender}, {age[1:-1]}')
    if gender == 'Female':
        if age == '(0-2)' or age == '(4-6)' or age== '(8-12)' or age == '(15-20)' or age == '(25-32)':
            print(f'{file_name}, {gender}, {age[1:-1]}')

            match_file = os.path.join(match_path, file_name)
            move(
                file_name, match_file)


