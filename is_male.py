import cv2
import math
import argparse
import sys
import os
from os import path
from shutil import move
from pathlib import Path, PureWindowsPath

# Usage: py is_male.py

gadpath = r"C:/Program Files/Python-3.7.6/gad/"

faceProto = gadpath + "opencv_face_detector.pbtxt"
faceModel = gadpath + "opencv_face_detector_uint8.pb"
genderProto = gadpath + "gender_deploy.prototxt"
genderModel = gadpath + "gender_net.caffemodel"

MODEL_MEAN_VALUES=(78.4263377603, 87.7689143744, 114.895847746)
genderList=['Male','Female']

faceNet=cv2.dnn.readNet(faceModel,faceProto)
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


target_dir = os.getcwd()
target_dir = os.path.join(target_dir, 'data_src')
target_dir = os.path.join(target_dir, 'aligned')
match_path = os.path.join(target_dir, 'male')

#Count how many files in the directory
file_count = len(os.listdir(target_dir))

#Show some stats
print("Checking " + str(file_count) + " files")

#Make sure the path exists and if not, create it.
if not path.isdir(match_path):
    try:
        os.mkdir(match_path)
    except OSError:
        print("Creation of the directory %s failed" % match_path)
    else:
        print("Successfully created the directory %s " % match_path)

for thisFile in os.listdir(target_dir):
    file_name = os.path.join(target_dir, thisFile)

    image = file_name

    frame = cv2.imread(image)

    if not frame is None:
        resultImg,faceBoxes=highlightFace(faceNet,frame)
        if not faceBoxes is None:
            for faceBox in faceBoxes:
                face=frame[max(0,faceBox[1]):
                           min(faceBox[3],frame.shape[0]-1),max(0,faceBox[0])
                           :min(faceBox[2], frame.shape[1]-1)]
                if not face is None and face.size > 255:
                    blob=cv2.dnn.blobFromImage(face, 1.0, (227,227), MODEL_MEAN_VALUES, swapRB=False, crop=False)
                    genderNet.setInput(blob)
                    genderPreds=genderNet.forward()
                    gender=genderList[genderPreds[0].argmax()]

                    if not gender is None and gender == 'Male':
                        if os.path.isfile(file_name):
                            move(
                                file_name, match_path)

