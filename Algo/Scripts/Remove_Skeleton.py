

##############################################################################################################
##This script removes the Skeleton from a video that has the skeleton of the mobilenet on it by removing the##
##red pixels and replacing them with a mean of a 4x4 window around the removed pixel.                       ##

import argparse
import logging
import time
import cv2
import numpy as np
import sys
import os
import logging
import matplotlib.pyplot as plt
import copy

parser = argparse.ArgumentParser(description='Writing a Video from frames')
parser.add_argument('--video_file', type=str, default='../chris1/Video_1_skeleton.mpeg')
parser.add_argument('--video_output',type=str,default='../chris1_without_skeleton.avi')
args = parser.parse_args()

cap = cv2.VideoCapture(args.video_file)
fps = cap.get(cv2.CAP_PROP_FPS)
length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print("length",length)
videotime = length / fps
print("fps",fps)
print("videotime",videotime)


fourcc = cv2.VideoWriter_fourcc('X','V','I','D')
video_output = cv2.VideoWriter(args.video_output,fourcc, 10.0, (int(cap.get(3)),int(cap.get(4))))
frame_number=1

start_time=time.time()
frame_number=1
while cap.isOpened():
    ret_val, image = cap.read()
    #We don't compute the edges, the skeleton is in the center
    aj=len(image[0])//10
    bj=len(image[0])-aj

    #For go through all the pixels
    for i in range(len(image)):
        for j in range(aj,bj):
            
            b,g,r=image[i][j]
            if (b<=85 and g<=85 and r>=120) or (b<=30 and g<=30 and r>=70) or(b<=10 and g<=10 and r>=60) : #We check if the pixel is similar to the ones on the skeleton
                matrix=[]
                #We make a mean on a 4x4 window of the surroundings pixels
                for k in [-4,-3,-2,-1,0,1,2,3,4]:
                    for s in [-4,-3,-2,-1,0,1,2,3,4]:
                        if abs(k)==4 or abs(s)==4:
                            b1,g1,r1=image[i+k][j+s]
                            #Check if the pixel is not similar to the ones on the skeleton
                            if not ((b1<=85 and g1<=85 and r1>=120) or (b1<=30 and g1<=30 and r1>=70) or(b1<=10 and g1<=10 and r1>=60)): 
                                matrix.append(image[i+k][j+s])

                matrix=np.array(matrix)
                image[i][j]=np.mean(matrix,axis=0)
    
    #Add the frame to the video   
    #cv2.imwrite('./chris1_1without_skeleton/frame%d.jpg'%frame_number,image)
    video_output.write(image)
    print("Frame:",frame_number)
    frame_number+=1
    
            
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
video_output.release()
cap.release()
cv2.destroyAllWindows()
print("end time : ",time.time()-start_time)
