import numpy as np
import json
import matplotlib.pyplot as plt
from collections import OrderedDict
from pprint import pprint
import matplotlib
import cv2
import argparse

############################################################################################################################################
##This script converts the skeleton2D data of the kinect into the json/txt dictionary format of the Kinect, then plots the skeleton of the##
##kinect and the Mobilenet and saves the figures in a directory called frames. Those frames will be later on used to make a video using   ## 
##the write_video script.																												  ##		

parser = argparse.ArgumentParser(description='Writing a Video from frames')
parser.add_argument('--file_name', type=str, default='./Données/Yassir/data_04-20-13/skeletons2D.txt')
parser.add_argument('--directory', type=str, default='./Données/Yassir/data_04-20-13')

args = parser.parse_args()

file_name=args.file_name
text=open(file_name,'r')

dataKinect={}
positions={}
body_Parts={2:'mShoulder',3:'Head',4:'lShoulder',5:'lElbow',6:'lWrist',8:'rShoulder',9:'rElbow',10:'rWrist',12:'lHip',13:'lKnee',14:'lAnkle',16:'rHip',17:'rKnee',18:'rAnkle'}
lines = text.readlines()   
frame_number=1
for line in lines:
    list_line=line.split(' ')
    positions[str(frame_number)]={}
    for bPart in body_Parts.keys():
        x=float(list_line[2*bPart])
        y=float(list_line[2*bPart+1])
        if 0<=x<=1920 and 0<=y<=1080:
            positions[str(frame_number)][body_Parts[bPart]]=[0,x,y]

            
    frame_number+=1
dataKinect['positions']=positions

#Saving the new file format:
new_file_name=re.sub(".txt","_interpolated.txt",file_name)
with open("%s/rgbKinect.txt"%args.directory, 'w') as outfile:
             json.dump(dataKinect, outfile, sort_keys = True, indent = 4,
                       ensure_ascii = False)