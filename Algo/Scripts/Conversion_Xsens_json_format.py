import numpy as np
import json
import scipy.interpolate
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
from pprint import pprint
import argparse
import re
######################################################################################################
##This script converts the data of the Xsens which are in xml format to a json format in a txt file.##
##Only the positions are taken.

parser = argparse.ArgumentParser(description='Writing a Video from frames')
parser.add_argument('--file_name', type=str, default='../Données/Xsens/chris1/chris1_1.mvnx')

args = parser.parse_args()

#Path of the file :
file_name=args.file_name
print(file_name)

#Getting the data
tree = ET.parse(file_name)
root = tree.getroot()


position={}

common_body_parts=['Head', 'lAnkle', 'lElbow', 'lHip', 'lKnee', 'lShoulder', 'lWrist', 'mShoulder', 'rAnkle', 'rElbow', 'rHip', 'rKnee', 'rShoulder', 'rWrist']

body_parts={"T8":5,"Head":7,"RightShoulder":8,"RightUpperArm":9,"RightForeArm":10,"RightHand":11,"LeftShoulder":12,"LeftUpperArm":13,"LeftForeArm":14,"LeftHand":15,"RightUpperLeg":16,"RightLowerLeg":17,"RightFoot":18,"LeftUpperLeg":20,"LeftLowerLeg":21,"LeftFoot":22}

data={'positions':{}} #New format of data
positions=data['positions']	#Matching the data in the same format of the Kinect, only the positions are taken
list_body_parts=list(body_parts.keys())
Indexs=[]
Times=[]
for frame in root[2][2][2:]:
    Indexs.append(frame.get("index"))
    time=frame.get("time")
    Times.append(time)
    positions[time]={}
    All_frame_positions=frame[1].text.split()
    for b_part in list_body_parts:
        part_id=int(body_parts[b_part])
        x=All_frame_positions[3*part_id-3]
        y=All_frame_positions[3*part_id-2]
        z=All_frame_positions[3*part_id-1]
        positions[time][b_part]=[x,y,z]
        
#Save the data in a txt file format
new_file_name=re.sub(".mvnx",".txt",file_name)
with open(new_file_name, 'w') as outfile:
     json.dump(data, outfile, sort_keys = True, indent = 4,
               ensure_ascii = False)
