
##########################################################################################################################
##This script supperpose the data of the Mobilenet with the data of the Kinect automatically, it can be also be used to ##
##supperpose any skeleton data with another one as long as the two are in the same plan. In this case the Mobilenet has ##
##2D coordinates x,y while the Kinect has 3D coordinates z,x,y.

import numpy as np
import json
import scipy.interpolate
import matplotlib.pyplot as plt
from collections import OrderedDict
from pprint import pprint
import argparse

parser = argparse.ArgumentParser(description='Supperpose 2 skeletons')

parser.add_argument('--file_Kinect', type=str, default='../Données/Kinect/chris1/chris1_1.txt')
parser.add_argument('--file_Xsens', type=str, default='../Données/Mobilenet/chris1/chris1_1.txt')

args = parser.parse_args()



#Importing Mobilenet Data
file_Xsens=args.file_Xsens
with open(file_Xsens) as f2:
    dataXsens=json.load(f2,object_pairs_hook=OrderedDict)
positionsXsens=dataXsens['positions']

#Importing Kinect Data to match Mobilenet With them
file_Kinect=args.file_Kinect
with open(file_name) as f:
    dataKinect=json.load(f,object_pairs_hook=OrderedDict)
positionsKinect=dataKinect['positions']

#We look for the first frame to get the match

first_frame_Kinect=positionsKinect[list(positionsKinect.keys())[0]]
first_frame_Xsens=positionsXsens[list(positionsXsens.keys())[0]]


Times2=list(dataXsens['positions'].keys())
positions2=dataXsens['positions']

teta=np.pi/11-np.pi/50
Rotation_y=np.array([[np.cos(teta),-np.sin(teta),0],[np.sin(teta),np.cos(teta),0],[0,0,1]])
translateXSens=[0.003499753200566387, 0.5451910735626221]

for frame in positions2.keys():
    frame_pos=positions2[frame]

    #For Xsens : rotation then translation
   
    for bPart in frame_pos.keys():
        pos=frame_pos[bPart]
        pos_float=[]
        for coord in pos:
            pos_float.append(float(coord))
        frame_pos[bPart]=np.dot(pos_float,Rotation_y)

    #Scaling
    scale=[0.9,1]
    for bPart in frame_pos.keys():
        pos=frame_pos[bPart]

        scale_x=pos[1]*scale[0]
        scale_y=pos[2]*scale[1]
        frame_pos[bPart]=[pos[0],scale_x,scale_y]
        
    #Translating
    for bPart in frame_pos.keys():
        pos=frame_pos[bPart]
        pos_float=[]
        for coord in pos:
            pos_float.append(float(coord))
        trans_x=-pos_float[1]+translateXSens[0]-2*0.006724384613335133
        trans_y=pos_float[2]-translateXSens[1]
        frame_pos[bPart]=[pos_float[0],trans_x,trans_y]

XsensL=list(positions2.keys())
XsensL_float=[]
for i in XsensL:
    XsensL_float.append(int(i))
XsensL_float=sorted(XsensL_float)


new_Xsens_Times=[]
Reference_Times=[t/58.75 for t in range(1,1629)]
index=0
dict_Time={}
new_pos={}
for time in Reference_Times:
    dict_Time[time]=[]
    list_times=[]
    while XsensL_float[index]/1000<time:
        dict_Time[time].append(XsensL_float[index])            
        list_times.append(XsensL_float[index])
        index+=1
    real_time=list_times[-1]
    new_pos[time]=positions2[str(real_time)]
    
dataXsens['positions']=new_pos

new_file_name=re.sub(".txt","_transformed.txt",file_Xsens)
with open(new_file_name, 'w') as outfile:
     json.dump(dataXsens, outfile, sort_keys = True, indent = 4,
               ensure_ascii = False)