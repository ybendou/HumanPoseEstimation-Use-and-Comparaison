import numpy as np
import json
import scipy.interpolate
import matplotlib.pyplot as plt
from collections import OrderedDict
from pprint import pprint
import argparse
import re
##########################################################################################################################
##This script supperpose the data of the Mobilenet with the data of the Kinect automatically, it can be also be used to ##
##supperpose any skeleton data with another one as long as the two are in the same plan. In this case the Mobilenet has ##
##2D coordinates x,y while the Kinect has 3D coordinates z,x,y.


parser = argparse.ArgumentParser(description='Writing a Video from frames')
parser.add_argument('--file_Kinect', type=str, default='../Données/Kinect/chris1/chris1_1.txt')
parser.add_argument('--file_Mobilenet', type=str, default='../Données/Mobilenet/chris1/chris1_1.txt')
args = parser.parse_args()


#Importing Mobilenet Data
file_Mobilenet=args.file_Mobilenet
with open(file_Mobilenet) as f3:
    dataMobilenet=json.load(f3,object_pairs_hook=OrderedDict)
positionsMobilenet=dataMobilenet['positions']

#Importing Kinect Data to match Mobilenet With them
file_Kinect=args.file_Kinect
with open(file_Kinect) as f:
    dataKinect=json.load(f,object_pairs_hook=OrderedDict)
positionsKinect=dataKinect['positions']

#We look for the first frame to get the match

first_frame_Kinect=positionsKinect[list(positionsKinect.keys())[0]]
ind=0
first_frame_Mobilenet=positionsMobilenet[list(positionsMobilenet.keys())[ind]]

while ('rKnee' or 'rAnkle' or 'lKnee' or 'lAnkle') not in first_frame_Mobilenet.keys():
    ind+=1
    first_frame_Mobilenet=positionsMobilenet[list(positionsMobilenet.keys())[ind]]

    

#We get the coords of the knees and the ankles

x_RkneeK,y_RkneeK=float(first_frame_Kinect['rKnee'][1]),float(first_frame_Kinect['rKnee'][2])
x_RAnkleK,y_RAnkleK=float(first_frame_Kinect['rAnkle'][1]),float(first_frame_Kinect['rAnkle'][2])
x_LkneeK,y_LkneeK=float(first_frame_Kinect['lKnee'][1]),float(first_frame_Kinect['lKnee'][2])
x_LAnkleK,y_LAnkleK=float(first_frame_Kinect['lAnkle'][1]),float(first_frame_Kinect['lAnkle'][2])

x_RkneeM,y_RkneeM=first_frame_Mobilenet['rKnee'][0],first_frame_Mobilenet['rKnee'][1]
x_RAnkleM,y_RAnkleM=first_frame_Mobilenet['rAnkle'][0],first_frame_Mobilenet['rAnkle'][1]
x_LkneeM,y_LkneeM=first_frame_Mobilenet['lKnee'][0],first_frame_Mobilenet['lKnee'][1]
x_LAnkleM,y_LAnkleM=first_frame_Mobilenet['lAnkle'][0],first_frame_Mobilenet['lAnkle'][1]

#Compute the distance of the right and left legs for the Kinect and the Mobilenet
distance_RightM=np.sqrt((x_RkneeM-x_RAnkleM)**2+(y_RkneeM-y_RAnkleM)**2)
distance_LeftM=np.sqrt((x_LkneeM-x_LAnkleM)**2+(y_LkneeM-y_LAnkleM)**2)

distance_RightK=np.sqrt((x_RkneeK-x_RAnkleK)**2+(y_RkneeK-y_RAnkleK)**2)
distance_LeftK=np.sqrt((x_LkneeK-x_LAnkleK)**2+(y_LkneeK-y_LAnkleK)**2)

#We make a mean on the distance between the ratio of the right and the left legs to get the right scale
scale=((distance_RightK/distance_RightM)+(distance_LeftK/distance_LeftM))/2

##############Translation#########################################

#Computing the first frame in order to obtain the correct translation to make : 
frame=list(positionsMobilenet.keys())[0]
    
#Scaling the first frame    
pos=first_frame_Mobilenet['mShoulder']

scale_x=pos[0]*2*scale
scale_y=pos[1]*scale

#We get the mShoulder pos for the Kinect and the Mobilenet
mShoulder_Kinect_pos=first_frame_Kinect['mShoulder']
mShoulder_Mobilenet_pos=[scale_x,scale_y]

#Compute the translation to make to match the mShoulders
translateMobilenet=[mShoulder_Mobilenet_pos[0]-mShoulder_Kinect_pos[1],mShoulder_Mobilenet_pos[1]-mShoulder_Kinect_pos[2]]
###################################################################

for frame in positionsMobilenet.keys():
    mobilenet_pos=positionsMobilenet[frame]
    
    #Scaling:
    
    for bPart in mobilenet_pos.keys():
        pos=mobilenet_pos[bPart]

        scale_x=pos[0]*2*scale
        scale_y=pos[1]*scale
        mobilenet_pos[bPart]=[scale_x,scale_y]
   
    
    #Translation
    for bPart in mobilenet_pos.keys():
        pos=mobilenet_pos[bPart]

        trans_x=pos[0]-translateMobilenet[0]       #2*mShoulder_Kinect_pos[1]
        trans_y=2*mShoulder_Kinect_pos[2]-pos[1]+translateMobilenet[1]
        mobilenet_pos[bPart]=[trans_x,trans_y]
        
    positionsMobilenet[frame]=mobilenet_pos    #Save the new pos in the positions dictionary   

dataMobilenet['positions']=positionsMobilenet  #Save the new data
#Saving the output in a dictionary txt format
new_file_name=re.sub(".txt","_transformed.txt",file_Mobilenet)
with open(new_file_name, 'w') as outfile:
     json.dump(dataMobilenet, outfile, sort_keys = True, indent = 4,
               ensure_ascii = False)
    
