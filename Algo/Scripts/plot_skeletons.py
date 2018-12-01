import numpy as np
import json
import scipy.interpolate
import matplotlib.pyplot as plt
from collections import OrderedDict
from pprint import pprint
import argparse
##################################################################################################################
## This script allows to plot the skeletons of the mobilenet,kinect and Xsens which are in a json/txt format	##

parser = argparse.ArgumentParser(description='Writing a Video from frames')
parser.add_argument('--file_Kinect', type=str, default="../Données/Kinect/chris1/chris1_1_interpolated.txt")
parser.add_argument('--file_Mobilenet',type=str,default="../Données/Mobilenet/chris1/chris1_1_interpolated.txt")
parser.add_argument('--file_Xsens',type=str,default="../Données/Xsens/chris1/chris1_1_interpolated.txt")
parser.add_argument('--frame_index',type=int,default=1)
args = parser.parse_args()

#Importing the Kinect data
file_Kinect=args.file_Kinect
print(file_Kinect)
with open(file_Kinect) as f:
    data = json.load(f, object_pairs_hook=OrderedDict)
Times=list(data['positions'].keys())
positions=data['positions']



#Importing the Xsens data

file_Xsens=args.file_Xsens
print(file_Xsens)
with open(file_Xsens) as f2:
    data2 = json.load(f2, object_pairs_hook=OrderedDict)
Times2=list(data2['positions'].keys())
positions2=data2['positions']



#Importing the Mobilenet data
file_Mobilenet=args.file_Mobilenet
with open(file_Mobilenet) as f3:
    dataMobilenet=json.load(f3,object_pairs_hook=OrderedDict)
positions3=dataMobilenet['positions']




#Only the common body parts are plotted
common_body_parts=['Head', 'lAnkle', 'lElbow', 'lHip', 'lKnee', 'lShoulder', 'lWrist', 'mShoulder', 'rAnkle', 'rElbow', 'rHip', 'rKnee', 'rShoulder', 'rWrist']

#The maze of the skeletons, this maze is used to link the joints between them
mobilenet_maze={'Head':['mShoulder'],'mShoulder':['rShoulder','lShoulder','Head'],'rShoulder':['mShoulder','rElbow'],
                'rElbow':['rShoulder','rWrist'],'rWrist':['rElbow'],'lShoulder':['mShoulder','lElbow'],
                'lElbow':['lShoulder','lWrist'],'lWrist':['lElbow'],'rHip':['mShoulder','rKnee'],
                'rKnee':['rHip','rAnkle'],'rAnkle':['rKnee'],'lHip':['mShoulder','lKnee'],
                'lKnee':['lHip','lAnkle'],'lAnkle':['lKnee']}

Xsens_maze={'Head':['T8'], 'LeftFoot':['LeftLowerLeg'], 'LeftForeArm':['LeftUpperArm','LeftHand'],
            'LeftHand':['LeftForeArm'], 'LeftLowerLeg':['LeftFoot','LeftUpperLeg'], 'LeftShoulder':['LeftUpperArm','T8'],
            'LeftUpperArm':['LeftShoulder','LeftForeArm'], 'LeftUpperLeg':['LeftLowerLeg','T8'], 
            'RightFoot':['RightLowerLeg'], 'RightForeArm':['RightUpperArm','RightHand'],'RightHand':['RightForeArm'],
            'RightLowerLeg':['RightFoot','RightUpperLeg'], 'RightShoulder':['RightUpperArm','T8'],
            'RightUpperArm':['RightForeArm','RightShoulder'], 'RightUpperLeg':['RightLowerLeg','T8'], 
            'T8':['Head','RightShoulder','LeftShoulder']}

Kinect_maze={'Head':['mShoulder'], 'lAnkle':['lKnee'], 'lWrist':['lElbow'],
             'lKnee':['lAnkle','lHip'], 'lShoulder':['lElbow','mShoulder'],
             'lElbow':['lShoulder','lWrist'], 'lHip':['lKnee','mShoulder'],
             'rAnkle':['rKnee'], 'rWrist':['rElbow'], 'rKnee':['rAnkle','rHip'], 
             'rShoulder':['rElbow','mShoulder'], 'rElbow':['rWrist','rShoulder'], 
             'rHip':['rKnee','mShoulder'], 'mShoulder':['Head','rShoulder','lShoulder']}

#Plotting:
frames_index=[args.frame_index]
for i in frames_index:
    first_frame2=positions2[str(sorted([float(pos) for pos in list(positions2.keys())])[i])]
    first_frame=positions[str(sorted([float(pos) for pos in list(positions.keys())])[i])]
    mobilenet_pos=positions3[str(sorted([float(pos) for pos in list(positions3.keys())])[i])]
    x1=[]
    x2=[]
    x3=[]
    y1=[]
    y2=[]
    y3=[]
    z1=[]
    z2=[]
    for bPart in common_body_parts:
        x1.append((first_frame[bPart][1]))
        y1.append(first_frame[bPart][2])
        z1.append(first_frame[bPart][0])
    for bPart in first_frame2.keys():
        x2.append(float(first_frame2[bPart][1]))
        y2.append(float(first_frame2[bPart][2]))
        z2.append(first_frame2[bPart][0])
    for bPart in mobilenet_pos.keys():

        x3.append(mobilenet_pos[bPart][0])
        y3.append(mobilenet_pos[bPart][1])
    #Kinect

    plt.plot(x1,y1,'ro')	#Plotting the joints

    for point in Kinect_maze:	#Plotting the skeleton
        connected_points=Kinect_maze[point]
        for p in connected_points:
            plt.plot([first_frame[p][1],first_frame[point][1]],[first_frame[p][2],first_frame[point][2]],color='red')


    #Xsens

    plt.plot(x2,y2,'go')	#Plotting the joints

    for point in Xsens_maze:	#Plotting the skeleton
        connected_points=Xsens_maze[point]
        for p in connected_points:
            plt.plot([first_frame2[p][1],first_frame2[point][1]],[first_frame2[p][2],first_frame2[point][2]],color='green')

    #Mobilenet
    plt.plot(x3,y3,'bo')	#Plotting the joints
    for point in mobilenet_maze:	#Plotting the skeleton
        if point in mobilenet_pos:
            connected_points=mobilenet_maze[point]
            for p in connected_points:
                if p in mobilenet_pos:
                    plt.plot([mobilenet_pos[p][0],mobilenet_pos[point][0]],[mobilenet_pos[p][1],mobilenet_pos[point][1]],color='blue')
    
    plt.show()


