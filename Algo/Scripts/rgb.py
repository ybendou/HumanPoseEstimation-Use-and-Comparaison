import numpy as np
import json
import matplotlib.pyplot as plt
from collections import OrderedDict
from pprint import pprint
import matplotlib
import cv2
import argparse

############################################################################################################################################
##This script plots the skeleton of the kinect and the Mobilenet and saves the figures in a directory called frames. 					  ##
##Those frames will be later on used to make a video using the write_video script.	 													  ##
																										

parser = argparse.ArgumentParser(description='Writing a Video from frames')
parser.add_argument('--directory', type=str, default='../Données/Yassir/data_04-20-13')
parser.add_argument('--file_Kinect', type=str, default='../Données/Yassir/data_04-20-13/rgbKinect.txt')
parser.add_argument('--file_Mobilenet', type=str, default='../Données/Yassir/data_04-20-13/rgbMobilenet.txt')
parser.add_argument('--nb_frames',type=int,default=126)
args = parser.parse_args()

file_Kinect=args.file_Kinect
file_mobilenet=args.file_Mobilenet
with open(file_Kinect) as f:
    dataKinect = json.load(f, object_pairs_hook=OrderedDict)
with open(file_mobilenet) as f2:
    dataMobilenet = json.load(f2, object_pairs_hook=OrderedDict)

positions=dataKinect['positions']
positions3=dataMobilenet['positions']


common_body_parts=['Head', 'lAnkle', 'lElbow', 'lHip', 'lKnee', 'lShoulder', 'lWrist', 'mShoulder', 'rAnkle', 'rElbow', 'rHip', 'rKnee', 'rShoulder', 'rWrist']
mobilenet_maze={'Head':['mShoulder'],'mShoulder':['rShoulder','lShoulder','Head'],'rShoulder':['mShoulder','rElbow'],
                'rElbow':['rShoulder','rWrist'],'rWrist':['rElbow'],'lShoulder':['mShoulder','lElbow'],
                'lElbow':['lShoulder','lWrist'],'lWrist':['lElbow'],'rHip':['mShoulder','rKnee'],
                'rKnee':['rHip','rAnkle'],'rAnkle':['rKnee'],'lHip':['mShoulder','lKnee'],
                'lKnee':['lHip','lAnkle'],'lAnkle':['lKnee']}
Kinect_maze={'Head':['mShoulder'], 'lAnkle':['lKnee'], 'lWrist':['lElbow'],
             'lKnee':['lAnkle','lHip'], 'lShoulder':['lElbow','mShoulder'],
             'lElbow':['lShoulder','lWrist'], 'lHip':['lKnee','mShoulder'],
             'rAnkle':['rKnee'], 'rWrist':['rElbow'], 'rKnee':['rAnkle','rHip'], 
             'rShoulder':['rElbow','mShoulder'], 'rElbow':['rWrist','rShoulder'], 
             'rHip':['rKnee','mShoulder'], 'mShoulder':['Head','rShoulder','lShoulder']}



import sys
sys.path.insert(0, '../../tf-openpose')
from tf_pose import common
from matplotlib.pyplot import figure


for i in range(1,126):
    image=cv2.imread("%s/rgbjpg/%s.jpg"%(args.directory,format(i,"04")))
    first_frame=positions[str(sorted([int(pos) for pos in list(positions.keys())])[i])]
    mobilenet_pos=positions3[str(sorted([float(pos) for pos in list(positions3.keys())])[i])]
    x1=[]

    x3=[]
    y1=[]

    y3=[]
    z1=[]
    figure(num=None, figsize=(24, 18), dpi=80, facecolor='w', edgecolor='k')

    image = cv2.imread("%s/rgbjpg/%s.jpg"%(args.directory,format(i,"04")))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    for bPart in common_body_parts:
        if bPart in first_frame:
            x1.append((first_frame[bPart][1]))
            y1.append(first_frame[bPart][2])
            z1.append(first_frame[bPart][0])

    for bPart in mobilenet_pos.keys():

        x3.append(mobilenet_pos[bPart][0]*1920)
        y3.append(mobilenet_pos[bPart][1]*1080)
    #Kinect
    plt.plot(x1,y1,'ro',label='Kinect')


    for point in Kinect_maze:
        if point in first_frame:
            connected_points=Kinect_maze[point]
            for p in connected_points:
                if p in first_frame:
                    plt.plot([first_frame[p][1],first_frame[point][1]],[first_frame[p][2],first_frame[point][2]],color='red')

    #Mobilenet
    plt.plot(x3,y3,'yo',label='Mobilenet')
    for point in mobilenet_maze:
        if point in mobilenet_pos:
            connected_points=mobilenet_maze[point]
            for p in connected_points:
                if p in mobilenet_pos:
                    plt.plot([1920*mobilenet_pos[p][0],1920*mobilenet_pos[point][0]],[1080*mobilenet_pos[p][1],1080*mobilenet_pos[point][1]],color='yellow')
    
    plt.legend()
    fig=matplotlib.pyplot.gcf()
    fig.savefig('%s/frames/frame%s.png'%(args.directory,i))
    plt.clf()
#Re
#out.release()
cv2.destroyAllWindows()
