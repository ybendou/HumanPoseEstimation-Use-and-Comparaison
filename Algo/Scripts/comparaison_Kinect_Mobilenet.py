import numpy as np
import json
import matplotlib.pyplot as plt
from collections import OrderedDict
from pprint import pprint
import matplotlib
import cv2
import argparse
import sys
sys.path.insert(0, '../../tf-openpose')
from tf_pose import common
from matplotlib.pyplot import figure


####################################################################################################
##This script compares between the data of the Kinect and the Mobilenet and plots several results.##
##It receivs as an input the files paths and the body part which we want to plot.                 ##

parser = argparse.ArgumentParser(description='Writing a Video from frames')
parser.add_argument('--file_Kinect', type=str, default='../Données/Yassir/data_04-20-13/rgbKinect.txt')
parser.add_argument('--file_Mobilenet',type=str,default='../Données/Yassir/data_04-20-13/rgbMobilenet.txt')
parser.add_argument('--body_part',type=str,default='lWrist')
parser.add_argument('--L',type=int,default=1920)
parser.add_argument('--H',type=int,default=1080)

args = parser.parse_args()
L=args.L
H=args.H
file_Kinect=args.file_Kinect
file_Mobilenet=args.file_Mobilenet
body_part=args.body_part

with open(file_Kinect) as f:
    dataKinect = json.load(f, object_pairs_hook=OrderedDict)
with open(file_Mobilenet) as f2:
    dataMobilenet = json.load(f2, object_pairs_hook=OrderedDict)

positions_Kinect=dataKinect['positions']
positions_Mobilenet=dataMobilenet['positions']


common_body_parts=['Head', 'lAnkle', 'lElbow', 'lHip', 'lKnee', 'lShoulder', 'lWrist', 'mShoulder', 'rAnkle', 'rElbow', 'rHip', 'rKnee', 'rShoulder', 'rWrist']

#Connecting body parts of the Mobilenet with Kinect
body_parts_Mobilenet={"Head":"Head","lAnkle":"rAnkle",'lElbow':'rElbow', 'lHip':'rHip', 'lKnee':'rKnee', 
                      'lShoulder':'rShoulder', 'lWrist':'rWrist', 'mShoulder':'mShoulder', 'rAnkle':'lAnkle', 
                      'rElbow':'lElbow', 'rHip':'lHip', 'rKnee':'lKnee', 'rShoulder':'lShoulder', 
                      'rWrist':'lWrist'}
#Body Parts of the Kinect
bPartsKinect=list(list(positions_Kinect.values())[0].keys())


#Filling the variance dictionnary including the variance between the three algorithms for each body part in all times
Variances={}
    #For each time we create a new dictionary containing all body parts and their variances
for time in positions_Mobilenet.keys():
    Variances[str(float(time))]={}
    #Filling for each body part
    for bPart in common_body_parts:

        if (body_parts_Mobilenet[bPart] in positions_Mobilenet[time].keys()) and (bPart in positions_Kinect[str(int(float(time)))].keys()):
            xM=positions_Mobilenet[time][body_parts_Mobilenet[bPart]][0]*L		#The Mobilenet Data are in pixel/total_images_pixels
            yM=positions_Mobilenet[time][body_parts_Mobilenet[bPart]][1]*H
            xK=positions_Kinect[str(int(float(time)))][bPart][1]
            yK=positions_Kinect[str(int(float(time)))][bPart][2]
            var=np.var(([xK,yK],[xM,yM]))
            Variances[str(float(time))][bPart]=var
        else:
            Variances[str(float(time))][bPart]=0

#Plot of the evolution of the variance of the tree distances for a body part

Times=list(Variances.keys())
Times_float=[]
for time in Times:
    Times_float.append(float(time))
Times_float=sorted(Times_float)
Var_bPart=[]


for time in Times_float:
    Var_bPart.append(np.sqrt(Variances[str(time)][body_parts_Mobilenet[body_part]]))
    
plt.plot(Times_float,Var_bPart,label=body_part)

plt.title('Ecart-type de la Mobilenet, Kinect')
plt.legend()
fig = matplotlib.pyplot.gcf()
fig.savefig('../Données/Courbes/Variance%s.jpg'%body_part)
plt.show()
    

#Plots the x and y positions of the body_part:

x_bPart_valuesK=[]
y_bPart_valuesK=[]

x_bPart_valuesM=[]
y_bPart_valuesM=[]
    
Times_float=[]
Times=list(Variances.keys())
Times_float=[]
for time in Times:
    Times_float.append(float(time))
Times_float=sorted(Times_float)
MbPart=body_parts_Mobilenet[bPart]

for time in Times_float:
    if bPart in positions_Kinect[str(int(time))].keys():
        xK=positions_Kinect[str(int(time))][bPart][1]
        yK=positions_Kinect[str(int(time))][bPart][2]
        
    else:
        xK=-1
        yK=-1
    x_bPart_valuesK.append(xK)
    y_bPart_valuesK.append(yK)
    if body_parts_Mobilenet[bPart] in positions_Mobilenet[str(float(time))].keys():
        xM=positions_Mobilenet[str(time)][body_parts_Mobilenet[bPart]][0]*1920
        yM=positions_Mobilenet[str(time)][body_parts_Mobilenet[bPart]][1]*1080
    else:
        xM=-1
        yM=-1
    x_bPart_valuesM.append(xM)
    y_bPart_valuesM.append(yM)

#Plotting the y values
plt.plot(Times_float,y_bPart_valuesM,'blue',label='Mobilenet')
plt.plot(Times_float,y_bPart_valuesK,'red',label='Kinect')
plt.legend()
plt.title("y values %s"%body_part)

fig.savefig('../Données/Courbes/y_values_%s.jpg'%body_part)

plt.show()

#Plotting the x values
plt.plot(Times_float,x_bPart_valuesM,'blue',label='Mobilenet')
plt.plot(Times_float,x_bPart_valuesK,'red',label='Kinect')

plt.legend()
plt.title("x values  %s"%body_part)
fig.savefig('../Données/Courbes/x_values_%s.jpg'%body_part)

plt.show()

