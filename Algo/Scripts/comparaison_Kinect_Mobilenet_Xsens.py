import numpy as np
import json
import scipy.interpolate
import matplotlib.pyplot as plt
from collections import OrderedDict
from pprint import pprint
import matplotlib
import argparse

##################################################################################################################
## This script allows compare between the data of the Xsens, Mobilenet and the Kinect.							##

parser = argparse.ArgumentParser(description='Writing a Video from frames')
parser.add_argument('--file_Kinect', type=str, default="../Données/Kinect/chris1/chris1_1_interpolated.txt")
parser.add_argument('--file_Mobilenet',type=str,default="../Données/Mobilenet/chris1/chris1_1_interpolated.txt")
parser.add_argument('--file_Xsens',type=str,default="../Données/Xsens/chris1/chris1_1_interpolated.txt")
parser.add_argument('--body_part',type=str,default='lEblow')
args = parser.parse_args()

#All the files paths
file_kinect=args.file_Kinect
file_xsens=args.file_Xsens
file_mobilenet=args.file_Mobilenet


#We import all the files in a json format 
with open(file_kinect) as f1:
    dataKinect = json.load(f1, object_pairs_hook=OrderedDict)
with open(file_xsens) as f2:
    dataXsens = json.load(f2, object_pairs_hook=OrderedDict)
with open(file_mobilenet) as f3:
    dataMobilenet = json.load(f3, object_pairs_hook=OrderedDict)



#We collect the positions and copy them in variables
positions_Kinect=dataKinect['positions']
positions_Xsens=dataXsens['positions']
positions_Mobilenet=dataMobilenet['positions']


body_part=args.body_part
#Connecting body parts of the Xsens with Kinect
body_parts_Xsens={"Head":"Head","mShoulder":"T8","rShoulder":"RightUpperArm","rElbow":"RightForeArm",
                  "rWrist":"RightHand","lShoulder":"LeftUpperArm","lElbow":"LeftForeArm","lWrist":"LeftHand",
                  "rHip":"RightUpperLeg","rKnee":"RightLowerLeg","rAnkle":"RightFoot","lHip":"LeftUpperLeg",
                  "lKnee":"LeftLowerLeg","lAnkle":"LeftFoot"}

#Fixing the problem of right and left between the Mobilenet and the Kinect
body_parts_Mobilenet={"Head":"Head","lAnkle":"rAnkle",'lElbow':'rElbow', 'lHip':'rHip', 'lKnee':'rKnee', 
                      'lShoulder':'rShoulder', 'lWrist':'rWrist', 'mShoulder':'mShoulder', 'rAnkle':'lAnkle', 
                      'rElbow':'lElbow', 'rHip':'lHip', 'rKnee':'lKnee', 'rShoulder':'lShoulder', 
                      'rWrist':'lWrist'}
#Body Parts of the Kinect
bPartsKinect=list(list(positions_Kinect.values())[0].keys())
common_body_parts=['Head', 'lAnkle', 'lElbow', 'lHip', 'lKnee', 'lShoulder', 'lWrist', 'mShoulder', 'rAnkle', 
                   'rElbow', 'rHip', 'rKnee', 'rShoulder', 'rWrist']

#Filling the variance dictionnary including the variance between the three algorithms for each body part in all times
Variances={}
    #For each time we create a new dictionary containing all body parts and their variances
for time in positions_Kinect.keys():
    Variances[time]={}
    #Filling for each body part
    for bPart in common_body_parts:
        #Since the Xsens has different body parts names, we look for its equivalent in the body_parts_Xsens dictionnary
        XbPart=body_parts_Xsens[bPart]
        
        #Since the Right and Left of the Mobilenet and Kinect are opposite, we look for its opposite the body_parts_Mobilenet dictionnary
        var=np.var((positions_Kinect[time][bPart][1:],positions_Mobilenet[time][body_parts_Mobilenet[bPart]],positions_Xsens[time][XbPart][1:]))

        Variances[time][bPart]=var






#Plot of the evolution of the variance of the tree distances for a body part
bPart=body_parts_Mobilenet[body_part]

Times=list(Variances.keys())
Times_float=[]
for time in Times:
    Times_float.append(float(time))
Times_float=sorted(Times_float)
Var_bPart=[]


for time in Times_float:
    Var_bPart.append(Variances[str(time)][body_parts_Mobilenet[body_part]])

plt.plot(Times_float,Var_bPart,label=body_part)

plt.title('Variance de la Mobilenet, Kinect et Xsens')
plt.legend()
fig = matplotlib.pyplot.gcf()
fig.savefig('../Données/Courbes/Variance_%s.jpg'%body_part)
plt.show()

#Comparaison with the terrain field (Xsens) Variances
Difference_Mobilenet=[]
Difference_Kinect=[]
bPart=body_parts_Mobilenet[body_part]

Times_float=[]
Times=list(Variances.keys())
Times_float=[]
for time in Times:
    Times_float.append(float(time))
Times_float=sorted(Times_float)

for time in Times_float:
    XbPart=body_parts_Xsens[bPart]
    MbPart=body_parts_Mobilenet[bPart]
    diff_Mobilenet=np.sqrt(np.var((positions_Mobilenet[str(time)][MbPart][:],positions_Xsens[str(time)][XbPart][1:])))
    diff_Kinect=np.sqrt(np.var((positions_Kinect[str(time)][bPart][1:],positions_Xsens[str(time)][XbPart][1:])))
    Difference_Mobilenet.append(diff_Mobilenet)
    Difference_Kinect.append(diff_Kinect)
    
plt.plot(Times_float,Difference_Mobilenet,color='blue',label='Var Mob-Xs')
plt.plot(Times_float,Difference_Kinect,color='red',label='Var Kinect-Xs')
plt.title("Variance entre chaque deux algo pour %s"%body_part)
plt.legend()
fig = matplotlib.pyplot.gcf()
fig.savefig('../Données/Courbes/Variance_entre_chaque_algo_%s.jpg'%body_part)
plt.show()


#Comparaison with the terrain field (Xsens) Distances
Difference_Mobilenet=[]
Difference_Kinect=[]

Times_float=[]
Times=list(Variances.keys())
Times_float=[]
for time in Times:
    Times_float.append(float(time))
Times_float=sorted(Times_float)
MbPart=body_parts_Mobilenet[bPart]

for time in Times_float:
    diff_Mobilenet=np.sqrt((positions_Mobilenet[str(time)][MbPart][0]-positions_Xsens[str(time)][XbPart][1])**2+(positions_Mobilenet[str(time)][MbPart][1]-positions_Xsens[str(time)][XbPart][2])**2)
    diff_Kinect=np.sqrt((positions_Kinect[str(time)][bPart][1]-positions_Xsens[str(time)][XbPart][1])**2+(positions_Kinect[str(time)][bPart][2]-positions_Xsens[str(time)][XbPart][2])**2)
    
    Difference_Mobilenet.append(diff_Mobilenet)
    Difference_Kinect.append(diff_Kinect)
    
plt.plot(Times_float,Difference_Mobilenet,color='blue',label='dMobil--dXsens')
plt.plot(Times_float,Difference_Kinect,color='red',label='dKinect--dXsens')
plt.legend()
plt.title("Comparaison vérité terrain pour %s"%body_part)
fig = matplotlib.pyplot.gcf()
fig.savefig('../Données/Courbes/Comparaison vérité terrain pour %s.jpg'%body_part)
plt.show()


#Getting all the x and y values of the body part

bPart=body_parts_Mobilenet[body_part]
x_bPart_valuesX=[]
y_bPart_valuesX=[]

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
    xX=positions_Xsens[str(time)][body_parts_Xsens[bPart]][1]
    yX=positions_Xsens[str(time)][body_parts_Xsens[bPart]][2]
    x_bPart_valuesX.append(xX)
    y_bPart_valuesX.append(yX)
    
    xK=positions_Kinect[str(time)][bPart][1]
    yK=positions_Kinect[str(time)][bPart][2]
    x_bPart_valuesK.append(xK)
    y_bPart_valuesK.append(yK)
    
    xM=positions_Mobilenet[str(time)][body_parts_Mobilenet[bPart]][0]
    yM=positions_Mobilenet[str(time)][body_parts_Mobilenet[bPart]][1]
    x_bPart_valuesM.append(xM)
    y_bPart_valuesM.append(yM)
    
plt.plot(Times_float,y_bPart_valuesX,'green',label='Xsens')
plt.plot(Times_float,y_bPart_valuesM,'blue',label='Mobilenet')
plt.plot(Times_float,y_bPart_valuesK,'red',label='Kinect')
plt.legend()
plt.title("y values after interpolation %s"%body_part)
fig = matplotlib.pyplot.gcf()
axis="y"
fig.savefig('../Données/Courbes/%s_values_%s.jpg'%(axis,body_part))
plt.show()



plt.plot(Times_float,x_bPart_valuesX,'green',label='Xsens')
plt.plot(Times_float,x_bPart_valuesM,'blue',label='Mobilenet')
plt.plot(Times_float,x_bPart_valuesK,'red',label='Kinect')

plt.legend()
plt.title("x values after interpolation %s"%body_part)
fig = matplotlib.pyplot.gcf()
axis="x"
fig.savefig('../Données/Courbes/%s_values_%s.jpg'%(axis,body_part))
plt.show()