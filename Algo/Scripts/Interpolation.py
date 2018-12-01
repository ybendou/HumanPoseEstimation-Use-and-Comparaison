import numpy as np
import json
import scipy.interpolate
import matplotlib.pyplot as plt
from collections import OrderedDict
from pprint import pprint
import argparse
import re
##############################################################################################################
##This script interpolates the data of the a camera Kinect, Mobilenet,Xsens... Given a fps and a Times end, ##
##it interpolates all the data to the new time table as long as the initial time is lower than the new time ##
##in that case, it gives an error unless an extrapolation is done.                                          ##
##It also takes in consideration wheter the z coordinates exist or not.      
                               ##
parser = argparse.ArgumentParser(description='Writing a Video from frames')
parser.add_argument('--file_name', type=str, default='../Données/Kinect/chris1/chris1_1.txt')
parser.add_argument('--camera', type=str, default='Kinect')

args = parser.parse_args()

camera=args.camera #Choosing which data to interpolate Kinect or Xsens
fps=30 #Frequency of measure for the interpolated data
#frames_count=len(list(positions.keys())) #Number of frames in the data file

save=True #In order to save the data in a txt format file


form='.txt'

file_name=args.file_name
#Importing the data
with open(file_name) as f:
    data = json.load(f, object_pairs_hook=OrderedDict)
    
Times=list(data['positions'].keys()) #Setting the times table
positions=data['positions'] #Setting the positions data


######################Functions##########################################
#This function returns 4 arrays, X Y Z positions for each body part and the time table, it deletes the times where 
#the body part has not been detected 
def body_positions(body_Part,Times,positions,camera='Kinect'):
    
    x_bPart_values={}
    y_bPart_values={}
    if camera!='Mobilenet': #The mobilenet doesn't have a z coordinate
        z_bPart_values={}
    
    tronq_times=[]
    for time in Times:
        bParts=list(positions[time].keys())
        if body_Part in bParts:
            x_bPart_values[time]=positions[time][body_Part][-2]
            y_bPart_values[time]=positions[time][body_Part][-1]
            if camera!='Mobilenet':
                z_bPart_values[time]=positions[time][body_Part][0]

            tronq_times.append(time)
    tronq_times=np.array(tronq_times)

    x_bPart_values_list=list(x_bPart_values.values())
    x_bPart_values_array=np.array(x_bPart_values_list)

    y_bPart_values_list=list(y_bPart_values.values())
    y_bPart_values_array=np.array(y_bPart_values_list)
    
    if camera!='Mobilenet':
        z=z_bPart_values_list=list(z_bPart_values.values())
        z_bPart_values_array=np.array(z_bPart_values_list)
        return(x_bPart_values_array,y_bPart_values_array,z_bPart_values_array,tronq_times)
    else:
        return(x_bPart_values_array,y_bPart_values_array,tronq_times)

#This function makes the interpolation of the tree arrays X,Y,Z by giving it the X,Y,Z arrays, the Times_float and the
#new times array on which the interpolation will be based
def interpolation(x_bPart_values_array,y_bPart_values_array,Times_float,new_times_array,z_bPart_values_array=np.array([]),camera='Kinect'):

    tau = Times_float[-1] - Times_float[0]
    new_xbPart_values = np.zeros(new_times_array.shape)
    new_ybPart_values = np.zeros(new_times_array.shape)
    
    if camera!='Mobilenet':
        new_zbPart_values = np.zeros(new_times_array.shape)
    
    y_gen = scipy.interpolate.interp1d(([t-Times_float[0] for t in Times_float]), y_bPart_values_array)
    y_gen(new_times_array)
    
    for i in range(len(new_times_array)):
        new_ybPart_values[i]=y_gen(new_times_array[i])
    
    x_gen = scipy.interpolate.interp1d(([t-Times_float[0] for t in Times_float]), x_bPart_values_array)
    x_gen(new_times_array)
    
    for i in range(len(new_times_array)):
        new_xbPart_values[i]=x_gen(new_times_array[i])
    
    if camera!='Mobilenet':
        z_gen = scipy.interpolate.interp1d(([t-Times_float[0] for t in Times_float]), z_bPart_values_array)
        z_gen(new_times_array)
        for i in range(len(new_times_array)):
            new_zbPart_values[i]=z_gen(new_times_array[i])
            
        return(new_xbPart_values,new_ybPart_values,new_zbPart_values,list(new_times_array))
    else:
        return(new_xbPart_values,new_ybPart_values,list(new_times_array))


#This function makes the whole process of interpoling the coordinates of a body part, it receives the body part name,
#the positions dictionnary, the real times array and the new times array on which the interpolation will be based on and
#returns a array of the new coordinates of the body part for all the new times
def new_body_positions(body_part,Times,positions,times_array,camera='Kinect'):
    
    if camera!='Mobilenet':
        x_bPart_values_array,y_bPart_values_array,z_bPart_values_array,tronq_times=body_positions(body_part,Times,positions)
    else:
        x_bPart_values_array,y_bPart_values_array,tronq_times=body_positions(body_part,Times,positions,camera)
        
    Times_float=[]
    for time in tronq_times:
        Times_float.append(float(time))

    if camera!='Mobilenet':
        new_xbPart_values,new_ybPart_values,new_zbPart_values,new_Times_float=interpolation(x_bPart_values_array,y_bPart_values_array,Times_float,new_times_array,z_bPart_values_array=z_bPart_values_array)
        new_bPart_Positions=np.stack((new_zbPart_values,new_xbPart_values,new_ybPart_values),axis=-1)

    else:
        new_xbPart_values,new_ybPart_values,new_Times_float=interpolation(x_bPart_values_array,y_bPart_values_array,Times_float,new_times_array,camera=camera)
        new_bPart_Positions=np.stack((new_xbPart_values,new_ybPart_values),axis=-1)
    
    return(new_bPart_Positions,new_Times_float)



################# main ################################
if __name__=='__main__':
    if camera!='Mobilenet':
        bParts=list(list(positions.values())[0].keys()) #Get the body Parts list
    else:
        bParts=['Head', 'mShoulder', 'rShoulder', 'rElbow', 'rWrist', 'lShoulder', 'lElbow', 'lWrist', 'rHip', 'rKnee', 'rAnkle', 'lHip', 'lKnee', 'lAnkle']

    T=27 #Duration Time

    new_positions={} 
    fps=30

    new_times_array = np.arange(0, T, 1/fps) #Times array with constant frequency fps

    for time in new_times_array:
        new_positions[str(time)]={} #Initialize the new position dictionary
    
    for bpart in bParts:
        if camera!='Mobilenet':
            new_body_pos=new_body_positions(bpart,Times,positions,new_times_array)[0]
        else:
            new_body_pos=new_body_positions(bpart,Times,positions,new_times_array,camera=camera)[0]
        for i in range(len(new_body_pos)):
            new_positions[str(new_times_array[i])][bpart]=list(new_body_pos[i])

   
    

    interpolated_data={} #Creating the new output data 
    interpolated_data['positions']=new_positions #Filling the new output data with the new interpolated positions
    
    #Saving the output
    if save:
        new_file_name=re.sub(".txt","_interpolated.txt",file_name)
        with open(new_file_name, 'w') as outfile:
             json.dump(interpolated_data, outfile, sort_keys = True, indent = 4,
                       ensure_ascii = False)