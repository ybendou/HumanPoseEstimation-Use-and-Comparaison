import numpy as np
import cv2
import time
import argparse

###############################################################################
##This script takes several pictures and regroup them into a mp4 video format##

parser = argparse.ArgumentParser(description='Writing a Video from frames')
parser.add_argument('--directory', type=str, default='../Données/Yassir/data_04-36-03/rgbjpg')
parser.add_argument('--prefixe',type=str,default='')
parser.add_argument('--nb_frames',type=int,default=125)

args = parser.parse_args()

fps=10


if args.prefixe:
	image=cv2.imread("%s/%s1.png"%(args.directory,args.prefixe))
else:
	image=cv2.imread("%s/%s.jpg"%(args.directory,format(1,"04")))

print("%s/%s1"%(args.directory,args.prefixe))
print("image",image)
L,H=image.shape[1],image.shape[0]

fourcc = cv2.VideoWriter_fourcc('X','V','I','D')
out = cv2.VideoWriter('%s/rgb.mp4'%args.directory,fourcc, 10, (L,H))

total_frames=args.nb_frames #Number of frames
for num in range(1,total_frames+1):
	if args.prefixe=='':
		frame=cv2.imread("%s/%s.jpg"%(args.directory,format(num,"04")))
	else:
		frame=cv2.imread("%s/%s%s.png"%(args.directory,args.prefixe,num))
	out.write(frame)    #Writing 
#Re
out.release()				#Releasing everything at the end.
cv2.destroyAllWindows()
