import argparse
import logging
import time

import cv2 as cv
import numpy as np
import sys
import os
import json

start=time.time()
logger = logging.getLogger('TfPoseEstimator-Video')
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

if __name__ == '__main__':

    #All the args argument:
    parser = argparse.ArgumentParser(description='tf-pose-estimation Video')
    parser.add_argument('--video', type=str, default='')
    parser.add_argument('--resolution', type=str, default='432x368', help='network input resolution. default=432x368')
    parser.add_argument('--model', type=str, default='mobilenet_thin', help='cmu / mobilenet_thin/ caffe')
    parser.add_argument('--show-process', type=bool, default=False,
                        help='for debug purpose, if enabled, speed for inference is dropped.')
    parser.add_argument('--resize-out-ratio', type=float, default=4.0,
                        help='if provided, resize heatmaps before they are post-processed. default=1.0')
    parser.add_argument('--showBG', type=bool, default=True, help='False to show skeleton only.')
    parser.add_argument('--save', type=str, default='', help='Give the path of the folder where to save the frames, default not in saving mode')
    parser.add_argument('--nb_images', type=int, default=140, help='Number of frames in which to devide the video, default=140')
    parser.add_argument('--live', type=bool, default=False, help='True to show all the frames of the video')
    parser.add_argument('--scale', type=float, default=0.5, help='Scale of the image to compute for caffe mode')
    parser.add_argument('--screen', type=bool, default=True, help='Show the frames')
    parser.add_argument('--save_video', type=str, default='', help='Save in a video format')
    parser.add_argument('--save_data', type=str, default='', help='Save data of the pos in a dictionnary txt format')
    args = parser.parse_args()

    fps_time = 0
    frame_number=1
    count=1

    
    #Look for the video
    cap = cv.VideoCapture(args.video)
    screen=args.screen
    width, height = map(int, args.resolution.split('x'))
    Duration_Kinect=132
    total_frames=132
    fps=total_frames/Duration_Kinect
    print('screen : ',screen)

    #Saving the processed frames in a directory:
    if args.save!='':
        print('Saving the frames at %s'%args.save)
    
    #Saving the processed video in a video format in a directory:
    if args.save_video!='':
        print('Saving the video at %s ...'%args.save_video)
        fourcc = cv.VideoWriter_fourcc('X','V','I','D')
        video_output = cv.VideoWriter('%s/output.mp4'%args.save_video,fourcc, 20.0, (int(cap.get(3)),int(cap.get(4))))
        
        print("Saving the video done")
    
    nb_images=args.nb_images
    
    #Number of the frames in the video:
    length = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
    
    #Check for the right model, change to its directory and import the required files:
    if args.model=='mobilenet_thin' or args.model=='cmu' or args.model=='mobilenet_fast' or args.model=='mobilenet_accurate':
        sys.path.insert(0, '../tf-openpose')
        from tf_pose.estimator import TfPoseEstimator
        from tf_pose.networks import get_graph_path, model_wh
        logger.debug('initialization %s : %s' % (args.model, get_graph_path(args.model)))
        w, h = model_wh(args.resolution)
        #e = TfPoseEstimator(get_graph_path(args.model), target_size=(w, h))
        if w == 0 or h == 0:
            e = TfPoseEstimator(get_graph_path(args.model), target_size=(432, 368))
        else:
            e = TfPoseEstimator(get_graph_path(args.model), target_size=(w, h))
    elif args.model=='caffe':
        start_load=time.time()
        sys.path.append('/home/y17bendo/stage2018/Realtime_Multi-Person_Pose_Estimation/testing/python')
        import pylab as plt
        from PIL import Image
        import processing_image
        from processing_image import import_param, processing
    
        #Importing the parameters of the net, and loading the net:
        param,model,net=import_param()
        print("loading the net took : %f"%(time.time()-start_load))
        parametre_search=args.scale
    if args.save_data:
        data={'positions':{}}
        
    #Changing back to default directory:
    sys.path.append(r'/home/y17bendo/stage2018/Algo')
    os.chdir('/home/y17bendo/stage2018/Algo')

    #Live mode, all the frames are processed, be careful the frames can be very low:
    if args.live:
        ratio=1    
    else:
        ratio=int(length/nb_images)

    if cap.isOpened() is False:
        print("Error opening video stream or file")

    try:
        while cap.isOpened():    
            #Capturing the frame:       
            ret_val, image = cap.read()
            #Frame to process:
            if count%ratio==0:
                #Choosing the algorithm to process with (caffe implementation, tensorflow mobilenet_thin of tensorflow cmu model):
                if args.model=='mobilenet_thin' or args.model=='cmu' or args.model=='mobilenet_fast' or args.model=='mobilenet_accurate':
                    #Returning a list of the human parts:
                    humans = e.inference(image, resize_to_default=(w > 0 and h > 0), upsample_size=args.resize_out_ratio)
                    if not args.showBG:
                        image = np.zeros(image.shape)
                    #Drawing the human parts on top of the image:
                    image = TfPoseEstimator.draw_humans(image, humans, imgcopy=False)
                
                elif args.model=='caffe':
                    #Returning the image processed in an array format:
                    try:
                                                                  
                        image_before_rgb=processing(image,param,parametre_search,model,net)           
                        b1,g1,r1 = cv.split(image_before_rgb)
                        image = cv.merge((r1,g1,b1))
                    except: 
                        frame_number+=1
                #Showing the frames processed:
                if screen==True:
                    cv.putText(image, "FPS: %f" % (1.0 / (time.time() - fps_time)), (10, 10),  cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    #cv.imshow('computation result', image)

                #Saving the frames in a directory:                                                               
                if args.save!='':
                    cv.imwrite("%s/%d.jpg" % (args.save,frame_number), image)

                #Saving the video output in a directory:
                if args.save_video!='':
                    video_output.write(image)

                print("fps : ",(1.0 / (time.time() - fps_time)))
                print("frame number :",frame_number)        
                
                #Saving the data output in a dictionary txt format:
                if args.save_data:
                    human=humans[0]

                    body_position={}
                    body_parts={0:"Head",1:"mShoulder",2:"rShoulder",3:"rElbow",4:"rWrist",5:"lShoulder",6:"lElbow",7:"lWrist",8:"rHip",9:"rKnee",10:"rAnkle",11:"lHip",12:"lKnee",13:"lAnkle"}
                    for bPart in body_parts.keys():
                        if bPart in human.body_parts:
                            x=human.body_parts[bPart].x
                            y=human.body_parts[bPart].y
                            pos=[x,y]
                            body_position[body_parts[bPart]]=pos
                    duration=frame_number/fps        
                    data['positions'][duration]=body_position
                frame_number +=1
            fps_time = time.time()

            #End of the video:
            if count==length:
                break
            count+=1

            #Interrupt of the process by pressing on 'q', can be done if the screen is on:
            if cv.waitKey(1) & 0xFF == ord('q'):
                break

        #Releasing everything:
        if args.save_video!='':
            video_output.release()
        cap.release()
        cv.destroyAllWindows()
        logger.debug('finished+')
        print("end time : ",time.time()-start)
        if args.save_data:
            with open("%s/rgb.txt"%args.save_data, 'w') as outfile:
                json.dump(data, outfile, sort_keys = True, indent = 4,ensure_ascii = False)
    except:
        if args.save_video!='':
            video_output.release()
        cap.release()
        cv.destroyAllWindows()
        print("The program was suddenly interrupted, end time : ",time.time()-start)
        if args.save_data:
            with open("%s/rgbMobilenet.txt"%args.save_data, 'w') as outfile:
                json.dump(data, outfile, sort_keys = True, indent = 4,ensure_ascii = False)

