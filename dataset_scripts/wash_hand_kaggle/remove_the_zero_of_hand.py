# for this file ,get the skeleton of kaggle dataset and try test

# for this file ,get the skeleton of kaggle dataset and try test
import cv2
import mediapipe as mp
import numpy as np
import os
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands
import re




path_data='/media/liang/ssd2/wash_hand_3/Domain-and-View-point-Agnostic-Hand-Action-Recognition-main/datasets/'
for action in os.listdir(path_data+'HandWashDataset/'):
        v_i=0
        for video in os.listdir(path_data+'HandWashDataset/'+action):
            v_i=v_i+1
            # For webcam input:
            cap = cv2.VideoCapture(path_data+'HandWashDataset/'+action+'/'+video)
            video_name=video.split('.')[0]
            subject=video_name[-1]
            path_joint= path_data+'handwash/handwashkaggel/'+str(v_i)+'/'+action+'/'
            if not os.path.exists(path_joint):
                os.makedirs(path_joint)
            f=open(path_joint+'joint_processed.txt','w')
            
            

            with open(path_joint+'joint.txt') as f_r: 
                skels = f_r.read().splitlines()
                skels = np.array([ list(map(float, l.split())) for l in skels ])
                skels=skels[~(skels==0).all(1)]
                if skels.shape[0]<=1:
                    print(path_joint)

                        
                ## here we can begin to process it to others
            for i in range(skels.shape[0]):

                     
                list_skeleton=skels[i].tolist()
                list_skeleton=list(map(lambda x:str(x),list_skeleton))
                skeleton_array = ' '.join(list_skeleton)
                f.write(skeleton_array + '\n')  
            f.close()
            
            
            
            
            
            
            
       
            
       
        
            
            
            
            
            
            
            
            
            
            def load_data(data_format = 'common_minimal'):
                total_data = { sbj:{} for sbj in subjects }
                for sbj in subjects:
                    for a in actions:
                        with open(os.path.join(path_dataset, sbj, a, 'joint.txt')) as f: skels = f.read().splitlines()
                        skels = np.array([ list(map(float, l.split())) for l in skels ])
                        # remove the zero row ,no matter left or right ?have problem ,because recently only analysis one hand ,so process the skeleton file at first
                        skels = skels.reshape((skels.shape[0], 21, 3))
               