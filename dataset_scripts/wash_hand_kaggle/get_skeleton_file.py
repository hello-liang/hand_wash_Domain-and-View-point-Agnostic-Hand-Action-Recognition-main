# for this file ,get the skeleton of kaggle dataset and try test
import cv2
import mediapipe as mp
import numpy as np
import os
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands
import re

def process_output_skelenton_to_array(results):
    # not sure the type of mediapipe output ,I use this function convert it to array
    out=np.zeros(126)
    # Print handedness and draw hand landmarks on the image.
    if not results.multi_hand_landmarks:
        out=out
        #can not find a hand ,initialize to 0
    elif len(results.multi_handedness)==1:
        
        results_class=str(results.multi_handedness[0])
        results_class=re.split('label: "|"\n}\n',results_class)
        results_class=results_class[1]
        
        if results_class=="Left":
            hand_landmarks=str(results.multi_hand_landmarks[0])
            hand_landmarks=re.split('\n}\nlandmark {\n  x: |\n  y: |\n  z: |\n}\n|landmark {\n  x: ',hand_landmarks)
            out[0:63]=hand_landmarks[1:64] 
        else:
            hand_landmarks=str(results.multi_hand_landmarks[0])
            hand_landmarks=re.split('\n}\nlandmark {\n  x: |\n  y: |\n  z: |\n}\n|landmark {\n  x: ',hand_landmarks)
            out[63:126]=hand_landmarks[1:64] 
    
    elif len(results.multi_handedness)==2: #2 hand right first then left 

        
        hand_landmarks=results.multi_hand_landmarks[0]
        hand_landmarks=str(hand_landmarks)
        hand_landmarks=re.split('\n}\nlandmark {\n  x: |\n  y: |\n  z: |\n}\n|landmark {\n  x: ',hand_landmarks)
        out[0:63]=hand_landmarks[1:64]  
        hand_landmarks=results.multi_hand_landmarks[1]
        hand_landmarks=str(hand_landmarks)
        hand_landmarks=re.split('\n}\nlandmark {\n  x: |\n  y: |\n  z: |\n}\n|landmark {\n  x: ',hand_landmarks)
        out[63:126]=hand_landmarks[1:64]  
    else:
        print("have more than two hand？")
        print(len(results.multi_handedness))           
    return out

# For static images:
with mp_hands.Hands(model_complexity=0,min_detection_confidence=0.5,min_tracking_confidence=0.5) as hands:
    path_data='/media/liang/ssd2/wash_hand_3/Domain-and-View-point-Agnostic-Hand-Action-Recognition-main/datasets/'
    for action in os.listdir(path_data+'HandWashDataset/'):
        for video in os.listdir(path_data+'HandWashDataset/'+action):
            # For webcam input:
            cap = cv2.VideoCapture(path_data+'HandWashDataset/'+action+'/'+video)
            video_name=video.split('.')[0]
            subject=video_name.split('_')[4]
            if not os.path.exists(path_data+'handwash/handwashkaggel/'+subject+'/'+video_name+'/'):
                os.makedirs(path_data+'handwash/handwashkaggel/'+subject+'/'+video_name+'/')
            f=open(path_data+'handwash/handwashkaggel/'+subject+'/'+video_name+'/joint.txt','w')
            while cap.isOpened():
                success, image = cap.read()
                if not success:
                  print("Ignoring empty camera frame.")
                  # If loading a video, use 'break' instead of 'continue'.
                  break
            
                # To improve performance, optionally mark the image as not writeable to
                # pass by reference.
                image.flags.writeable = False
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                results = hands.process(image)
                skeleton_array=process_output_skelenton_to_array(results)
                list_skeleton=skeleton_array[63:126].tolist()
                list_skeleton=list(map(lambda x:str(x),list_skeleton))
                skeleton_array = ' '.join(list_skeleton)
                f.write(skeleton_array + '\n')  
            f.close()
            cap.release()
        
