import cv2
import mediapipe as mp
import pandas as pd  
import os
import numpy as np

def image_processed(file_path):
    
    # reading the static image
    hand_img = cv2.imread(file_path)

    # Image processing
    # 1. Convert BGR to RGB
    img_rgb = cv2.cvtColor(hand_img, cv2.COLOR_BGR2RGB)

    # 2. Flip the img in Y-axis
    img_flip = cv2.flip(img_rgb, 1)

    # accessing MediaPipe solutions
    mp_hands = mp.solutions.hands

    # Initialize Hands
    hands = mp_hands.Hands(
        static_image_mode=True,
        model_complexity=1, 
        max_num_hands=2, 
        min_detection_confidence=0.3)

    # Results
    results = hands.process(img_flip)
    hands.close()

    try:
        data = results.multi_hand_landmarks[ :1] #Pembacaan landmark dari value 0-1
        data = ''.join(map(str, data)) #penggabungan 2 list [0] dan [1]
        print(data)
        data = data.strip().split('\n')
        garbage = ['landmark {', '  visibility: 0.0', '  presence: 0.0', '}'] 
        #Menghapus string yg tidak diinginkan
        without_garbage = []

        for i in data:
            if i not in garbage:
                without_garbage.append(i)

        clean = []

        for i in without_garbage:
            i = i.strip()
            clean.append(i[2:])

        for i in range(0, len(clean)):
                    clean[i] = float(clean[i])

        return(clean)
    except:
        return(np.zeros([1,63], dtype=int)[0]) #Assign nilai 0 pada landmark tidak terdeteksi

def make_csv():
    
    mypath = 'hand_detect/dataset'
    file_name = open('zero.csv', 'a')

    for each_folder in os.listdir(mypath):
        if '._' in each_folder:
            pass

        else:
            for each_number in os.listdir(mypath + '/' + each_folder):
                if '._' in each_number:
                    pass
                
                else:
                    label = each_folder

                    file_loc = mypath + '/' + each_folder + '/' + each_number

                    data = image_processed(file_loc)
                    try:
                        for id, i in enumerate(data):
                            if id == 0: #if id<=1
                                print(i)
                            
                            file_name.write(str(i))
                            file_name.write(',')

                        file_name.write(label)
                        file_name.write('\n')
                    
                    except:
                        file_name.write('0')
                        file_name.write(',')

                        file_name.write('None')
                        file_name.write('\n')
       
    file_name.close()
    print('DONE, Data CSV Created !!!')

if __name__ == "__main__":
    make_csv()

#Modified by Glory Aditya
#Source : https://github.com/dongdv95/hand-gesture-recognition
#Source : https://youtu.be/dlyVy_LNCEQ