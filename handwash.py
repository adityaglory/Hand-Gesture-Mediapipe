import cv2 as cv
import pickle
import mediapipe as mp
import numpy as np 
import time

fps_time = 0
# accessing MediaPipe solutions
mp_hands = mp.solutions.hands

def image_processed(hand_img):
    # Image processing
    # 1. Convert BGR to RGB
    img_rgb = cv.cvtColor(hand_img, cv.COLOR_BGR2RGB)
    # 2. Flip the img in Y-axis
    img_flip = cv.flip(img_rgb, 1)
    # Initialize Hands
    hands = mp_hands.Hands(static_image_mode=False,
    max_num_hands=2, min_detection_confidence=0.2)
    # Results
    results = hands.process(img_flip)
    hands.close()

    try:
        data = results.multi_hand_landmarks[ :2] #Pembacaan landmark dari value 0-1
        data = ''.join(map(str, data)) #penggabungan 2 list [0] dan [1]
        #print(data)

        data = data.strip().split('\n')

        garbage = ['landmark {', '  visibility: 0.0', '  presence: 0.0', '}']

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
        return(np.zeros([1,63], dtype=int)[0])

# load model
with open('svmsafe.pkl', 'rb') as f:
    svm = pickle.load(f)

cap = cv.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit()
i = 1    
while True:
    ret, frame = cap.read()
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    # frame = cv.flip(frame,1)
    data = image_processed(frame)
    data = np.array(data)
    y_pred = svm.predict(data.reshape(-1,63))
    print("Classification is: ", y_pred)
    font = cv.FONT_HERSHEY_SIMPLEX
    org = (150, 100)
    fontScale = 2
    color = (255, 255, 255)
    thickness = 3
    # Using cv2.putText() method
    cv.putText(frame, "FPS: %f" % (1.0 / (time.time() - fps_time)), 
    (15, 25),cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2,)
    fps_time = time.time()
    frame = cv.putText(frame, str(y_pred), org, font, 
                    fontScale, color, thickness, cv.LINE_AA)
    cv.imshow('Hasil', frame)
    if cv.waitKey(1) == ord('q'): #Press Q for escape
        break

cap.release()
cv.destroyAllWindows()