# Imports
import cv2
import numpy as np
import mediapipe as mp
from collections import deque
import os
import time


# Deques to handle colour points of different colour
bluep = [deque(maxlen=1024)]
greenp = [deque(maxlen=1024)]
redp = [deque(maxlen=1024)]
yellowp = [deque(maxlen=1024)]


# These indexes used to mark the points in particular Deque of specific colour
blue_index = 0
green_index = 0
red_index = 0
yellow_index = 0


# Used for dilation purpose 
kernel = np.ones((5,5),np.uint8)

colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 255, 255)]
colorIndex = 0

# Here is code for Palette setup
paintWindow = np.zeros((471, 636, 3)) + 255

# Displays the project name (comment this if you are uncommenting the below code)
paintWindow = cv2.rectangle(paintWindow, (160, 1), (485, 65), (0, 0, 0), 2)
cv2.putText(paintWindow, "AirPelette", (200, 45), cv2.FONT_HERSHEY_TRIPLEX, 1.5, (0, 0, 0), 2, cv2.LINE_AA)

# Uncomment for displaying colour on paint window
# paintWindow = cv2.rectangle(paintWindow, (40, 1), (140, 65), (0, 0, 0), 2)
# paintWindow = cv2.rectangle(paintWindow, (160, 1), (255, 65), (255, 0, 0), 2)
# paintWindow = cv2.rectangle(paintWindow, (275, 1), (370, 65), (0, 255, 0), 2)
# paintWindow = cv2.rectangle(paintWindow, (390, 1), (485, 65), (0, 0, 255), 2)
# paintWindow = cv2.rectangle(paintWindow, (505, 1), (600, 65), (0, 255, 255), 2)
# paintWindow = cv2.rectangle(paintWindow, (531, 204), (636, 267), (128, 128, 128), 2)  # New "Save & Quit" button
# cv2.putText(paintWindow, "CLEAR", (49, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
# cv2.putText(paintWindow, "BLUE", (185, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
# cv2.putText(paintWindow, "GREEN", (298, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
# cv2.putText(paintWindow, "RED", (420, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
# cv2.putText(paintWindow, "YELLOW", (520, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
# cv2.putText(paintWindow, "Save & Quit", (540, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)  # New "Save & Quit" text

cv2.namedWindow('Paint', cv2.WINDOW_AUTOSIZE)


# initialize mediapipe
mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mpDraw = mp.solutions.drawing_utils


# Initialize the webcam
cap = cv2.VideoCapture(0)
ret = True

# Get the desktop path
desktop_path = os.path.join(os.path.join(os.environ['USERPROFILE']), 'Desktop\\AirPalette_storage')

while ret:
    # Read each frame from the webcam
    ret, frame = cap.read()

    x, y, c = frame.shape

    # Mirroring your input(remove or comment this line if you dont want to mirror your input)
    frame = cv2.flip(frame, 1)
    #hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    frame = cv2.rectangle(frame, (40,1), (140,65), (0,0,0), 2)
    frame = cv2.rectangle(frame, (160,1), (255,65), (255,0,0), 2)
    frame = cv2.rectangle(frame, (275,1), (370,65), (0,255,0), 2)
    frame = cv2.rectangle(frame, (390,1), (485,65), (0,0,255), 2)
    frame = cv2.rectangle(frame, (505,1), (600,65), (0,255,255), 2)
    frame = cv2.rectangle(frame, (531, 204), (636, 267), (128, 128, 128), 2)  # New "Save & Quit" button

    cv2.putText(frame, "CLEAR", (49, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, "BLUE", (185, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, "GREEN", (298, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, "RED", (420, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, "YELLOW", (520, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, "Save & Quit", (536, 236), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)  # New "Save & Quit" text

    # Get hand landmark prediction
    result = hands.process(framergb)

    # post process the result
    if result.multi_hand_landmarks:
        landmarks = []
        for handslms in result.multi_hand_landmarks:
            for lm in handslms.landmark:
                lmx = int(lm.x * 640)
                lmy = int(lm.y * 480)

                landmarks.append([lmx, lmy])


            # Drawing landmarks on frames
            mpDraw.draw_landmarks(frame, handslms, mpHands.HAND_CONNECTIONS)
        fore_finger = (landmarks[8][0],landmarks[8][1])
        center = fore_finger
        thumb = (landmarks[4][0],landmarks[4][1])
        cv2.circle(frame, center, 3, (0,255,0),-1)
        print(center[1]-thumb[1])
        if (thumb[1]-center[1]<30):
            bluep.append(deque(maxlen=512))
            blue_index += 1
            greenp.append(deque(maxlen=512))
            green_index += 1
            redp.append(deque(maxlen=512))
            red_index += 1
            yellowp.append(deque(maxlen=512))
            yellow_index += 1
            
        elif center[1] <= 65:
            if 40 <= center[0] <= 140 and 1 <= center[1] <= 65: # Clear Button
                bluep = [deque(maxlen=512)]
                greenp = [deque(maxlen=512)]
                redp = [deque(maxlen=512)]
                yellowp = [deque(maxlen=512)]

                blue_index = 0
                green_index = 0
                red_index = 0
                yellow_index = 0

                paintWindow[67:,:,:] = 255
            elif 160 <= center[0] <= 255 and 1 <= center[1] <= 65:
                    colorIndex = 0 # Blue
            elif 275 <= center[0] <= 370 and 1 <= center[1] <= 65:
                    colorIndex = 1 # Green
            elif 390 <= center[0] <= 485 and 1 <= center[1] <= 65:
                    colorIndex = 2 # Red
            elif 505 <= center[0] <= 600 and 1 <= center[1] <= 65:
                    colorIndex = 3 # Yellow
                    
        # Saves the image with different name every time (i have used the time function) and breaks the loop and closes the program            
        elif 535 <= center[0] <= 636 and 210 <= center[1] <= 267: # Save and Quit area check
                timestamp = str(int(time.time()))
                output_path = os.path.join(desktop_path, f'AirPalette_output_{timestamp}.png')
                cv2.imwrite(output_path, paintWindow)
                break 
            
        else :
            if colorIndex == 0:
                bluep[blue_index].appendleft(center)
            elif colorIndex == 1:
                greenp[green_index].appendleft(center)
            elif colorIndex == 2:
                redp[red_index].appendleft(center)
            elif colorIndex == 3:
                yellowp[yellow_index].appendleft(center)
    # Append the next deques when nothing is detected to avoid messing up
    else:
        bluep.append(deque(maxlen=512))
        blue_index += 1
        greenp.append(deque(maxlen=512))
        green_index += 1
        redp.append(deque(maxlen=512))
        red_index += 1
        yellowp.append(deque(maxlen=512))
        yellow_index += 1

    # Draw lines of all the colors on the canvas and frame
    points = [bluep, greenp, redp, yellowp]

    # this loop wont paint the camera input area.
    # for j in range(len(points[0])):
    #         for k in range(1, len(points[0][j])):
    #             if points[0][j][k - 1] is None or points[0][j][k] is None:
    #                 continue
    #             cv2.line(paintWindow, points[0][j][k - 1], points[0][j][k], colors[0], 2)

    # this loop will paint the both the areas.
    for i in range(len(points)):
        for j in range(len(points[i])):
            for k in range(1, len(points[i][j])):
                if points[i][j][k - 1] is None or points[i][j][k] is None:
                    continue
                cv2.line(frame, points[i][j][k - 1], points[i][j][k], colors[i], 2)
                cv2.line(paintWindow, points[i][j][k - 1], points[i][j][k], colors[i], 2)

    cv2.imshow("Output", frame) 
    cv2.imshow("Paint", paintWindow)

    if cv2.waitKey(1) == ord('q'):
        break

# release the webcam and destroy all active windows
cap.release()
cv2.destroyAllWindows()