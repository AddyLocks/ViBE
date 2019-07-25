# ====================================================================================================
#                       ViBE: Visual Background Extraction Algorithm
# ====================================================================================================
#
# Original Paper: ViBe: A Universal Background Subtraction Algorithm for Video Sequences
# Authors: Olivier Barnich, Marc Van Droogenbroeck
# Paper can be found at: http://orbi.ulg.ac.be/bitstream/2268/145853/1/Barnich2011ViBe.pdf
# Codes from the authors can be found at: http://www.telecom.ulg.ac.be/research/vibe/
# ====================================================================================================

import numpy as np
import cv2
import random
import time

video  = cv2.VideoCapture('Test_Video.avi')                     # Path to video for processing
height = int(video.get(4))                                      # Resolution of the video
width  = int(video.get(3))

# ====================================================================================================
# Hyper-Parameters
# ====================================================================================================

N = 20                                                # Sample number per pixel
R = np.ones((height, width, N), dtype=np.uint8) * 20  # Radius of the Sphere
Number_min = 2                                        # No of close samples to be a part of the background
Rand_Samples = 20                                     # Amount of random sampling

frame_number = 50                                     # Frame from which processing starts
FramesToSkip = 1                                      # Frames to skip in between ( 1 is normal )
Pad = 10                                              # Padding around the bounding box

# ====================================================================================================
# Pre-defining random vectors (to make computation faster)
# ====================================================================================================
Random_Vector_N = [random.randint(0, N - 1) for _ in range(100000)]   # Vector containing random values from 0 to N
Random_Index_N = 0                                                    # Index of above vector

Random_Vector_N2 = [random.randint(0, N - 1) for _ in range(100000)]  # Same as above
Random_Index_N2 = 0

Random_Vector_Phai = [random.randint(0, Rand_Samples - 1) for _ in range(100000)]   # Same, from 0 to Rand_Samples
Random_Index_Phai = 0                                                               # Index of Above Vector

Random_Vector_Phai2 = [random.randint(0, Rand_Samples - 1) for _ in range(100000)]  # Same as above
Random_Index_Phai2 = 0

I_vector = np.zeros(100000, dtype=np.int8)  # Vectors for finding immediate neighbours of a pixel
J_vector = np.zeros(100000, dtype=np.int8)

index_neighbour = 0

for index_neighbour in range(0, 100000):
    i, j = 0, 0
    while i == 0 and j == 0:
        i = random.randint(-1, 1)
        j = random.randint(-1, 1)
    I_vector[index_neighbour] = i
    J_vector[index_neighbour] = j
    index_neighbour = index_neighbour + 1

# ====================================================================================================
# Initializations of Important Matrices
# ====================================================================================================

print("Height of image: ", height)
print("Width of image: ", width)

Segmentation = np.zeros((height, width), dtype=np.uint8)         # Will Store Final Image Segmentation Model
Background_Model = np.zeros((height, width, N), dtype=np.uint8)  # Stores Background Model
frame_3D = np.zeros((height, width, N), dtype=np.uint8)          # A Matrix used later
compare_matrix = np.zeros((height, width, N), dtype=np.uint8)    # A Matrix used later

# ====================================================================================================
# Function for Initialization of Background Model with Noisy first frame
# ====================================================================================================


def number_plus_noise(number):
    number = number + random.randint(-10, 10)
    if number > 255:
        number = 255
    if number < 0:
        number = 0
    return np.uint8(number)

# ====================================================================================================
#  initialise Background_Model with first frame + Noise
# ====================================================================================================


ret, coloured_frame = video.read()
frame = cv2.cvtColor(coloured_frame, cv2.COLOR_BGR2GRAY)

for x in range(0, height):
    for y in range(0, width):
        for n in range(0, N):
            Background_Model[x, y, n] = number_plus_noise(frame[x, y])
# ====================================================================================================
#  The main code
# ====================================================================================================

while ret:
    start = time.time()                                       # To calculate time for one iteration
    video.set(1, frame_number)                                # Set frame to acquire to frame_number
    ret, coloured_frame = video.read()                        # Acquire Frame
    frame_number = frame_number + FramesToSkip                # Update frame_number for next iteration
    frame = cv2.cvtColor(coloured_frame, cv2.COLOR_BGR2GRAY)  # Convert coloured frame to gray scale

    for n in range(0, N):                                             # frame converted to 3 dimension for line 100
        frame_3D[:, :, n] = frame[:, :]

    compare_matrix = np.less(np.abs(Background_Model - frame_3D), R)  # Checks if frame pixels are within threshold R

    for x in range(0, height):                      # If yes, true is stored, else false is stored in compare matrix
        for y in range(0, width):
            data = 0
            for n in range(0, N):
                if compare_matrix[x, y, n]:
                    data = data + 1
                    if data >= Number_min:          # If any >= Number_min number of trues are there in a pixel
                        Segmentation[x, y] = 0      # It's a background pixel
                        break
                if n == N - 1:
                    Segmentation[x, y] = 255        # Else its a foreground pixel

    for x in range(0, height):
        for y in range(0, width):
            if Segmentation[x, y] == 0:                         # If a pixel is a background pixel
                rand = Random_Vector_Phai[Random_Index_Phai]    # Choose a random number between 0 and Rand_Samples
                Random_Index_Phai = Random_Index_Phai + 1
                if Random_Index_Phai == 100000:
                    Random_Index_Phai = 0

                if rand == 0:                                   # If it is zero (1/Rand_Samples probability)
                    rand = Random_Vector_N[Random_Index_N]      # Choose a random number between 0 and N
                    Random_Index_N = Random_Index_N + 1
                    if Random_Index_N == 100000:
                        Random_Index_N = 0
                    Background_Model[x, y, rand] = frame[x, y]  # Update Background Model with this pixel

                rand = Random_Vector_Phai2[Random_Index_Phai2]  # Again, random number between 0 and Rand_Samples
                Random_Index_Phai2 = Random_Index_Phai2 + 1
                if Random_Index_Phai2 == 100000:
                    Random_Index_Phai2 = 0

                if rand == 0:                                   # If it is zero (1/Rand_Samples probability)
                    rand = Random_Vector_N2[Random_Index_N2]    # Choose a random number between 0 and N
                    Random_Index_N2 = Random_Index_N2 + 1
                    if Random_Index_N2 == 100000:
                        Random_Index_N2 = 0
                    try:                                        # Update a neighbour's Background Model
                        Background_Model[x+I_vector[index_neighbour], y+J_vector[index_neighbour], rand] = frame[x, y]
                        index_neighbour = index_neighbour + 1   # Neighbour chosen randomly
                        if index_neighbour == 100000:
                            index_neighbour = 0
                    except:
                        pass
    Segmentation = cv2.medianBlur(Segmentation, 7)              # Apply a Median Filter

# ====================================================================================================
#  Bounding Box
# ====================================================================================================

    contours, hierarchy = cv2.findContours(Segmentation, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    for c in contours:
        i, j, h, w = cv2.boundingRect(c)
        if h > 10 and w > 10:                   # Put bounding box only if height and width of box is greater than 10
            cv2.rectangle(coloured_frame, (i-Pad, j-Pad), (i+h+Pad, j+w+Pad), (0, 255, 0), 2)
        else:
            for x in range(i, i+h):
                for y in range(j, j+w):
                    try:
                        Segmentation[x, y] = 0  # Else, convert all those pixels in 10x10 to background pixels
                    except:
                        pass

# ====================================================================================================
#  Printing and Displaying
# ====================================================================================================

    cv2.imshow('Actual Frame!', coloured_frame)                            # Print actual frame with the bounding boxes
    cv2.imshow('Foreground is white, Background is Black!', Segmentation)  # Print Image Segmentation obtained

    print("Frame number: ", frame_number)                                  # Print frame number
    end = time.time()                                                      # Print time it took for this iteration
    print("Time for processing this frame: ", (end - start))
    if cv2.waitKey(20) & 0xFF == ord('q'):                                 # Break while loop if video ends
        break

# ====================================================================================================
# End
# ====================================================================================================

