import cv2
import csv
import time
import math as m
import mediapipe as mp
import os
import shutil

# Calculate angle.
def getAngle(a, b, c):
    ang = m.degrees(m.atan2(c[1]-b[1], c[0]-b[0]) - m.atan2(a[1]-b[1], a[0]-b[0]))
    return ang + 360 if ang < 0 else ang

def zoom_center(img, zoom_factor=1.5):

    y_size = img.shape[0]
    x_size = img.shape[1]
    
    # define new boundaries
    x1 = int(0.5*x_size*(1-1/zoom_factor))
    x2 = int(x_size-0.5*x_size*(1-1/zoom_factor))
    y1 = int(0.5*y_size*(1-1/zoom_factor))
    y2 = int(y_size-0.5*y_size*(1-1/zoom_factor))

    # first crop image then scale
    img_cropped = img[y1:y2,x1:x2]
    return cv2.resize(img_cropped, None, fx=zoom_factor, fy=zoom_factor)


"""
Function to send alert. Use this function to send alert when bad posture detected.
Feel free to get creative and customize as per your convenience.
"""

def process_images(folder_dir, output, problematic, getNose=True):
    for image_str in os.listdir(folder_dir):

        image = cv2.imread(folder_dir+image_str)

        h, w = image.shape[:2]

        # Convert the BGR image to RGB.
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Process the image.
        keypoints = pose.process(image)

        # Convert the image back to BGR.
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Use lm and lmPose as representative of the following methods.
        lm = keypoints.pose_landmarks
        lmPose = mp_pose.PoseLandmark
        
        if lm:
            if getNose:
                nose = int(lm.landmark[lmPose.NOSE].x * w)
            else:
                nose = image_str.split('-')[1]
            # Left shoulder.
            l_shldr_x = int(lm.landmark[lmPose.LEFT_SHOULDER].x * w)
            l_shldr_y = int(lm.landmark[lmPose.LEFT_SHOULDER].y * h)
            # Right shoulder
            r_shldr_x = int(lm.landmark[lmPose.RIGHT_SHOULDER].x * w)
            r_shldr_y = int(lm.landmark[lmPose.RIGHT_SHOULDER].y * h)
            # Left elbow.
            l_elbow_x = int(lm.landmark[lmPose.LEFT_ELBOW].x * w)
            l_elbow_y = int(lm.landmark[lmPose.LEFT_ELBOW].y * h)
            # Right elbow.
            r_elbow_x = int(lm.landmark[lmPose.RIGHT_ELBOW].x * w)
            r_elbow_y = int(lm.landmark[lmPose.RIGHT_ELBOW].y * h)
            # Left hip.
            l_hip_x = int(lm.landmark[lmPose.LEFT_HIP].x * w)
            l_hip_y = int(lm.landmark[lmPose.LEFT_HIP].y * h)
            # Right hip.
            r_hip_x = int(lm.landmark[lmPose.RIGHT_HIP].x * w)
            r_hip_y = int(lm.landmark[lmPose.RIGHT_HIP].y * h)
            # Left wrist.
            l_wrist_x = int(lm.landmark[lmPose.LEFT_WRIST].x * w)
            l_wrist_y = int(lm.landmark[lmPose.LEFT_WRIST].y * h)
            # Right wrist.
            r_wrist_x = int(lm.landmark[lmPose.RIGHT_WRIST].x * w)
            r_wrist_y = int(lm.landmark[lmPose.RIGHT_WRIST].y * h)
            # Left knee.
            l_knee_x = int(lm.landmark[lmPose.LEFT_KNEE].x * w)
            l_knee_y = int(lm.landmark[lmPose.LEFT_KNEE].y * h)
            # Right knee.
            r_knee_x = int(lm.landmark[lmPose.RIGHT_KNEE].x * w)
            r_knee_y = int(lm.landmark[lmPose.RIGHT_KNEE].y * h)
            # Left ankle.
            l_ankle_x = int(lm.landmark[lmPose.LEFT_ANKLE].x * w)
            l_ankle_y = int(lm.landmark[lmPose.LEFT_ANKLE].y * h)
            # Right ankle.
            r_ankle_x = int(lm.landmark[lmPose.RIGHT_ANKLE].x * w)
            r_ankle_y = int(lm.landmark[lmPose.RIGHT_ANKLE].y * h)

            # Calculate shoulder angles.
            l_shldr = getAngle((l_elbow_x, l_elbow_y), (l_shldr_x, l_shldr_y), (l_hip_x, l_hip_y))
            r_shldr = getAngle((r_elbow_x, r_elbow_y), (r_shldr_x, r_shldr_y), (r_hip_x, r_hip_y))
            # Calculate elbow angles.
            l_elbow = getAngle((l_shldr_x, l_shldr_y), (l_elbow_x, l_elbow_y), (l_wrist_x, l_wrist_y))
            r_elbow = getAngle((r_shldr_x, r_shldr_y), (r_elbow_x, r_elbow_y), (r_wrist_x, r_wrist_y))
            # Calculate hip angles.
            l_hip = getAngle((l_knee_x, l_knee_y), (l_hip_x, l_hip_y), (l_shldr_x, l_shldr_y))
            r_hip = getAngle((r_knee_x, r_knee_y), (r_hip_x, r_hip_y), (r_shldr_x, r_shldr_y))
            # Calculate knee angles.
            l_knee = getAngle((l_hip_x, l_hip_y), (l_knee_x, l_knee_y), (l_ankle_x, l_ankle_y))
            r_knee = getAngle((r_hip_x, r_hip_y), (r_knee_x, r_knee_y), (r_ankle_x, r_ankle_y))

            cv2.circle(image, (l_shldr_x, l_shldr_y), 5, yellow, -1)
            cv2.circle(image, (r_shldr_x, r_shldr_y), 5, yellow, -1)
            cv2.circle(image, (l_elbow_x, l_elbow_y), 5, yellow, -1)   
            cv2.circle(image, (r_elbow_x, r_elbow_y), 5, yellow, -1)          
            cv2.circle(image, (l_hip_x, l_hip_y), 5, yellow, -1)
            cv2.circle(image, (r_hip_x, r_hip_y), 5, yellow, -1)
            cv2.circle(image, (l_wrist_x, l_wrist_y), 5, yellow, -1)
            cv2.circle(image, (r_wrist_x, r_wrist_y), 5, yellow, -1)
            cv2.circle(image, (l_knee_x, l_knee_y), 5, yellow, -1)
            cv2.circle(image, (r_knee_x, r_knee_y), 5, yellow, -1)
            cv2.circle(image, (l_ankle_x, l_ankle_y), 5, yellow, -1)
            cv2.circle(image, (r_ankle_x, r_ankle_y), 5, yellow, -1)

            cv2.line(image, (l_shldr_x, l_shldr_y), (l_elbow_x, l_elbow_y), green, 4)
            cv2.line(image, (l_wrist_x, l_wrist_y), (l_elbow_x, l_elbow_y), green, 4)
            cv2.line(image, (l_hip_x, l_hip_y), (l_shldr_x, l_shldr_y), green, 4)
            cv2.line(image, (l_hip_x, l_hip_y), (l_knee_x, l_knee_y), green, 4)
            cv2.line(image, (l_ankle_x, l_ankle_y), (l_knee_x, l_knee_y), green, 4)

            cv2.line(image, (r_shldr_x, r_shldr_y), (r_elbow_x, r_elbow_y), green, 4)
            cv2.line(image, (r_wrist_x, r_wrist_y), (r_elbow_x, r_elbow_y), green, 4)
            cv2.line(image, (r_hip_x, r_hip_y), (r_shldr_x, r_shldr_y), green, 4)
            cv2.line(image, (r_hip_x, r_hip_y), (r_knee_x, r_knee_y), green, 4)
            cv2.line(image, (r_ankle_x, r_ankle_y), (r_knee_x, r_knee_y), green, 4)

            cv2.line(image, (r_hip_x, r_hip_y), (l_hip_x, l_hip_y), green, 4)
            cv2.line(image, (r_shldr_x, r_shldr_y), (l_shldr_x, l_shldr_y), green, 4)

            cv2.imwrite(f'{output}{image_str}.jpeg', image)

            with open("dataset.csv", 'a', encoding = 'utf-8', newline='') as f:
                writer = csv.writer(f, delimiter=',')
                writer.writerow([image_str, l_shldr, r_shldr, l_elbow, r_elbow, l_hip, r_hip, l_knee, r_knee, nose, image_str.split('-')[0]])
        else:
            cv2.imwrite(f'{problematic}{image_str}', image)



def sendWarning(x):
    pass


# =============================CONSTANTS and INITIALIZATIONS=====================================#
# Initilize frame counters.
good_frames = 0
bad_frames = 0

# Font type.
font = cv2.FONT_HERSHEY_SIMPLEX

# Colors.
blue = (255, 127, 0)
red = (50, 50, 255)
green = (127, 255, 0)
dark_blue = (127, 20, 0)
light_green = (127, 233, 100)
yellow = (0, 255, 255)
pink = (255, 0, 255)

# Initialize mediapipe pose class.
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
# ===============================================================================================#


if __name__ == "__main__":

    problematic = 'problematic/'
    problematic2 = 'problematic2/'
    problematic3 = 'problematic3/'
    output = 'output/'

    if os.path.exists(output):
        shutil.rmtree(output)

    if os.path.exists(problematic):
        shutil.rmtree(problematic)

    if os.path.exists(problematic2):
        shutil.rmtree(problematic2)

    if os.path.exists(problematic3):
        shutil.rmtree(problematic3)

    os.mkdir(output)

    os.mkdir(problematic)

    os.mkdir(problematic2)

    os.mkdir(problematic3)

    folder_dir = 'images/'
    folder_dir2 = 'recortes/'
    folder_dir3 = 'recortes2/'

    with open("dataset.csv", 'w', encoding = 'utf-8', newline='') as f:
        writer = csv.writer(f, delimiter=',')
        writer.writerow(["image", "l_shldr", "r_shldr", "l_elbow", "r_elbow", "l_hip", "r_hip", "l_knee", "r_knee", "nose","class"])

    process_images(folder_dir, output, problematic)
    process_images(folder_dir2, output, problematic2)
    process_images(folder_dir3, output, problematic3, False)
