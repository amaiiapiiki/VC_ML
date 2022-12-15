import cv2

cap = cv2.VideoCapture('videos\dataset_limpio_amaia.mp4')

if (cap.isOpened()== False): 
    print("Error opening video stream or file")

fps = 20

#calculate the interval between frame. 
interval = int(1000/fps) 
print("FPS: ",fps, ", interval: ", interval)
# Read the video
cont = 0
while(cap.isOpened()):
    ret, frame = cap.read()
    if ret == True:
        cv2.imwrite(f'Output/1-{cont}.jpeg',frame)
        cont += 1

cap.release()