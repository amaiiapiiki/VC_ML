import os
import cv2
import shutil
import numpy as np
import pandas as pd

sections = {'1':np.array([[762, 300],[690,353],[971,353],[975,300]]),
            '2':np.array([[975,300],[971,353],[1247,353],[1185,300]]),
            '3':np.array([[800,271],[762, 300],[975,300],[976,267]]),
            '4':np.array([[970,268],[970,300],[1185,300],[1151,268]]),
            '5':np.array([[443,370],[672,250],[1271,244],[1496,363],[1250,356,],[1151,271],[800,273],[689,362]])}

def paint(image, c):
    if len(image.split('-')) == 3:
        image = image.split('-')[0] + '-' + image.split('-')[2]

    img = cv2.imread('images/'+image)

    img2 = cv2.imread('images/'+image)

    cv2.fillPoly(img2, pts=[sections[str(c)]], color=(0, 0, 255))

    added = cv2.addWeighted(img2, 0.3, img, 0.7, 0)
    return added, image


if __name__ == "__main__":

    if os.path.exists('finalResults/'):
        shutil.rmtree('finalResults/')

    os.mkdir('finalResults/')

    df = pd.read_csv('dataset.csv')

    for index, row in df.iterrows():
        newImage, name = paint(row['image'], int(row['class']))
        cv2.imwrite('finalResults/'+name, newImage)