import cv2
import os
folder = "/Users/soebhussain/Desktop/style transfer /Summer2018/VOT2014/ball_img"
#def load_images_from_folder(folder):
    #images = []
for filename in os.listdir(folder):
    img = cv2.imread(os.path.join(folder,filename))
    cv2.imshow('sahi :D ',img)
    cv2.waitKey(1)
#cv2.killAllWindows()
    #return images