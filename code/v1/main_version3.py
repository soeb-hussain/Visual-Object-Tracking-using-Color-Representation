#number of samples for the particle filter 
from estimate import * 
from his import * 
from intiTarget_V2 import * 
from part_init import * 
from observ_V2 import * 
from select import * 
from propagate_version_3 import *
from update import * 
from numpy import ndarray
import time 
import numpy as np
from bhata import * 
import math
from propagate_temp import propagate_temp
def listdir_nohidden(path):
    for f in os.listdir(path):
        if not f.startswith('.'):
            yield f



N_sample = 5
iuo = 1 
iteration = 4

#number of bins for histograms
bin = 8

#sigme for ubdate  step 
sigma = 0.1

#constant_w = 1/math.sqrt(2*(math.pi)*(sigma**2))
#Noise vector for dynamic model X(t+1) = AX(t) + Noise of size 7 
n = [200,160, 200,200,30,30,0.1]

#range of particle during intialization
Vx_max = 20
Vy_max = 20
sc_max = 0.1

#covariance value 
covariance = 0.1
t_2 = [0,0,0,0,0,0,0] 
t_1 = [0,0,0,0,0,0,0] 
t_3 = [0,0,0,0,0,0,0] 

import cv2
import numpy as np 

channel = 2



im = cv2.imread('/Users/soebhussain/Desktop/BTP_final/sequence2/00000001.jpg')
#img2 = im[200:246, 113:159]

hist, target_array = intiate_target(im,bin,channel)

param = [Vx_max ,Vy_max, sc_max]


S ,W = intiate_particle(target_array,N_sample,param)
temp = [111,200,111,98,137,98,137,200]


# cv2.imshow('original',original)
# cv2.waitKey(0)

img2 = im[min(temp[1::2]):max(temp[1::2]),min(temp[0::2]):max(temp[0::2])]
cv2.imshow('Foreground',img2)
cv2.waitKey(1)
target = hist
#state of target from intial frames 
# state = [x ,y ,vx,vy,Hx,Hy,sc]
#[x,y] = centroid 
#{vx,vy] = velocities 
#sc = scaling factor 
import cv2
import os
import time


folder = "/Users/soebhussain/Desktop/BTP_final/sequence2"
saved = "/Users/soebhussain/Desktop/BTP_final/sequence3"
tum = 0
for filename in listdir_nohidden(folder):
    img = cv2.imread(os.path.join(folder,filename))
    soebhussain = img
    iuo = iuo + 1 
    
    tic = time.clock()
    SS = select(S,W)
    for so in range(iteration):

        
        if so==0:
            S = propagate(SS,n,target_array,t_1,t_2,t_3,iuo,iteration,so)
            W,W_old,temp_S = observe(img,S,target,sigma,bin,channel,W,img2)
        else:
            S = propagate_temp(S,W_old,temp_S,W)
            W,W_old,temp_S = observe(img,S,target,sigma,bin,channel,W,img2)




        t_start = time.clock()
        
        W = W/sum(W)

        t_end = time.clock()

        
        


        if so == iteration-1:
            posterior = estimate(S,W)
            SS = posterior
        
    toc = time.clock()
    print(toc-tic,'full loop')


    sup = posterior



    x = int(sup[1] - sup[5])
    y = int(sup[0] - sup[4])
    w = int(2 * sup[5])
    h = int(2* sup[4])
    cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    if (iuo==1):
        t_1 = sup
    if (iuo == 2):
        t_2 = t_1
        t_1 = sup
    if (iuo >=3):
        t_2 = t_1
        t_3 = t_2
        t_1 = sup
    #target = update(target,posterior,img,bin,covariance)
    sup = posterior 
    x = int(sup[1] - sup[5])
    y = int(sup[0] - sup[4])
    w = int(2 * sup[5])
    h = int(2* sup[4])
    cv2.rectangle(img,(x,y),(x+w,y+h),(255,232,190),1)
    cv2.imshow('finally something to see'+str(filename),img)
    cv2.imwrite(os.path.join(saved,"face-" + str(filename) + ".jpg"), img)
    cv2.waitKey(1)
