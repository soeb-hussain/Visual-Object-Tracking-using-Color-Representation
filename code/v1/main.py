#number of samples for the particle filter 
from estimate import * 
from his import * 
from intiTarget_V2 import * 
from part_init import * 
from observ_V2 import * 
from select import * 
from propagate import *
from update import * 
from numpy import ndarray
import time 
import numpy as np
from bhata import * 
import math
def listdir_nohidden(path):
    for f in os.listdir(path):
        if not f.startswith('.'):
            yield f



N_sample = 50
iuo = 1 

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


#tic = time.clock()

im = cv2.imread('/Users/soebhussain/Desktop/Summer2018/VOT2014/ball_im/00000001.jpg')
img2 = im[200:246, 113:159]
#crop_img = intiate_target(im,8)
#cv2.imshow("cropped", im)
#cv2.waitKey(1)
#toc = time.clock()
#print(toc -tic)

#tic = time.clock()
hist, target_array = intiate_target(im,bin,channel)
#1print(hist)
#toc = time.clock()
#print(toc -tic)
#print('target_array') #2123131231313123123122132321323

param = [Vx_max ,Vy_max, sc_max]
#tic = time.clock()


S ,W = intiate_particle(target_array,N_sample,param)

#toc = time.clock()
##tic = time.clock()
#W = observe(im,S,hist,sigma,bin)
#toc = time.clock()
#print(toc -tic)
#print(W)

target = hist
#state of target from intial frames 
# state = [x ,y ,vx,vy,Hx,Hy,sc]
#[x,y] = centroid 
#{vx,vy] = velocities 
#sc = scaling factor 
import cv2
import os

#cv2.imshow("cropped", im)
#cv2.waitKey(1)
#toc = time.cloc


folder = "/Users/soebhussain/Desktop/Summer2018/VOT2014/ball_im"
tum = 0
#cv2.waitKey(0)
for filename in listdir_nohidden(folder):
    img = cv2.imread(os.path.join(folder,filename))
    soebhussain = img
    #cv2.imshow('sahi :D ',img)
    #print("';';';';';';';'")
    #print(target)
    iuo = iuo + 1 
    #print(os.path.join(folder,filename))

    #print('==============')
    #print(len(S))
    #print('==============')

    #tic = time.clock()
    SS = select(S,W)
    #


    #toc = time.clock()
    #print('<><><><><><><><>')
    #print('select')
    #print(toc -tic)

    #print(len(SS))
    #print(len(S))

    #print('==============')
    #print(len(S))
    #print('==============')
    #print(W)
    #print(SS)
    #q = int(input())
    

    tic = time.clock()
    S = propagate(SS,n,target_array,t_1,t_2,t_3,iuo)
    #print("''''''''''''")
    #print(S)
    #rint('-------------')
    #iop = int(input())

    #toc = time.clock()
    #print('==============')
    #print('propagate')
    #print(toc -tic)

    #print('==============')
    #print(len(S))
    #print('==============')




    #tic = time.clock()
    #print(img.shape)
    #for channel in range(0,1):
    W = observe(img,S,target,sigma,bin,channel,W,img2)
    #print(W)
    #print(W)
    W = W/sum(W)
    #print(W)
    toc = time.clock()
    #print('__----____-----______-----_____')
    print('observe')

    print(toc -tic)
    W = W/sum(W)
    
    #tic = time.clock()

    posterior = estimate(S,W)



    #print(posterior)

    #print(W)


    ##print(len(S))
    #print(S)
    #print('==============')
    #toc = time.clock()
    #print(toc -tic)
    #print('esitmate')

    #tic = time.clock()

    
    #d = bhatta_dist(hist,hist)
    #print('p_hyp')
    #print( d)
    #d = bhatta_dist(target,target)
    #print('target')
    #print(d)
    #toc = time.clock()
    #print(toc -tic)
    #print('[[[[[[[[[[[[[[[[[[[[')
    #print('update')

    WinMax = max(W)
    # WinMax> 0.2:
    #target = update(target,posterior,img,bin,covariance)

    print('####################')
    #print(WinMax)
    #print(W)
    W = np.ndarray.tolist(W)
    t = W.index(WinMax)
    sup = S[t] 
    #rec = [sup[1]-sup[5],sup[0]-sup[4],2*sup[5],2*sup[4]]
    #print(WinMax)
    x = int(sup[1] - sup[5])
    y = int(sup[0] - sup[4])
    w = int(2 * sup[5])
    h = int(2* sup[4])
    #print(x,y,w,h)
    cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    #cv2.imshow('yeaaahhh finally something to see',img)
    #print(WinMax)
    #print(t)
    
    if (iuo==1):
        t_1 = sup

    if (iuo == 2):
        t_2 = t_1
        t_1 = sup


    if (iuo >=3):
        t_2 = t_1
        t_3 = t_2
        t_1 = sup

    #if (iuo > 3):


     
    #WinMax = max(W)
    # WinMax> 0.2:
    #target = update(target,posterior,img,bin,covariance)

    #print('####################')
    #print(WinMax)
    #print(W)
    #W = np.ndarray.tolist(W)
    #t = W.index(WinMax)
    sup = posterior 
    #rec = [sup[1]-sup[5],sup[0]-sup[4],2*sup[5],2*sup[4]]
    #print(WinMax)
    x = int(sup[1] - sup[5])
    y = int(sup[0] - sup[4])
    w = int(2 * sup[5])
    h = int(2* sup[4])
    #print(x,y,w,h)
    cv2.rectangle(img,(x,y),(x+w,y+h),(255,232,190),2)
    cv2.imshow('finally something to see',img)
    cv2.waitKey(1)



    





    #toc = time.clock()
    #print(toc -tic)
    #print('imshow')

    #tic = time.clock()
    #cv2.waitKey(1000)
    #tum = tum+1 
    #print('/\/\/\/\/\//\/\/\/\//\/\/\/\/')
    #print(tum)
    #print('/\/\/\/\/\//\/\/\/\//\/\/\/\/')
    #toc = time.clock()
    #print(toc -tic)
    #print('waitKey')

