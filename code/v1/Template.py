import cv2
import numpy as np 
from histor import histor
from bhatu import bhatta_dist
import math
from bhat_his import * 
bin = 8
im = cv2.imread('/Users/soebhussain/Desktop/Summer2018/VOT2014/ball_img/0000000.jpg')
#img2 = im.crop(( 200.35, 113.74, 245.48, 159.32))
#200.35,159.32,
#200.35,113.74,
#45.48,113.74,
#245.48,159.32
crop = im[114:160 , 200:246]
sigma = 0.1
constant_w = 1/math.sqrt(2*(math.pi)*(sigma**2))


brop = im[134:180 , 200:246] # brop is plus 20 

#crop = np.zeros((46,46))
x = 200
y = 113
w = 46
h = 46

#d = main_his_bh(crop,srop)
#print(d)


trop = im[114:160 , 210:256] # trop is + in every dimen
srop = im[1:47 , 1:47]
#x,y,w,h = cv2.boundingRect(im)
cv2.rectangle(im,(x,y),(x+w,y+h),(255,0,0),2)

cv2.rectangle(im,(x+5,y+5),(x+5+w,y+h+5),(255,0,0),1)
cv2.rectangle(im,(1,1),(47,47),(255,0,255),1)


br = histor(bin, brop)
cr = histor(bin,crop)
tr = histor(bin,trop)
sr = histor(bin,srop)
h = [ cr , br ];
#d = bhatta_dist(cr,r)
#print(d)

def mean( hist ):
    mean = 0.0;
    for i in hist:
        mean += i;
    mean/= len(hist);
    return mean;

def bhatta ( hist1,  hist2):
    # calculate mean of hist1
    h1_ = mean(hist1);

    # calculate mean of hist2
    h2_ = mean(hist2);

    # calculate score
    score = 0;
    for i in range(len(hist1)):
        score += math.sqrt( hist1[i] * hist2[i] );
    # print h1_,h2_,score;
    score = math.sqrt(abs( 1 - ( 1 / math.sqrt(abs(h1_*h2_*8*8) ) ) * score ));
    return score;

# generate and output scores
scores = [];
for i in range(len(h)):
    score = [];
    for j in range(len(h)):
        score.append( bhatta(h[i],h[j]) );
    scores.append(score);

sum2 = max(scores[0])
for i in range(len(h)):
    scores[0][i] = 1 - (scores[0][i]/sum2)

print(scores[0][1])

d = main_his_bh(crop, brop)
print('main_his_bh')
print(1-d)


cv2.imshow("cropped", im)
#cv2.imshow('as',crop_img)
cv2.waitKey(10000)

