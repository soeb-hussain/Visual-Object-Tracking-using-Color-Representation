import cv2
import os
from matplotlib.pyplot import imshow
import matplotlib.pyplot as plt
import numpy as np
from copy import deepcopy
from matplotlib.path import Path
from subprocess import check_output
import math
from pylab import *
from numpy import (array, dot, arccos, clip)
from numpy.linalg import norm
import time
def histor(bin,cap):
    import numpy as np
    import math
    from copy import deepcopy
    
    soeb = np.asarray(cap)
    Red = np.asarray(soeb[:,:,0])
    Blue = np.asarray(soeb[:,:,1])
    green = np.asarray(soeb[:,:,2])
    r = (Red//32) * 64
    b = (Blue//32)* 8
    g = green//32
    sum = r + g+ b
    hist_un = np.zeros((512,1))
    #print(sum)
    for i in range(Red.shape[0]):
        for j in range(Red.shape[1]):
            hist_un[sum[i][j]] = hist_un[sum[i][j]] + 1
    return hist_un,sum

def bhatta_dist(p,q):




    h = [ p , q];
    import math
    import numpy as np

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
            score += math.sqrt( abs(hist1[i] * hist2[i]) );
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

    #print(1 - scores[0][1])
    sum2 = max(scores[0])
    
    scores[0][i] = 1 - (scores[0][i]/sum2)
    return (scores[0][1])
import math

def rotate(origin, point, angle):
    """
    Rotate a point counterclockwise by a given angle around a given origin.

    The angle should be given in radians.
    """
    ox, oy = origin
    px, py = point

    qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
    qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)
    return qx, qy

def tracker_dat_intialize(raw_image,region):



    out_region = region[:]
    Cropped_Img = raw_image[(out_region[0]-out_region[2]):(out_region[0]+out_region[2])\
                            ,(out_region[1]-out_region[3]):(out_region[1]+out_region[3])]
    cv2.imshow('Check in tracker_dat_intialize',Cropped_Img)
    cv2.waitKey(1)
    hist_ref_in,q = histor(8,Cropped_Img)
    for i in range(2,4):
        out_region[i] = int(out_region[i]*1.414)
    
    Cropped_Img = raw_image[(out_region[0]-out_region[2]):(out_region[0]+out_region[2])\
                            ,(out_region[1]-out_region[3]):(out_region[1]+out_region[3])]
 
    hist_ref_out,w = histor(8,Cropped_Img)
    return hist_ref_in,hist_ref_out,q

# def getfbBBoxMask(fbBoundingBox,img,temp):
#     '''
#     gives the mask for bounding fbBoundingBox
#     '''

#     x1 = fbBoundingBox[0]-round(fbBoundingBox[2])
#     x2 = fbBoundingBox[0]+round(fbBoundingBox[2])
#     y1 = fbBoundingBox[1]-round(fbBoundingBox[3])
#     y2 = fbBoundingBox[1]+round(fbBoundingBox[3])


#     Cx = (x1+x2)//2
#     Cy = (y1+y2)//2
#     # print(Cx,Cy)
#     Nx11,Ny11 = rotate((Cx,Cy), (x1,y1), math.radians(fbBoundingBox[4]))
#     Nx12,Ny12 = rotate((Cx,Cy), (x1,y2), math.radians(fbBoundingBox[4]))
#     Nx21,Ny21 = rotate((Cx,Cy), (x2,y1), math.radians(fbBoundingBox[4]))
#     Nx22,Ny22 = rotate((Cx,Cy), (x2,y2), math.radians(fbBoundingBox[4]))
    
#     nx, ny,trash = img.shape
#     x = [Nx11 ,Nx12 ,Nx21,Nx22]
#     y = [Ny11 ,Ny12 ,Ny21,Ny22]
    
#     poly_verts = [(x[0],y[0]),(x[2],y[2]),(x[3],y[3]),(x[1],y[1])]


#     ctrl = [[y[0],x[0]],[y[2],x[2]],[y[3],x[3]],[y[1],x[1]]]
#     ctr = [np.array(ctrl,dtype=np.int32)]
#     if temp!=3:
#         if temp==0:
#             cv2.drawContours(img,[ctr[0]],0,(0,0,1),2)
#         elif temp==1:
#             cv2.drawContours(img,[ctr[0]],0,(0,255,0),1)
#         else:
#             cv2.drawContours(img,[ctr[0]],0,(255,0,0),1)
#         # cv2.imshow('visualization',img)
#         # cv2.waitKey(0)

#     # Create vertex coordinates for each grid cell...
#     # (<0,0> is at the top left of the grid in this system)

#     x, y = np.meshgrid(np.arange(nx), np.arange(ny))
#     x, y = x.flatten(), y.flatten()
    

#     points = np.vstack((x,y)).T

#     path = Path(poly_verts)
#     grid = path.contains_points(points)
#     grid = grid.reshape((ny,nx))
    
#     l,m=grid.shape
#     # print(l,m,'XxxxxxxxxXXXXXXXXX')
    
#     x_i = []
#     y_i = []
#     sum_RGB = []
#     hist = np.zeros((512,1))
#     # print(grid.shape,img.shape)
#     raw_image=img[:,:,:]
#     #print(l,m,raw_image.shape,grid.shape)
#     for j in range(1,m+1):
#         for i in range(1,l+1):
#             if grid[i-1,j-1]==False:
#                 raw_image[j-1,i-1,:]=(0,0,0)
#                 #print(False)
#             else:
#                 y_i.append(j-Cx)
#                 x_i.append(i-Cy)
#                 t = ((raw_image[j-1][i-1][0]//32) * 64) + ((raw_image[j-1][i-1][1]//32) * 8) + (raw_image[j-1][i-1][2]//32)
#                 #print(i,j,((raw_image[j,i,0]//32) * 64) + ((raw_image[j,i,0]//32) * 8) + (raw_image[j,i,0]//32))
#                 sum_RGB.append(t)
#                 #print(t)
#                 hist[t]=hist[t] + 1
#     #print(len(sum_RGB),'[][][][]][')
#     #cv2.imshow('in fbboundingbox',raw_image)
#     #cv2.waitKey(0)  
#     return raw_image,x_i,y_i,sum_RGB,hist,Cx,Cy


def getfbBBoxMask(fbBoundingBox,img,temp):
    '''
    gives the mask for bounding fbBoundingBox
    '''

    x1 = fbBoundingBox[0]-round(fbBoundingBox[2])
    x2 = fbBoundingBox[0]+round(fbBoundingBox[2])
    y1 = fbBoundingBox[1]-round(fbBoundingBox[3])
    y2 = fbBoundingBox[1]+round(fbBoundingBox[3])


    Cx = (x1+x2)//2
    Cy = (y1+y2)//2
    # print(Cx,Cy)
    Nx11,Ny11 = rotate((Cx,Cy), (x1,y1), math.radians(fbBoundingBox[4]))
    Nx12,Ny12 = rotate((Cx,Cy), (x1,y2), math.radians(fbBoundingBox[4]))
    Nx21,Ny21 = rotate((Cx,Cy), (x2,y1), math.radians(fbBoundingBox[4]))
    Nx22,Ny22 = rotate((Cx,Cy), (x2,y2), math.radians(fbBoundingBox[4]))
    
    nx, ny,trash = img.shape
    x = [Nx11 ,Nx12 ,Nx21,Nx22]
    y = [Ny11 ,Ny12 ,Ny21,Ny22]
    
    poly_verts = [(x[0],y[0]),(x[2],y[2]),(x[3],y[3]),(x[1],y[1])]


    ctrl = [[y[0],x[0]],[y[2],x[2]],[y[3],x[3]],[y[1],x[1]]]
    ctr = [np.array(ctrl,dtype=np.int32)]
    original_o = deepcopy(img)
    if temp!=3:
        if temp==0:
            cv2.drawContours(original_o,[ctr[0]],0,(0,0,1),2)
        elif temp==1:
            cv2.drawContours(original_o,[ctr[0]],0,(0,255,0),1)
        else:
            cv2.drawContours(original_o,[ctr[0]],0,(255,0,0),1)
        cv2.imshow('visualization',original_o)
        cv2.waitKey(1)

    # Create vertex coordinates for each grid cell...
    # (<0,0> is at the top left of the grid in this system)

    x, y = np.meshgrid(np.arange(nx), np.arange(ny))
    x, y = x.flatten(), y.flatten()
    

    points = np.vstack((x,y)).T

    path = Path(poly_verts)
    grid = path.contains_points(points)
    grid = grid.reshape((ny,nx))
    
    l,m=grid.shape
    # print(l,m,'XxxxxxxxxXXXXXXXXX')
    
    x_i = []
    y_i = []
    sum_RGB = []
    hist = np.zeros((512,1))
    # print(grid.shape,img.shape)
    raw_image=img[:,:,:]
    #print(l,m,raw_image.shape,grid.shape)
    for j in range(1,m+1):
        for i in range(1,l+1):
            if grid[i-1,j-1]==False:
                raw_image[j-1,i-1,:]=(0,0,0)
                #print(False)
            else:
                y_i.append(j-Cx)
                x_i.append(i-Cy)
                t = ((raw_image[j-1][i-1][0]//32) * 64) + ((raw_image[j-1][i-1][1]//32) * 8) + (raw_image[j-1][i-1][2]//32)
                #print(i,j,((raw_image[j,i,0]//32) * 64) + ((raw_image[j,i,0]//32) * 8) + (raw_image[j,i,0]//32))
                sum_RGB.append(t)
                #print(t)
                hist[t]=hist[t] + 1
    #print(len(sum_RGB),'[][][][]][')
    #cv2.imshow('in fbboundingbox',raw_image)
    #cv2.waitKey(0)  
    return raw_image,x_i,y_i,sum_RGB,hist,Cx,Cy



def score(S,original,hist_ref_in,hist_ref_out,s_ref,ratioMAX,ratioMIN):
    ut = np.array([1,0])
    fbBoundingBox=[S[0],S[1],S[4],S[5],0]

    #rotated90 = deepcopy(original)
    res=[]
    temp=2
    res_score=[]
    res_bb=[]
    tim = 0

    for _ in range(10):
        tic = time.clock()
        rotated90 = deepcopy(original)
        #print(fbBoundingBox)
        raw_image,x_i,y_i,sum_RGB,hist,Cx,Cy = getfbBBoxMask(fbBoundingBox,rotated90,1)
        fbBoundingBox_out=fbBoundingBox[:]
        for i in range(2,4):
            fbBoundingBox_out[i]=round(fbBoundingBox_out[i]*1.414)
        # cv2.imshow('pasas',raw_image)
        # cv2.waitKey(0)
        # print(_)
        rotated90 = deepcopy(original)
        raw_image_out,x_iy,y_iy,sum_RGBy,hist_out,Cx,Cy = getfbBBoxMask(fbBoundingBox_out,rotated90,3)
        # cv2.imshow('asas',raw_image_out)
        # cv2.waitKey(1)
        s_new = hist/(hist_out+0.000000000001)
        s_F = np.copy(s_new)
        s_B = np.copy(s_new)
        for i in range(len(hist_ref_in)):
            if s_new[i]<0.50:
                hist[i]=0
                hist_out[i]=0
                s_new[i]=0
            s_F[i] = s_new[i]*hist[i]
            s_B[i] = s_new[i]*(hist_out[i]-hist[i])

        score = s_F.sum()/(hist.sum())
        s_D = bhatta_dist(hist_ref_in,hist)
        res_score.append(score*s_D)
        res_bb.append(fbBoundingBox)
        su = [[0,0],[0,0]]
        su_trans = np.asarray([0,0])
        temp2 = deepcopy(original)
        # print(s_ref.sum(),'______________')
        for i in range(len(x_i)):
            t = [  [ y_i[i] , x_i[i] ]  ]
            p = [ [ (y_i[i]) ] , [ (x_i[i]) ] ]
            yo = np.matmul(p,t)
            su = su + yo*s_ref[sum_RGB[i]]/sum(s_ref)
            t = np.asarray([  y_i[i] , x_i[i]  ])
            su_trans = su_trans + t*s_ref[sum_RGB[i]]/sum(s_ref)

        #print(max(y_i),max(x_i),'he he he hasni rinkiya k papa')

        fbBoundingBox[0] = fbBoundingBox[0]+round(su_trans[0]/max(y_i))
        fbBoundingBox[1] = fbBoundingBox[1]+round(su_trans[1]/max(x_i))
        # if (fbBoundingBox[0]>=original.shape[0] - 60) or (fbBoundingBox[0]<= 60):
        #     fbBoundingBox[0] = min(original.shape[0] - 60,fbBoundingBox[0])
        #     fbBoundingBox[0] = max(60,fbBoundingBox[0])
        
        # if (fbBoundingBox[1]>=original.shape[1] - 60) or (fbBoundingBox[1]<= 60):
        #     fbBoundingBox[1] = min(original.shape[1] - 60,fbBoundingBox[1])
        #     fbBoundingBox[1] = max(60,fbBoundingBox[1])

        u,li = np.linalg.eig(su)
        # print(u,ratioMIN,ratioMAX,'..........')
        # print(su_trans)
        # print(su)
        fbBoundingBox[2] = fbBoundingBox[2] * (ratioMAX)/ math.sqrt(max(u))
        fbBoundingBox[3] = fbBoundingBox[3] * (ratioMIN)/math.sqrt(min(u))

        if fbBoundingBox[2]>(original.shape[0])/2:
            fbBoundingBox[2]= min(fbBoundingBox[0],original.shape[0]/2 - fbBoundingBox[0])

        if fbBoundingBox[3]>(original.shape[1])/2:
            fbBoundingBox[3]= min(fbBoundingBox[1],original.shape[1]/2 - fbBoundingBox[3])
        #print(u)
        if u[0]>u[1]:
            height = 0 
        else:
            height = 1
        v,u = li[height], np.array([1,0])
        c = dot(ut,v)/norm(ut)/norm(v)
        #correction = (180*arccos(clip(c, -1, 1))/3.14)
        ut=v[:]
        #fbBoundingBox[4]=(fbBoundingBox[4]+correction)%180
        x1 = fbBoundingBox[0]-round(fbBoundingBox[2])
        x2 = fbBoundingBox[0]+round(fbBoundingBox[2])
        y1 = fbBoundingBox[1]-round(fbBoundingBox[3])
        y2 = fbBoundingBox[1]+round(fbBoundingBox[3])
        #print(x1,x2,y1,y2)
        toc = time.clock()
        tim = tim + toc-tic
        print(toc-tic)
        # cv2.rectangle(temp2,(int(y1),int(x1)),(int(y2),int(x2)),(((20*5)%255),((20*5)%255),((20*5)%255)),1)
        # cv2.imshow('in score',temp2)
        # cv2.waitKey(1)
    print(tim,'asaaaa')
    max1_score = res_score.index(max(res_score[5:]))
    max1_bb=res_bb[max1_score]
    cv2.rectangle(temp2,(int(y1),int(x1)),(int(y2),int(x2)),(((20*5)%255),((20*5)%255),((20*5)%255)),1)
    cv2.imshow('in score',temp2)
    cv2.waitKey(1)

    S=[max1_bb[0],max1_bb[1],0,0,max1_bb[2],max1_bb[3],1]
    return S,res_score[max1_score],res_score[0]
        










