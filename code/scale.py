import cv2
import os
from matplotlib.pyplot import imshow
import matplotlib.pyplot as plt
import numpy as np
from copy import deepcopy
from matplotlib.path import Path
from subprocess import check_output
import math

##functions 
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


def tracker_dat_intialize(raw_image,region):
    out_region = region[:]
    Cropped_Img = raw_image[(out_region[0]-out_region[2]):(out_region[0]+out_region[2])\
                            ,(out_region[1]-out_region[3]):(out_region[1]+out_region[3])]
    cv2.imshow('Check in tracker_dat_intialize',Cropped_Img)
    cv2.waitKey(0)
    hist_ref_in,q = histor(8,Cropped_Img)
    for i in range(2,4):
        out_region[i] = int(out_region[i]*1.414)
    
    Cropped_Img = raw_image[(out_region[0]-out_region[2]):(out_region[0]+out_region[2])\
                            ,(out_region[1]-out_region[3]):(out_region[1]+out_region[3])]
 
    hist_ref_out,w = histor(8,Cropped_Img)
    return hist_ref_in,hist_ref_out,q


def rotate(origin, point, angle):
    """
    Rotate a point counterclockwise by a given angle around a given origin.

    The angle should be given in radians.
    """
    ox, oy = origin
    px, py = point

    qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
    qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)
    return round(qx), round(qy)


#fbBoundingBox = [centre x , centre y , x length , y height, angle in degree ]
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
    if temp!=3:
        if temp==0:
            cv2.drawContours(original_o,[ctr[0]],0,(0,0,1),2)
        elif temp==1:
            cv2.drawContours(original_o,[ctr[0]],0,(0,255,0),1)
        else:
            cv2.drawContours(original_o,[ctr[0]],0,(255,0,0),1)
        cv2.imshow('visualization',original_o)
        cv2.waitKey(0)

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


##original Code

original_o = cv2.imread('/Users/soebhussain/AnsarSoeb/bin/BTP/sequence/00000001.jpg')
original = deepcopy(original_o)
# cv2.imshow('original',original)
# cv2.waitKey(0)
region = [original.shape[0]/2 - 20 ,original.shape[1]/2 - 60,original.shape[0]/7,original.shape[1]/24]
Cropped_Img = original[(region[0]-region[2]):(region[0]+region[2]),(region[1]-region[3]):(region[1]+region[3])]

# cv2.imshow('Cropped check' ,Cropped_Img)
# cv2.waitKey(0)



#rotae the image

(h,w) = original.shape[:2]
# print(h,w)

center = (w/2,h/2)
# print(center)
#angle = 90

scale = 1.0

rotated90 = deepcopy(original)

region = [region[0],region[1],region[2],region[3]]


correction = 0
region.append(0)
trashi,x_i,y_i,sum_RGB,hist_ref_in,trash,trash = getfbBBoxMask(region,rotated90,3)
cv2.imshow('reference ',trashi)

out_region = region[:]
for i in range(2,4):
    out_region[i] = round(region[i]*1.414) 
rotated90 = deepcopy(original)
trash_img,x_i_trashi,y_itrashi,sum_rgb_trash,hist_ref_out,trash,trash = getfbBBoxMask(out_region,rotated90,3)

s_ref = hist_ref_in/(hist_ref_out + 0.00000001)
angle = 0


rotated90 = deepcopy(original)

fbBoundingBox = region[:4] + [0]
ut = np.array([1,0])
print('-----[]-----[]-----[]-----[]-----')
# cv2.imshow('reference out_region ',trash_img)
# cv2.waitKey(0)
run = True
temp=0
count = 0
while run:
    #fbBoundingBox[4]= (correction) 
    rotated90 = deepcopy(original)
    raw_image,x_i,y_i,sum_RGB,hist,Cx,Cy = getfbBBoxMask(fbBoundingBox,rotated90,temp)
    #print(x_i)
    cv2.imshow('check rotation  ',raw_image)
    cv2.waitKey(0)

    #print('~~~~similiarity between calculated hist and current hist~~~~~',bhatta_dist(hist, hist_ref_in),'~~~~~~~~~')
    if bhatta_dist(hist, hist_ref_in)<0.0001:
        run = True
        print('cvcv')
    fbBoundingBox_out = fbBoundingBox[:]
    for i in range(2,4):
        fbBoundingBox_out[i] = round(fbBoundingBox[i]*1.414) 
    rotated90 = deepcopy(original)
    raw_image_out,x_iy,y_iy,sum_RGBy,hist_out,Cx,Cy = getfbBBoxMask(fbBoundingBox_out,rotated90,3)

    s_new = hist/(hist_out + 0.00000001)

    print('----sum of s for reference and current candidate----',s_ref.sum(),s_new.sum(),'------')
    if s_ref.sum() > s_new.sum():
        EP=1
    EP =1
    ## for calculation of angle xi,yi,sum,histogram,s, and center is required
    # cv2.imshow('candidate out_region ',raw_image_out)
    # cv2.waitKey(0)

    su = [[0, 0],[0, 0]]
    
    for i in range(len(x_i)):
        t = [  [ y_i[i] , x_i[i] ]  ]
        p = [ [ (y_i[i]) ] , [ (x_i[i]) ] ]
        yo = np.matmul(p,t)
        #print(sum_RGB,s_ref[sum_RGB[i]]
        su = su + yo*s_ref[sum_RGB[i]]/s_ref.sum()

    u,li=np.linalg.eig(su)
    print(',,,,,,,',u)
    if count ==0:
        
        count =1 
        ratioLW = max(fbBoundingBox[2:4])/min(fbBoundingBox[2:4])
        ratioMAX = math.sqrt(max(u))#round(fbBoundingBox[2]/math.sqrt(max(u)))
        ratioMIN = math.sqrt(min(u))#round(fbBoundingBox[3]/math.sqrt(min(u)))
        fbBoundingBox[2]= round(fbBoundingBox[3])
        fbBoundingBox[3]=round(fbBoundingBox[2])
        temp = 1

    else:
        fbBoundingBox[2]= fbBoundingBox[2] *(ratioMAX)/ math.sqrt(max(u))
        fbBoundingBox[3]= fbBoundingBox[3]*(ratioMIN)/math.sqrt(min(u))


    
    #if u[0]>u[1]:
    #    height = 0
    #else:
    #    height = 1

    












#cv2.destroyAllWindows() # destroys the window showing image



























