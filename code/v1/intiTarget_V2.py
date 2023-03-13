#version 2 of project 
# contextual information embedding 
#new target intialization 
#helps us determine what we need to track 
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
	cv2.waitKey(0)
	hist_ref_in,q = histor(8,Cropped_Img)
	for i in range(2,4):
		out_region[i] = int(out_region[i]*1.414)
	
	Cropped_Img = raw_image[(out_region[0]-out_region[2]):(out_region[0]+out_region[2])\
							,(out_region[1]-out_region[3]):(out_region[1]+out_region[3])]
 
	hist_ref_out,w = histor(8,Cropped_Img)
	return hist_ref_in,hist_ref_out,q

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
			cv2.drawContours(img,[ctr[0]],0,(0,0,1),2)
		elif temp==1:
			cv2.drawContours(img,[ctr[0]],0,(0,255,0),1)
		else:
			cv2.drawContours(img,[ctr[0]],0,(255,0,0),1)
		cv2.imshow('visualization',img)
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



def intiate_target(obs , bin,channel):

	from histor import histor
	import cv2
	from histo_un_back import * 
	import numpy as np 


	import math

	temp = [111.0,200.0,111.0,98.0,137.0,98.0,137.0,200.0]

	rotated90=deepcopy(obs)
	# cv2.imshow('original',original)
	# cv2.waitKey(0)
	region = [1,2,3,4,0]
	region[1]= (int(max(temp[0::2])) + int(min(temp[0::2])))/2
	region[3]= (int(max(temp[0::2])) - int(min(temp[0::2])))/2
	region[0]= (int(max(temp[1::2])) + int(min(temp[1::2])))/2
	region[2]= (int(max(temp[1::2])) - int(min(temp[1::2])))/2

	trashi,x_i,y_i,sum_RGB,hist_ref_in,trash,trash = getfbBBoxMask(region,rotated90,3)
	out_region = region[:]
	for i in range(2,4):
		out_region[i] = round(region[i]*1.414) 
	rotated90 = deepcopy(obs)
	trash_img,x_i_trashi,y_itrashi,sum_rgb_trash,hist_ref_out,trash,trash = getfbBBoxMask(out_region,rotated90,3)
	s_ref = hist_ref_in/(hist_ref_out+0.0000000001)
	for i in range(len(s_ref)):
		if s_ref[i]<=0.50:
			hist_ref_in[i]=0
			hist_ref_out[i]=0
			s_ref[i]=0


	su = [[0, 0],[0, 0]]
	su_trans = np.asarray([0, 0])
	for i in range(len(x_i)):
		t = [  [ y_i[i] , x_i[i] ]  ]
		p = [ [ (y_i[i]) ] , [ (x_i[i]) ] ]
		yo = np.matmul(p,t)
		
		su = su + yo*s_ref[sum_RGB[i]]/sum(s_ref)
		t = np.asarray([  y_i[i] , x_i[i]  ])
		su_trans = su_trans + t*s_ref[sum_RGB[i]]/sum(s_ref)

	u,li=np.linalg.eig(su)
	#ratioLW = max(fbBoundingBox[2:4])/min(fbBoundingBox[2:4])
	ratioMAX = math.sqrt(max(u)) #round(fbBoundingBox[2]/math.sqrt(max(u)))
	ratioMIN = math.sqrt(min(u))
	print(u,s_ref.sum())
	out=[]
	out.append(hist_ref_in)
	out.append(hist_ref_out)
	out.append(s_ref)
	out.append([ratioMAX,ratioMIN])

	info = region[:]#[ 113, 200, 46, 46 ]   #x,y,2Hx,2Hy 
	tx = info[0] + info[2]/2   # centroid 
	ty = info[1] + info[3]/2  # centroid 


	Hy  = info[2]/2
	Hx  = info[3]/2 # Hx


	targetparticle = [tx , ty, 0, 0, Hx ,Hy, 1]
	Cx = int(1.414*Hx)
	Cy = int(1.414*Hy)
	contextual = [tx , ty, 0, 0, Cx , Cy , 1]


	Point_inita = [tx - Hx, ty - Hy]
	point_final = [tx + Hx, ty + Hy] 



	Point_inita_cont = [tx - Cx, ty - Cy]
	point_final_cont = [tx + Cx, ty + Cy] 
	#print(Point_inita)
	#print(point_final)

	cap = obs[Point_inita[0]:point_final[0], Point_inita[1]:point_final[1]] 

	Neighbour_cap = obs[Point_inita_cont[0]:point_final_cont[0], Point_inita_cont[1]:point_final_cont[1]]


	# mid = []
	# cv2.imshow('sdasdasw',trash_img)
	# # cv2.waitKey(1)
	# mid.append(Hx)
	# mid.append(Hy)

	# weight_factor = math.sqrt(mid[0]**2 + mid[1]**2)
	# target , target_unweighted = histor(bin,cap,mid,weight_factor)
	# back_plus_fore_unweighted = histor_unweighted_background(bin,cap)
	# target_weighted_deleted = target
	# back_unweighted = np.asarray(back_plus_fore_unweighted)

	# print(target.shape)

	# for i in range(len(target)):
	# 	back_unweighted[i] = back_plus_fore_unweighted[i] - target_unweighted[i]
	# 	if ((target[i]) <= (back_unweighted[i]) ):
	# 		target_weighted_deleted[i] = 0 





	

	#checking parameters _________
	#print(weight_factor)
	#print(mid[1])
	#print(bin)
	#checking parameter^^^^^^^^^^^^^
	
	return out , targetparticle
