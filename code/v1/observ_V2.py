#observe function 
#input is frame , sample set S , target histo , sigma value 
# and bins

#output is updated weight W
from score import score

from copy import deepcopy

def observe(obs,S,target,sigma,bin,channel,W,img2):

	
	import numpy as np 
	import math
	from histor import * 
	from bhata import * 
	from numpy import linalg
	import cv2
	import time
	from bhat_his import * 
	#print(obs.shape)
	from histo_un_back import * 

	S_array = np.asarray(S)
	N = S_array.shape[0]
	
	#if channel==0:
	dont_target_weighted_deleted = target
	W = np.zeros((N))
	W_old = np.zeros((N))
	constant_w = 1/math.sqrt(2*(math.pi)*(sigma**2))
	#print('!!!!!!!!!!!!!!!!!!!!')
	#print(constant_w)
	#print('observe')
	sum_bta = 0
	sum_hi = 0 
	Point_inita_cont = [3,2]

	for i in range(N): 
		s = S[i]
		#print('********************')
		#print(s)
		tl = [max(1,s[0] - s[4]) , max(1,s[1]-s[5])]
		brx = s[0] + s[4]
		Cx = int(1.414*s[4])
		Cy = int(1.414*s[5])
		tx = s[0]
		ty = s[1]
		contextual = [tx , ty, 0, 0, Cx , Cy , 1]
		Point_inita_cont = [max(1,tx - Cx), max(1,ty - Cy)]
		point_final_cont = [min(tx + Cx,obs.shape[0]), min(obs.shape[1],ty + Cy)] 
		#print(obs.shape)
		if brx > obs.shape[0]:
			brx = obs.shape[0]
		bry = s[1] + s[5]
		if bry > obs.shape[1]:
			brx = obs.shape[1]
		br = [brx , bry]
		mid = []
		mid.append(s[4]/2)
		mid.append(s[5]/2)
		#print(Point_inita_cont)
		#print(point_final_cont)
		#soeb = np.zeros((int(mid[0]*2) ,int(mid[1]*2)))
		window = obs[int(tl[0]) : int(br[0]), int(tl[1]):int(br[1]) ]
		Neighbour_cap = obs[int(Point_inita_cont[0]):int(point_final_cont[0]), int(Point_inita_cont[1]):int(point_final_cont[1])]
		cap = window
		sup = s

		#cv2.imshow('in observe',window)
		#cv2.waitKey(0)

		

		#target_weighted_deleted[:] = 0 if (target[:] < back_unweighted[:]) else target[:]

		#sup = posterior
		#print(posterior)
		#print(sup)
		#np.seterr(divide='ignore', invalid='ignore')
		#tic = time.clock()
		#rec = [sup[1]-sup[5],sup[0]-sup[4],2*sup[5],2*sup[4]]
		#print(rec)
		x = int(sup[1] - sup[5])
		y = int(sup[0] - sup[4])
		w = int(2 * sup[5])
		h = int(2* sup[4])
		#print(x,y,w,h)

		img = obs
		#print(i)
		#cv2.rectangle(img,(x,y),(x+w,y+h),(((i*5)%255),((i*5)%255),((i*5)%255)),1)
		#cv2.imshow('see',img)
		#cv2.waitKey(1)
		#cv2.imshow('asd',window)
		#cv2.waitKey(1)

		#for l in range(int(mid[0]*2)):
		#	for j in range(int(mid[1]*2)):
		#		soeb[l][j] = 1 # math.sqrt((mid[0]-l)**2 + (mid[1]-j)**2)
		a = linalg.norm([window.shape[0], window.shape[1] ])
		so = [ int(window.shape[0]/2), int(window.shape[1]/2)]
		hist_ref_in=target[0]
		hist_ref_out=target[1]
		s_ref = target[2]

		[ratioMAX,ratioMIN]=target[3]
		# cv2.imshow('check in observe',img)

		# cv2.waitKey(0)
		S[i],d,a = score(s,img,hist_ref_in,hist_ref_out,s_ref,ratioMAX,ratioMIN)
		s = S[i]
		#print('###############')
		#print(i)
		#print('op')
		#print('................')
		# #print(p_hyp)
		# #print('ssssssssssssssss')

		W[i] =  constant_w * math.exp(-(d**2)/(2*(sigma**2)))
		W_old[i] =  constant_w * math.exp(-(a**2)/(2*(sigma**2)))
		# #print(W[i])




	#for i in range(N):
		#s = S[i]
		#rec = [s[1]-s[5],s[0]-s[4],2*s[5],2*s[4]]
		#x = int(s[1] - s[5])
		#y = int(s[0] - s[4])
		#w = int(2 * s[5])
		#h = int(2* s[4])
		#cv2.rectangle(obs,(x,y),(x+w,y+h),(255,120,0),1)
		#print('aaaaaaaaaaa')
		#cv2.imshow('finally something to see',obs)
		#cv2.waitKey(1000)
		#print(i)
		#print(W[i])




	nnW = W
	nn_W_old = W_old
	#teamo = input()
	#print('======++++++======')
	#print(W[i])
	#print('[][][][]][][][][')
	#rint(sum(nnW))
	#print('[][][][]][][][][')

	#W = W/sum(nnW)
	#W_old = W_old/sum(W_old)
	#print('bhata')
	#print(sum_bta)
	#print('histor')
	#print(sum_hi)
	return W,W_old,S


