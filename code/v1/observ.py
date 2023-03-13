#observe function 
#input is frame , sample set S , target histo , sigma value 
# and bins

#output is updated weight W
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

	S_array = np.asarray(S)
	N = S_array.shape[0]
	#if channel==0:
	W = np.zeros((N))
	constant_w = 1/math.sqrt(2*(math.pi)*(sigma**2))
	#print('!!!!!!!!!!!!!!!!!!!!')
	#print(constant_w)
	#print('observe')
	sum_bta = 0
	sum_hi = 0 

	for i in range(N): 
		s = S[i]
		#print('********************')
		#print(s)
		tl = [max(1,s[0] - s[4]) , max(1,s[1]-s[5])]
		brx = s[0] + s[4]
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
		soeb = np.zeros((int(mid[0]*2) ,int(mid[1]*2)))
		window = obs[int(tl[0]) : int(br[0]), int(tl[1]):int(br[1]) ]
		sup = s
		img = obs
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
		#cv2.rectangle(img,(x,y),(x+w,y+h),(0,i,0),1)
		#cv2.imshow('see',img)
		#cv2.waitKey(1)
		#cv2.imshow('asd',window)
		#cv2.waitKey(1)

		#for l in range(int(mid[0]*2)):
		#	for j in range(int(mid[1]*2)):
		#		soeb[l][j] = 1 # math.sqrt((mid[0]-l)**2 + (mid[1]-j)**2)
		a = linalg.norm([window.shape[0], window.shape[1] ])
		so = [ int(window.shape[0]/2), int(window.shape[1]/2)]
		tic = time.clock()
		#d = main_his_bh(img2, window)

		p_hyp = histor(bin,window,so,a)
		toc = time.clock()
		sum_hi = sum_hi + toc - tic




		d = bhatta_dist(p_hyp,target)
		tic = time.clock()

		sum_bta = sum_bta + tic - toc
		#print('###############')
		#print(i)
		#print('op')
		#print('................')
		#print(p_hyp)
		#print('ssssssssssssssss')
		W[i] =  constant_w * math.exp(-(d**2)/(2*(sigma**2)))
		#print(W[i])




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
	#print('======++++++======')
	#print(W[i])
	#print('[][][][]][][][][')
	#rint(sum(nnW))
	#print('[][][][]][][][][')

	#W = W/sum(nnW)
	print('bhata')
	print(sum_bta)
	print('histor')
	print(sum_hi)
	return W


