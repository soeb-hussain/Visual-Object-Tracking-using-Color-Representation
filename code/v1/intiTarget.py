#helps us determine what we need to track 

#input is observation i.e. frame or an image 
#target os output  a prior state that we want to track


#get the boundbox around the tagret 


##w, h = 8, 5;
##Matrix = [[0 for x in range(w)] for y in range(h)] 

#cv2.imshow(img2)
def intiate_target(obs , bin,channel):

	from histor import histor
	import cv2
	


	import math

	info = [ 113, 200, 46, 46 ]   #x,y,2Hx,2Hy 
	tx = info[0] + info[2]/2   # centroid 
	ty = info[1] + info[3]/2  # centroid 


	Hy  = info[2]/2
	Hx  = info[3]/2 # Hx


	targetparticle = [tx , ty, 0, 0, Hx ,Hy, 1]


	Point_inita = [tx - Hx, ty - Hy]
	point_final = [tx + Hx, ty + Hy] 
	#print(Point_inita)
	#print(point_final)

	cap = obs[Point_inita[0]:point_final[0], Point_inita[1]:point_final[1]] 
	mid = []
	cv2.imshow('sdasdasw',cap)
	cv2.waitKey(1)
	mid.append(Hx)
	mid.append(Hy)

	weight_factor = math.sqrt(mid[0]**2 + mid[1]**2)
	target = histor(bin,cap,mid,weight_factor)
	

	#checking parameters _________
	#print(weight_factor)
	#print(mid[1])
	#print(bin)
	#checking parameter^^^^^^^^^^^^^
	
	return target,targetparticle