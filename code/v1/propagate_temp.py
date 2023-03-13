import random
import numpy as np
def propagate_temp(S,W_old,temp_S,W):
	global_index_W = np.where(W==(max(W)))
	global_index_W_old = np.where(W_old == (max(W_old)))
	#print(S[global_index_W_old],'...........',W_old[global_index_W_old],W[global_index_W])
	if W_old[global_index_W_old][0]>W[global_index_W][0]:
		globall = S[global_index_W_old][:]
	else:
		globall = temp_S[global_index_W_old][:]

	aplha = 0.1
	beta = 0.1
	gamma = 1 - aplha - beta
	for i in range(len(S)):
		if W_old[i]>W[i]:
			temp_S[i]= S[i][:]
		# print(S[i][0],temp_S[i],globall)
		print(S[i][0])
		S[i][0] = gamma*S[i][0] + beta*(temp_S[i][0]-S[i][0])*random.uniform(0, 1) + aplha*random.uniform(0, 1)*(globall[0][0]-S[i][0])
		print(S[i][0],'..........',temp_S[i][0],'......',globall[0])
		S[i][1] = gamma*S[i][1] + beta*(temp_S[i][0]-S[i][1])*random.uniform(0, 1) + aplha*random.uniform(0, 1)*(globall[0][1]-S[i][1])

		S[i][4] = gamma*S[i][4] + beta*(temp_S[i][0]-S[i][4])*random.uniform(0, 1) + aplha*random.uniform(0, 1)*(globall[0][4]-S[i][4])
		S[i][5] = gamma*S[i][5] + beta*(temp_S[i][0]-S[i][5])*random.uniform(0, 1) + aplha*random.uniform(0, 1)*(globall[0][5]-S[i][5])
		
		# S[i][2]=max(min(ps[2],vmaxy),-vmaxy)#-vmax if ps[2]<-vmax else ps[2]
  #       S[i][3]=max(min(ps[3],vmaxx),-vmaxx)#-vmax if ps[2]<-vmax else ps[2]
        #ps[2]=vmax if ps[2]>vmax else ps[2]  
        #ps[3]=-vmax if ps[3]<-vmax else ps[3]
        #p
        #print(S)
        #tu = int(input())
        #return S    
        #S[i][3]=vmax if ps[3]>vmax else ps[3]
        # S[i][4]=20 if ps[4]<20 else min(ps[4],50)#ps[4]
        # #ps[4]=30 if ps[4]>0 else min(ps[],30)
        # S[i][5]=20 if ps[5]<20 else min(ps[5],50)


        # S[i][4] =  min(S[i][4],S[i][5])
        # S[i][5] =  min(S[i][4],S[i][5])
        # #ps[5]=30 if ps[5]>30 else min(ps[5],30)
        # S[i][6]=-0.1 if ps[6]<-0.1 else min(0.1,ps[6])
        # S[i][0]= S[i][4]if ps[1]<60 else ps[0]
        # S[i][0]= (height-S[i][4]) if ps[0]>height-S[i][4] else ps[0]
        # S[i][1]= S[i][5]if ps[1]<60 else ps[1]
        # S[i][1]= (width-S[i][5]) if ps[1]>width-S[i][5] else ps[1]
        #ps[5] = 27
        #ps[4] = 27
	return S