#propagate function 

#this function propagates a simple function to a dynamic model 

#input SS and n - noise vector
# s is propagated state 

def propagate(SS, n_vector,tar,t_1,t_2,t_3,iuo,iteration,so):
    import numpy as np 
    import random
    import math
    SS_array = np.asarray(SS)
    #print('sssssssssssss')
    #print(SS)
    #print('sssssssssssss')
    Nu = SS_array.shape[0]
    Nue = 7
    S = np.zeros((Nu,Nue))
    
    A = np.identity(7)
    #SS_array.shape[1]
    #N = np.zeros((Nu))
    A[0][2] = 1 #random.randint(-2,2)
    A[1][3] = 1 #random.randint(-2,2)
    width = 320
    height = 240
    
    s = SS[0]

    s = np.asarray(s)
    ps = np.matmul(A,s)

    for i in range(Nu):
        s = SS[i]
        s = np.asarray(s)
        A[0][2] = 1#random.randint(-2,2)
        A[1][3] = 1#random.randint(-2,2)

        A[4][4]= 1 +  s[6]
        A[5][5]= 1 +  s[6]
        temp = np.matmul(s,A)
        #ps = np.matmul(A,s)
        for j in range(Nue):
            #A = np.identity(7)
            #A[0][2] = random.randint(-2,2)
            #A[1][3] = random.randint(-2,2)
            #A[4][4]= 1 + random.randint(-2,2) * s[6]
            #A[5][5]= 1 + random.randint(-2,2) * s[6]
            Ran = random.randint(-1,1)
            ps[j] = temp[j] +  Ran*math.sqrt(n_vector[j])

        #value_when_true if condition   
        #ps[0]=  max(min(ps[0],300),60)
        #ps[1]=  max(min(ps[1],180),60)
        
        
        vmaxy = 2
        vmaxx = 2 
        if iuo>=3:
            vmaxy = int((iteration+1-so)/2)*abs(int(1.5*t_1[0] - 2.0*t_2[0] + 0.5*t_3[0]))*(int(iteration+2-so)/2)  
            vmaxy = int((iteration+1-so)/2)*abs(int(1.5*t_1[1] - 2.0*t_2[1] + 0.5*t_3[1]))*(int(iteration+2-so)/2) 

          #random.randint(0,2) #tar[2];
        S[i][2]=max(min(ps[2],vmaxy),-vmaxy)#-vmax if ps[2]<-vmax else ps[2]
        S[i][3]=max(min(ps[3],vmaxx),-vmaxx)#-vmax if ps[2]<-vmax else ps[2]
        #ps[2]=vmax if ps[2]>vmax else ps[2]  
        #ps[3]=-vmax if ps[3]<-vmax else ps[3]
        #p
        #print(S)
        #tu = int(input())
        #return S    
        #S[i][3]=vmax if ps[3]>vmax else ps[3]
        S[i][4]=20 if ps[4]<20 else min(ps[4],50)#ps[4]
        #ps[4]=30 if ps[4]>0 else min(ps[],30)
        S[i][5]=20 if ps[5]<20 else min(ps[5],50)


        S[i][4] =  min(S[i][4],S[i][5])
        S[i][5] =  min(S[i][4],S[i][5])
        #ps[5]=30 if ps[5]>30 else min(ps[5],30)
        S[i][6]=-0.1 if ps[6]<-0.1 else min(0.1,ps[6])
        S[i][0]= S[i][4]if ps[1]<60 else ps[0]
        S[i][0]= (height-S[i][4]) if ps[0]>height-S[i][4] else ps[0]
        S[i][1]= S[i][5]if ps[1]<60 else ps[1]
        S[i][1]= (width-S[i][5]) if ps[1]>width-S[i][5] else ps[1]
        #ps[5] = 27
        #ps[4] = 27
        #print(ps)
        #print(i)
        #iop = int(input())
        #S.append(ps)
    return S




