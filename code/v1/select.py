# selection of sample to form new set 

#select function 

#input 
# S - set of particle 
# W set of weight 

def select(S,W):
	import numpy as np 
	from random import *

	SS =[]

	#print(len(S))


	N = (np.asarray(S)).shape[0]
	C = np.zeros((N))

	for i in range(1,N):
		C[i] = C[i-1] +W[i]
		
		
	C = C/sum(C)
	#print(C)
	#print(len(C))
	
	for i in range(N):
		N = (np.asarray(S)).shape[0]
		C = np.zeros((N))

		for i in range(1,N):
			C[i] = C[i-1] +W[i]
		
		
		C = C/sum(C)
		CT = C
		tre = random() * C[N-1]
		#t = CT - tre
		for j in range(N):
			if CT[j]<tre:
				CT[j] = 1

		#print(CT)
		toe = min(CT) 
		#inde = CT.index(tre)
		inde = CT.tolist().index(toe)
		#print(inde)
		#print(tre)
		#print(tre)
		#index = int(10 *tre)
		#you = [x for x in CT if x >= tre]
		#i = CT.index(you)
		#for j in range(0,len(CT)):
		#	max = 1.0 
		#	if CT[j] < tre:
		#		CT[j] = 1.0
		#	if max > CT[j]:
		#		max = CT[j]
		#		index = j
		#		print('-------')
		#		print(tre)
		#		print(CT[j])

		#print('1111111111111')		
		
		#print("  index" + str(index))
		#print('================')

		#print(CT[inde])
		#print(inde)
		#print(tre)
		#print(toe)
		#io = int(input())		 
		#if index!=(-1):
		#print(inde)

		SS.append(S[int(inde)]) 
	return SS			





		





