import cv2
import numpy as np
import matplotlib.pyplot as plt
from os import listdir
from os.path import isfile, join
import time


def main_his_bh(imq,imd):
	n=1
	#Takes all of the images on the file specified folder
	#mypath = "images/"
	#fileNames = [ f for f in listdir(mypath) if isfile(join(mypath,f)) ]
	#imd = np.empty(len(fileNames), dtype=object)
	#for n in range(0, len(fileNames)):
	#	imd[n] = cv2.imread( join(mypath,fileNames[n]) )

	#print("Starting image matching process.")

	#tic = time.clock()
	#Goes through each and every single image setting the first one as query, and all as database.
	#print(len(imd))
	#for n in range(0, 1):

		#imq = imd.copy()

		#templateTopFour, templateMatchingScores = templateMatching(imd, imq, n)
		#topFourImages(templateTopFour, fileNames, n, "Template Matching")
		#imageScorePlacing(templateMatchingScores, fileNames, n, "Template Matching")
	histogramMatchingScores = histogramMatching(imd, imq, n)
		#topFourImages(histogramTopFour, fileNames, n, "Histogram Matching")
		#imageScorePlacing(histogramMatchingScores, fileNames, n, "Histogram Matching")

		#siftTopFour, siftMatchingScores = SIFT(imd, imq, n)
		#topFourImages(siftTopFour, fileNames, n, "SIFT")
		#imageScorePlacing(siftMatchingScores, fileNames, n, "SIFT")

		#print
	#toc = time.clock()
	#print(toc - tic)
	return (1 - histogramMatchingScores)	



def histogramMatching(imd, imq, imageNumber):
	#dict to hold final histogram matching results.
	#topFour = {}
	#totalScores = {}

	#calculate histogram of query image.
	histq = cv2.calcHist([imq], [2], None, [256], [0, 256])
	
	#calculate histogram of each database image.
	#histd = np.empty(len(imd), dtype=object)
	#for n in range(0, len(imd)):
	histd = cv2.calcHist([imd], [0], None, [256], [0, 256])

	#go through all the methods on each image and take average.
	for n in range(0, 1):
		resArr = []
		resArr.append(cv2.compareHist(histq, histd, cv2.HISTCMP_CORREL))
		resArr.append(1 - cv2.compareHist(histq, histd, cv2.HISTCMP_BHATTACHARYYA))

		score = np.sum(resArr) / len(resArr)
		totalScores = score

		#save top 4 scores
		#if len(topFour) < 4:
		#	topFour[n] = score
		#else:
			#topFour = topFourDictionary(topFour, n, score)

	return  totalScores
