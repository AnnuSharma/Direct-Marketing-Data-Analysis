import matplotlib.pyplot as plt
import csv
import random as rnd
import numpy
import sys
from numpy import *
from numpy.linalg import inv
from scipy.spatial.distance import pdist, squareform
import scipy
from sklearn import preprocessing
from sklearn import tree
from sklearn import svm
from sklearn import linear_model
from sklearn.ensemble import RandomForestClassifier

numpy.set_printoptions(threshold = 'nan')

testData = []
trainData = []
dtError = 0
svmError = 0
rfError = 0
lrError = 0


def main():
	dataImport()

def dataImport():
	global testData
	global trainData

	marketingTrain = open('/Users/annu/Desktop/Deposit_Marketing/Training.csv', 'rb')
	trainData = list(csv.reader(marketingTrain, delimiter = ','))

	marketingTest = open('/Users/annu/Desktop/Deposit_Marketing/Testing.csv', 'rb')
	testData = list(csv.reader(marketingTest, delimiter = ','))

	trainData = numpy.array(trainData)
	testData = numpy.array(testData)
	
	#One of K encoding of categorical data
	encoder = preprocessing.LabelEncoder()
	for j in (1,2,3,4,6,7,8,10,14,15):
		trainData[:,j] = encoder.fit_transform(trainData[:,j])
		testData[:,j] = encoder.fit_transform(testData[:,j])

	#Converting numpy strings to floats
	trainData = trainData.astype(numpy.float)
	testData = testData.astype(numpy.float)

	learnDecisionTree() #Good with handling categorical attributes
	learnRF() #Simple Ensemble classifier, must try
	learnSVM() #Tested
	learnLogReg() #Tested
	
def learnDecisionTree():
	global dtError

	DT = tree.DecisionTreeClassifier(max_depth=6)
	DT = DT.fit(trainData[:,:-1], trainData[:,-1])
	predictionsDT = DT.predict(testData[:,:-1])
	
	#validating predicions
	for i in range(0,len(testData)):
		if(testData[i][15] != predictionsDT[i]):
			dtError = dtError+1
	print "DT Error : ", float(dtError)/len(testData)*100.0

def learnSVM():
	global svmError

	SVM = svm.SVC()
	SVM = SVM.fit(trainData[:,:-1], trainData[:,-1])
	predictionsSVM = SVM.predict(testData[:,:-1])

	#validating predicions
	for i in range(0,len(testData)):
		if(testData[i][15] != predictionsSVM[i]):
			svmError = svmError+1
	print "SVM Error : ", float(svmError)/len(testData)*100.0

def learnRF():
	global rfError

	RF = RandomForestClassifier(n_estimators=40,max_depth=10)
	RF.fit(trainData[:,:-1], trainData[:,-1])	
	predictionsRF = RF.predict(testData[:,:-1])

	#validating predicions
	for i in range(0,len(testData)):
		if(testData[i][15] != predictionsRF[i]):
			rfError = rfError+1
	print "RF Error : ", float(rfError)/len(testData)*100.0

def learnLogReg():
	global lrError

	LR = linear_model.LogisticRegression(C = 5)
	LR.fit(trainData[:,:-1], trainData[:,-1])
	predictionsLR = LR.predict(testData[:,:-1])

	#validating predicions
	for i in range(0,len(testData)):
		if(testData[i][15] != predictionsLR[i]):
			lrError = lrError+1
	print "LR Error : ", float(lrError)/len(testData)*100.0

if __name__ == '__main__':
	main()