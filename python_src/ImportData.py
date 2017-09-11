# from svmutil import *
from sklearn.datasets import load_svmlight_file
import numpy as np

def ListDicToArray(XList):	
	d = -1
	n = len(XList)

	for x in XList:
		if bool(x):
			max_d = max(x.keys(),key=int)
			if max_d > d:
				d = max_d
	# d = max(list(max(x.keys(),key=int) for x in XList))
	XArray = np.zeros((n,d))
	for i in range(n):
		x = XList[i]	
		# [XArray[i,int(key)-1] = value, for key, value in XList[i].items()]
		for key, value in x.items():
			XArray[i,int(key)-1] = float(value)
	return XArray

def ImportData(data_dir,data_name):	
	# Y,X = svm_read_problem(data_dir+data_name)
	#change list Y, X to numpy array
	# X = ListDicToArray(X)
	# Y = np.asarray(Y)

	data = load_svmlight_file(data_dir+data_name)
	X = data[0]
	Y = data[1]
	X = X.todense()

	import pdb
	# pdb.set_trace()
	n,d = X.shape
	if (n != Y.size):
		print "X's length is not consistent with Y's length"
		quit()

	min_Y = np.amin(Y)
	max_Y = np.amax(Y)

	########################################################
	########################################################
	# change the label to {-1,+1} in order to train the five simulated experts, 
	# change back to {0,1} when call GF and EWAF
	if min_Y != -1 or max_Y != +1:		
		min_Y_ind = (Y==min_Y)
		max_Y_ind = (Y==max_Y)		
		Y[min_Y_ind] = -1
		Y[max_Y_ind] = +1	
	########################################################
	########################################################

	min_Y = np.amin(Y)
	max_Y = np.amax(Y)

	return X,Y