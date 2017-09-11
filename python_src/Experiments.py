
import numpy as np
import random
from numpy import random, mean, std, sqrt
import numpy as np
from PERCEPTRON import *
from ALMA import *
from AROW import *
from ROMMA import *
from PA1 import *
from EWAF import *
from GF import *
from math import *
import collections
import pickle
import os, pickle, yaml, errno

import matplotlib.pyplot as plt
from matplotlib.pylab import figure

from NormalizeData import NormalizeData
from ImportData import ImportData
from PlotVariedQuery import PlotVariedQuery
from VariedQuery import VariedQuery

def SetOptions(options,data_name):
	if 	data_name == 'a8a':
		options['a8a']={}
		options['a8a']['AEWAF'] = np.concatenate((np.linspace(0.05,0.99,num=20),np.linspace(0.99,1,num=10)))
		options['a8a']['RAEWAF'] = np.concatenate((np.linspace(0.000001,0.000005,num=5),np.linspace(0.000005,0.005,num=15),np.linspace(0.005,0.5,num=10)))	
		options['a8a']['AGF'] = np.concatenate((np.linspace(0.05,0.99,num=20),np.linspace(0.99,1,num=10))) 
		options['a8a']['RAGF'] = np.concatenate((np.linspace(0.,0.00005,num=5),np.linspace(0.00005,0.05,num=15),np.linspace(0.05,0.5,num=10)))			
		options['a8a']['normalized_functions'] = ['l0']
	elif data_name == 'codrna':		
		options['codrna']={}			
		options['codrna']['AEWAF'] = np.concatenate((np.linspace(0.05,0.5,num=15),np.linspace(0.5,1,num=15)))
		options['codrna']['RAEWAF'] = np.concatenate((np.linspace(0.000001,0.000005,num=5),np.linspace(0.000005,0.005,num=15),np.linspace(0.005,0.5,num=10)))	
		options['codrna']['AGF'] = np.concatenate((np.linspace(0.05,0.5,num=15),np.linspace(0.5,1,num=15)))
		options['codrna']['RAGF'] = np.concatenate((np.linspace(0.,0.00005,num=5),np.linspace(0.00005,0.05,num=15),np.linspace(0.05,0.5,num=10)))	
		options['codrna']['normalized_functions'] = ['std_scale','l2']
	elif data_name == 'covtype':
		options['covtype']={}				
		options['covtype']['AEWAF'] =np.concatenate((np.linspace(0.05,0.5,num=15),np.linspace(0.5,1,num=15)))
		options['covtype']['RAEWAF'] = np.concatenate((np.linspace(0.000001,0.000005,num=5),np.linspace(0.000005,0.005,num=15),np.linspace(0.005,0.5,num=10)))	
		options['covtype']['AGF'] = np.concatenate((np.linspace(0.05,0.5,num=15),np.linspace(0.5,1,num=15)))
		options['covtype']['RAGF'] = np.concatenate((np.linspace(0.,0.00005,num=5),np.linspace(0.00005,0.05,num=15),np.linspace(0.05,0.5,num=10)))	
		options['covtype']['normalized_functions'] = ['std_scale','l2']
	elif data_name == 'gisette':
		options['gisette']={}				
		options['gisette']['AEWAF'] = np.linspace(0.05,1,num=30)#np.concatenate((np.linspace(0.05,0.99,num=20),np.linspace(0.99,1.01,num=10)))
		options['gisette']['RAEWAF'] = np.concatenate((np.linspace(0.000001,0.000005,num=5),np.linspace(0.000005,0.005,num=15),np.linspace(0.005,0.5,num=10)))	
		options['gisette']['AGF'] = np.linspace(0.05,1,num=30)#np.concatenate((np.linspace(0.051,0.99,num=20),np.linspace(0.99,1.01,num=10)))
		options['gisette']['RAGF'] = np.concatenate((np.linspace(0.,0.00005,num=5),np.linspace(0.00005,0.05,num=15),np.linspace(0.05,0.5,num=10)))	
		options['gisette']['normalized_functions'] = ['std_scale','l2']
	elif data_name == 'mushrooms':
		options['mushrooms']={}				
		options['mushrooms']['AEWAF'] = np.concatenate((np.linspace(0.05,0.7,num=10),np.linspace(0.7,1,num=20)))
		options['mushrooms']['RAEWAF'] = np.concatenate((np.linspace(0.000001,0.000005,num=5),np.linspace(0.000005,0.005,num=15),np.linspace(0.005,0.5,num=10)))	
		options['mushrooms']['AGF'] = np.concatenate((np.linspace(0.05,0.7,num=10),np.linspace(0.7,1,num=20)))
		options['mushrooms']['RAGF'] = np.concatenate((np.linspace(0.,0.00005,num=5),np.linspace(0.00005,0.05,num=15),np.linspace(0.05,0.5,num=10)))	
		options['mushrooms']['normalized_functions'] = ['min_max_scale']
	elif data_name == 'spambase':			
		options['spambase']={}			
		options['spambase']['AEWAF'] = np.concatenate((np.linspace(0.05,0.99,num=15),np.linspace(0.99,1,num=15)))
		options['spambase']['RAEWAF'] = np.concatenate((np.linspace(0.000001,0.000005,num=5),np.linspace(0.000005,0.005,num=15),np.linspace(0.005,0.5,num=10)))	
		options['spambase']['AGF'] = np.concatenate((np.linspace(0.05,0.8,num=15),np.linspace(0.8,1,num=15)))
		options['spambase']['RAGF'] = np.concatenate((np.linspace(0.,0.00005,num=5),np.linspace(0.00005,0.05,num=15),np.linspace(0.05,0.5,num=10)))	
		options['spambase']['normalized_functions'] = ['std_scale']
	elif data_name == 'svmguide1':			
		options['svmguide1']={}			
		options['svmguide1']['AEWAF'] = np.concatenate((np.linspace(0.05,0.8,num=15),np.linspace(0.8,1,num=15)))
		options['svmguide1']['RAEWAF'] = np.concatenate((np.linspace(0.000001,0.000005,num=5),np.linspace(0.000005,0.005,num=15),np.linspace(0.005,0.5,num=10)))	
		options['svmguide1']['AGF'] = np.concatenate((np.linspace(0.05,0.5,num=15),np.linspace(0.5,1,num=15)))
		options['svmguide1']['RAGF'] = np.concatenate((np.linspace(0.,0.00005,num=5),np.linspace(0.00005,0.05,num=15),np.linspace(0.05,0.5,num=10)))		
		options['svmguide1']['normalized_functions'] = ['std_scale']
	elif data_name == 'magic04':
		options['magic04']={}			
		options['magic04']['AEWAF'] = np.linspace(0.05,1,num=30)#np.concatenate((np.linspace(0.05,0.99,num=20),np.linspace(0.99,1,num=10)))
		options['magic04']['RAEWAF'] = np.concatenate((np.linspace(0.000001,0.000005,num=5),np.linspace(0.000005,0.005,num=5),np.linspace(0.006,0.06,num=10),np.linspace(0.061,0.5,num=10)))	
		options['magic04']['AGF'] = np.linspace(0.05,1,num=30)#np.concatenate((np.linspace(0.05,0.7,num=10),np.linspace(0.7,0.8,num=15),np.linspace(0.8,1,num=5)))
		options['magic04']['RAGF'] = np.concatenate((np.linspace(0.,0.00005,num=10),np.linspace(0.00005,0.05,num=10),np.linspace(0.05,0.5,num=10)))	
		options['magic04']['normalized_functions']=['std_scale']#,'l0','l2','min_max_scale']
	elif data_name == 'w8a_pro':			
		options['w8a_pro']={}				
		options['w8a_pro']['AEWAF'] = np.concatenate((np.linspace(0.05,0.6,num=15),np.linspace(0.6,1,num=15)))
		options['w8a_pro']['RAEWAF'] = np.concatenate((np.linspace(0.000001,0.000005,num=5),np.linspace(0.000005,0.005,num=15),np.linspace(0.005,0.5,num=10)))	
		options['w8a_pro']['AGF'] = np.concatenate((np.linspace(0.05,0.5,num=15),np.linspace(0.5,1,num=15)))
		options['w8a_pro']['RAGF'] = np.concatenate((np.linspace(0.,0.00005,num=10),np.linspace(0.00005,0.05,num=10),np.linspace(0.05,0.5,num=10)))	
		options['w8a_pro']['normalized_functions']=['l0'] 		

def BatchVariedQuery(options):	
	
	for data_name in options['data_names']:
		print data_name
		# output_file_name = options['output_dir']+data_name

		options['data_name'] = data_name
		SetOptions(options,data_name)
		
		X,Y = ImportData(options['data_dir'],options['data_name'])
		n,d = X.shape
		print 'n=', n, 'd=',d
		
		for normalized_function in options[data_name]['normalized_functions']:
			# output_file_name = options['output_dir']+data_name+'_'+normalized_function
			
			options['normalized_function'] = normalized_function
			X = NormalizeData(X,normalized_function)
						
			p = range(n)
			random.shuffle(p)
			num_train = int(n*0.2)
			num_test = n - num_train		
			
			X_train = X[p[0:num_train],:]	
			Y_train = Y[p[0:num_train]]
			X_test = X[p[num_train:n],:]
			Y_test = Y[p[num_train:n]]

			acc1,w1 = PERCEPTRON(X_train,Y_train)		
			print("PERCEPTRON: {}".format(acc1))

			acc2,w2 = ROMMA(X_train,Y_train)
			print("ROMMA: {}".format(acc2))

			acc3,w3 = ALMA(X_train,Y_train)		
			print("ALMA: {}".format(acc3))

			acc4,w4 = PA1(X_train,Y_train)		
			print("PA1: {}".format(acc4))

			acc5,w5 = AROW(X_train,Y_train)
			print("AROW: {}".format(acc5))

			# print 'Experts Accuracy', acc1,acc2,acc3,acc4,acc5
			W_normal = np.concatenate((np.reshape(w1,(1,d)),np.reshape(w2,(1,d)),\
				np.reshape(w3,(1,d)),np.reshape(w4,(1,d)),np.reshape(w5,(1,d))),axis=0)
					
			test_indexes_all_folds = []
			for k in range(options['num_folds']):
				pk = range(num_test)
				random.shuffle(pk)
				test_indexes_all_folds.append(pk)
			options['test_indexes_all_folds'] = test_indexes_all_folds

			for num_noisy_experts in options['num_noisy_experts_list']:
				# output_file_name = output_file_name + '_' + str(num_noisy_experts)
				options['output_file_name'] = options['output_dir'] + data_name + '_' + str(num_noisy_experts) + '_' + normalized_function
				
				print 'num_noisy_experts', num_noisy_experts
				options['num_noisy_experts'] = num_noisy_experts
				options['num_all_experts'] = options['num_true_experts'] + options['num_noisy_experts']	

				W_all = W_normal
				
				for indNoisyExperts in range(options['num_noisy_experts']):					
					W_all = np.concatenate((W_all,np.zeros((1,d))),axis=0)
				options['W'] = W_all
				
				########################################################
				########################################################
				results_all_query = VariedQuery(options,X_test,Y_test)	
				########################################################
				########################################################		
				
				# write the varied query ratio to file				
				results_out =open(options['output_file_name']+'.pkl','w') 
				pickle.dump(results_all_query,results_out)
				results_out.close()

				option_out =open(options['output_dir'] + data_name + '_' + 'options.pkl', 'w') 
				pickle.dump(results_all_query,option_out)
				option_out.close()

				# write the fixed query ratio to file
				with open(options['output_file_name']+'_fixed.txt','w') as fin:
					for i in range(options['num_queries']):
						fin.write('\n')
						fin.write(('& Algorithm').ljust(12))  
						fin.write(('& Query(\%)').ljust(20))
						fin.write(('& Accuracy(\%)').ljust(20))
						fin.write(('& Time(s)').ljust(15))
						fin.write(('delta').ljust(0))
						fin.write('\n')
						for alg_name in options['algorithm_names']:
							# print alg_name, results_all_query[alg_name]['que'][i]
							fin.write(('& '+alg_name).ljust(12))
							fin.write(('& '+str((round(results_all_query[alg_name]['que'][i]*100,3)))+'$\pm$'+str(round(results_all_query[alg_name]['stdQue'][i]*100,3))).ljust(20))
							fin.write(('& '+str((results_all_query[alg_name]['acc'][i]))+'$\pm$'+str((results_all_query[alg_name]['stdAcc'][i]))).ljust(20))
							fin.write(('& '+str((results_all_query[alg_name]['time'][i]))).ljust(15))
							if alg_name in ['AEWAF','RAEWAF','AGF','RAGF']:
								fin.write((str(options[data_name][alg_name][i])).ljust(0))
							else:
								fin.write((str(options['deltas'][i])).ljust(0))
							fin.write('\n')					
				
				num_queries = options['num_queries']
				selected_indexes = {}
				selected_indexes['AEWAF'] = range(0,num_queries)
				selected_indexes['RAEWAF'] = range(0,num_queries)
				selected_indexes['REWAF'] = range(0,num_queries)
				selected_indexes['AGF'] = range(0,num_queries)
				selected_indexes['RAGF'] = range(0,num_queries)
				selected_indexes['RGF'] = range(0,num_queries)
				options['selected_indexes'] = selected_indexes

				PlotVariedQuery(results_all_query,options)

	print "Finished dataset: ", data_name

if __name__ == '__main__':
 	
 	options={}
 	options['data_dir'] = '../data/'	
	options['output_dir'] = '../results/AAAI2018/'
	
	try:
	    os.makedirs(options['output_dir'])
	except OSError, e:
		if e.errno != errno.EEXIST:
			raise

	options['num_true_experts'] = 5
	options['algorithm_names'] = ['EWAF','AEWAF','RAEWAF','REWAF','GF','AGF','RAGF','RGF']#,'REWAF1','RGF1']
	options['output_file_extension'] = '.pdf'		
	
	options['num_folds'] = 5
	options['num_queries'] = 30

	# threshold to control the query ratio
	options['deltas'] = np.linspace(0,1,num=options['num_queries'])
	options['deltas'][3] = 0.1
	options['deltas'][6] = 0.2	
	
	options['num_noisy_experts_list'] = [0]
	options['data_names']=['a8a','mushrooms'] 
	#['mushrooms','spambase','svmguide1','a8a','w8a_pro']
	BatchVariedQuery(options)