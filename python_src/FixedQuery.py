
import numpy as np
import random

import collections
import pickle
import os, pickle, yaml

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pylab import figure

from NormalizeData import *
from ImportData import *
from VariedQuery import *
from PlotVariedQuery import *
def SetOptions(options,data_name):
	if 	data_name == 'a8a':
		options['a8a']={}

		options['a8a']['AEWAF'] = np.concatenate((np.linspace(0.05,0.99,num=20),np.linspace(0.99,1,num=10)))
		options['a8a']['RAEWAF'] = np.concatenate((np.linspace(0.000001,0.000005,num=5),np.linspace(0.000005,0.005,num=15),np.linspace(0.005,0.5,num=10)))	
		options['a8a']['AGF'] = np.concatenate((np.linspace(0.05,0.99,num=20),np.linspace(0.99,1,num=10))) 
		options['a8a']['RAGF'] = np.concatenate((np.linspace(0.,0.00005,num=5),np.linspace(0.00005,0.05,num=15),np.linspace(0.05,0.5,num=10)))			
		options['a8a']['normalized_functions'] = ['l0']
	elif data_name == 'magic04':
		options['magic04']={}		

		options['magic04']['AEWAF'] = [0.9999999999999]#np.concatenate((np.linspace(0.05,0.99,num=20),np.linspace(0.99,1,num=10)))
		options['magic04']['RAEWAF'] = [9e-06]#np.concatenate((np.linspace(0.000001,0.000005,num=5),np.linspace(0.000005,0.005,num=15),np.linspace(0.005,0.5,num=10)))	
		options['magic04']['AGF'] = [0.66]#np.concatenate((np.linspace(0.05,0.99,num=20),np.linspace(0.99,1,num=10)))
		options['magic04']['RAGF'] = [1e-6]#p.concatenate((np.linspace(0.,0.00005,num=10),np.linspace(0.00005,0.05,num=10),np.linspace(0.05,0.5,num=10)))		
		options['RGF'] = [0.2]
		options['REWAF'] = [0.2]
		options['magic04']['normalized_functions']=['std_scale'] #'l0','l2','std_scale',
	elif data_name == 'codrna':		
		options['codrna']={}			
		options['codrna']['AEWAF'] = np.concatenate((np.linspace(0.05,0.5,num=15),np.linspace(0.5,1,num=15)))
		options['codrna']['RAEWAF'] = np.concatenate((np.linspace(0.000001,0.000005,num=5),np.linspace(0.000005,0.005,num=15),np.linspace(0.005,0.5,num=10)))	
		options['codrna']['AGF'] = np.concatenate((np.linspace(0.05,0.5,num=15),np.linspace(0.5,1,num=15)))
		options['codrna']['RAGF'] = np.concatenate((np.linspace(0.,0.00005,num=5),np.linspace(0.00005,0.05,num=15),np.linspace(0.05,0.5,num=10)))		
		options['codrna']['normalized_functions'] = ['l0','l2','min_max_scale','std_scale']
	elif data_name == 'covtype':
		options['covtype']={}				
		options['covtype']['AEWAF'] =np.concatenate((np.linspace(0.05,0.5,num=15),np.linspace(0.5,1,num=15)))
		options['covtype']['RAEWAF'] = np.concatenate((np.linspace(0.000001,0.000005,num=5),np.linspace(0.000005,0.005,num=15),np.linspace(0.005,0.5,num=10)))	
		options['covtype']['AGF'] = np.concatenate((np.linspace(0.05,0.5,num=15),np.linspace(0.5,1,num=15)))
		options['covtype']['RAGF'] = np.concatenate((np.linspace(0.,0.00005,num=5),np.linspace(0.00005,0.05,num=15),np.linspace(0.05,0.5,num=10)))
		options['covtype']['normalized_functions'] = ['l0','min_max_scale','std_scale']
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
	
	elif data_name == 'w8a_pro':			
		options['w8a_pro']={}				
		options['w8a_pro']['AEWAF'] = np.concatenate((np.linspace(0.05,0.6,num=15),np.linspace(0.6,1,num=15)))
		options['w8a_pro']['RAEWAF'] = np.concatenate((np.linspace(0.000001,0.000005,num=5),np.linspace(0.000005,0.005,num=15),np.linspace(0.005,0.5,num=10)))	
		options['w8a_pro']['AGF'] = np.concatenate((np.linspace(0.05,0.5,num=15),np.linspace(0.5,1,num=15)))
		options['w8a_pro']['RAGF'] = np.concatenate((np.linspace(0.,0.00005,num=10),np.linspace(0.00005,0.05,num=10),np.linspace(0.05,0.5,num=10)))	
		options['w8a_pro']['normalized_functions']=['l0'] 		

def BatchVariedQuery(options):
	options['data_dir'] = 'D:/Copy/dataset/'		
	options['num_true_experts'] = 5
	options['algorithm_names'] = ['EWAF','AEWAF','RAEWAF','REWAF','GF','AGF','RAGF','RGF']
	options['output_file_extension'] = '.pdf'
	options['deltas'] = [0.2]#np.linspace(0,1,num=options['num_queries'])
	# options['deltas'][0] = 0.1
	# options['deltas'][options['num_queries']-1] = 0.2
	for data_name in options['data_names']:
		print data_name
		options['data_name'] = data_name
		SetOptions(options,data_name)
		
		X,Y = ImportData(options['data_dir'],options['data_name'])
		n,d = X.shape
		print '# Instances: ',n, ' # Features: ',d
		
		for num_noisy_experts in options['num_noisy_experts_list']:
			print 'num_noisy_experts', num_noisy_experts
			options['num_noisy_experts'] = num_noisy_experts
			options['num_all_experts'] = options['num_true_experts'] + options['num_noisy_experts']	

			for normalized_function in options[data_name]['normalized_functions']:
				output_file_name = options['output_dir']+data_name+'_'+str(num_noisy_experts)+'_'+normalized_function
				options['output_file_name'] = output_file_name

				options['normalized_function'] = normalized_function
				X = NormalizeData(X,normalized_function)
				
				print options['output_file_name']
				results_all_query = VariedQuery(options,X,Y)			
				
				# write the varied query ratio to file				
				# out =open(output_file_name+'.txt','w') 
				# pickle.dump(results_all_query,out)
				# out.close()

				# write the fixed query ratio to file
				with open(output_file_name+'_fixed.txt','w') as fin:
					for i in range(options['num_queries']):
						fin.write('\n')
						fin.write(('& Algorithm').ljust(12))  
						fin.write(('& Query(\%)').ljust(20))
						fin.write(('& Regret(\%)').ljust(20))
						fin.write(('& Time(s)').ljust(15))
						fin.write(('delta').ljust(0))
						fin.write('\n')
						for alg_name in options['algorithm_names']:
							print alg_name, results_all_query[alg_name]['que'][i]
							fin.write(('& '+alg_name).ljust(12))
							fin.write(('& '+str((results_all_query[alg_name]['que'][i]))+'$\pm$'+str((results_all_query[alg_name]['stdQue'][i]))).ljust(20))
							fin.write(('& '+str((results_all_query[alg_name]['reg'][i]))+'$\pm$'+str((results_all_query[alg_name]['stdReg'][i]))).ljust(20))
							fin.write(('& '+str((results_all_query[alg_name]['time'][i]))).ljust(15))
							if alg_name in ['AEWAF','RAEWAF','AGF','RAGF']:
								fin.write((str(options[data_name][alg_name][i])).ljust(0))
							else:
								fin.write((str(options['deltas'][i])).ljust(0))
							fin.write('\n')	

				print 'Finished dataset: ', data_name
				
				# num_queries = options['num_queries']
				# selected_indexes = {}
				# selected_indexes['AEWAF'] = range(0,num_queries)
				# selected_indexes['RAEWAF'] = range(0,num_queries)
				# selected_indexes['REWAF'] = range(0,num_queries)
				# selected_indexes['AGF'] = range(0,num_queries)
				# selected_indexes['RAGF'] = range(0,num_queries)
				# selected_indexes['RGF'] = range(0,num_queries)
				# options['selected_indexes'] = selected_indexes

				# PlotVariedQuery(results_all_query,options)

if __name__ == '__main__':
 	options={}
	options['num_folds'] = 1
	options['num_queries'] = 1

	# normal settings
	# print 'Normal Setting'
	options['output_dir'] = '../results/fixed/'			
	options['num_noisy_experts_list'] = [0]
	options['data_names']=['magic04']#,'mushrooms','spambase','svmguide1','a8a','w8a_pro']
	BatchVariedQuery(options)