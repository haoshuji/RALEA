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

def VariedQuery(options,X_test,Y_test):			
	# X = options['X']
	# Y = options['Y']	
	n,d = X_test.shape #n data_name instances with d dimensions

	results_all_query={}

	
	forecaster_options={}
	forecaster_options['data_name'] = options['data_name']
	forecaster_options['output_dir'] = options['output_dir']
	forecaster_options['num_all_experts'] = options['num_all_experts']
	forecaster_options['num_noisy_experts'] = options['num_noisy_experts']
	forecaster_options['normalized_function'] = options['normalized_function']
	forecaster_options['output_file_name'] = options['output_file_name']
	for alg_name in options['algorithm_names']:
		results_all_query[alg_name] = {}
		results_all_query[alg_name]['que']=[]; results_all_query[alg_name]['stdQue']=[]
		results_all_query[alg_name]['reg']=[]; results_all_query[alg_name]['stdReg']=[]
		results_all_query[alg_name]['time']=[]; results_all_query[alg_name]['stdTime']=[]	
	
	forecaster_options['eta'] = math.sqrt(8*log(options['num_all_experts'])/n)
	
	
	forecaster_options['W']=options['W']	
	print 'Query Ratio: '
	for i in range(options['num_queries']):
		print i,
		forecaster_options['que_ind']=i				
		delta = options['deltas'][i]		
		
		_results_one_query={}			
		for alg_name in options['algorithm_names']:
			_results_one_query[alg_name]={}
			_results_one_query[alg_name]['que']=[]
			_results_one_query[alg_name]['reg']=[]
			_results_one_query[alg_name]['time']=[]  		

		for k in range(options['num_folds']):
			# print data_name, '_',i,'_',k	

			forecaster_options['fold_ind'] = k
			
			test_index = options['test_indexes_all_folds'][k]					
			XTest = X_test[test_index,:]
			YTest = Y_test[test_index]
			for alg_name in options['algorithm_names']:
				
				forecaster_options['alg_name'] = alg_name
				que = 0.0;	reg = 0.0;	time = 0.0
				
				if alg_name in ['AEWAF','AGF','RAEWAF','RAGF']:
					forecaster_options['delta'] = options[options['data_name']][alg_name][i]
				elif alg_name in ['REWAF','RGF']:							
					forecaster_options['delta']=options['deltas'][i]						
				elif alg_name =='REWAF1':
					forecaster_options['delta'] = _results_one_query['RAEWAF']['que'][0]
				elif alg_name == 'RGF1':
					forecaster_options['delta'] = _results_one_query['RAGF']['que'][0]														
				else:
					forecaster_options['delta'] = delta					
				
				
				if 'EWAF' in alg_name :
					que, reg, time = EWAF(XTest,YTest,forecaster_options)							
				else:
					que, reg, time = GF(XTest,YTest,forecaster_options)
				
				_results_one_query[alg_name]['que'].append(que)
				_results_one_query[alg_name]['reg'].append(reg)
				_results_one_query[alg_name]['time'].append(time)			
		
				# print alg_name, que

		for alg_name in options['algorithm_names']:
			# print alg_name, round(np.mean(_results_one_query[alg_name]['que'])*100,3)			
			if alg_name in ['GF','EWAF']	and i > 0:
				results_all_query[alg_name]['que'].append(results_all_query[alg_name]['que'][0])
				results_all_query[alg_name]['stdQue'].append(results_all_query[alg_name]['stdQue'][0])
				results_all_query[alg_name]['reg'].append(results_all_query[alg_name]['reg'][0])
				results_all_query[alg_name]['stdReg'].append(results_all_query[alg_name]['stdReg'][0])
				results_all_query[alg_name]['time'].append(results_all_query[alg_name]['time'][0])
				results_all_query[alg_name]['stdTime'].append(results_all_query[alg_name]['stdTime'][0])
			else:	
				results_all_query[alg_name]['que'].append(round(np.mean(_results_one_query[alg_name]['que']),6))
				results_all_query[alg_name]['stdQue'].append(round(np.std(_results_one_query[alg_name]['que']),6))
				results_all_query[alg_name]['reg'].append(round(np.mean(_results_one_query[alg_name]['reg'])*100,3))
				results_all_query[alg_name]['stdReg'].append(round(np.std(_results_one_query[alg_name]['reg'])*100,3))
				results_all_query[alg_name]['time'].append(round(np.mean(_results_one_query[alg_name]['time']),3))
				results_all_query[alg_name]['stdTime'].append(round(np.std(_results_one_query[alg_name]['time']),3))
	print '\n'	
	return results_all_query