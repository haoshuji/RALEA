import numpy as np
import random
from math import *
import collections
import pickle
import os, pickle, yaml
from numpy import random, mean, std, sqrt
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pylab import figure
from sklearn.preprocessing import normalize
from PlotVariedQuery import *
def SetSelectedIndexes(data_name,options):
	num_queries = options['num_queries']
	if data_name == 'a8a':
		if 'noise' in options['file_middle_name']:
			selected_indexes['AEWAF'] = range(0,18)+range(18,num_queries-1,8)
			selected_indexes['RAEWAF'] = range(0,21,7) + range(21,num_queries)
			selected_indexes['REWAF'] = range(1,num_queries-1,2)
			selected_indexes['AGF'] = range(0,18)#+range(18,num_queries-5,5)
			selected_indexes['RAGF'] = [0,1,2,4,6,7,9,11,15,17,23,26,29]
			selected_indexes['RGF'] = range(1,num_queries-1,2)
		else:
			selected_indexes['AEWAF'] = range(0,18)+range(18,num_queries-1,8)
			selected_indexes['RAEWAF'] = range(0,21,7) + range(21,num_queries)
			selected_indexes['REWAF'] = range(1,num_queries-1,2)
			selected_indexes['AGF'] = range(0,18)#+range(18,num_queries-5,5)
			selected_indexes['RAGF'] = range(0,21,2) + range(21,num_queries,2)
			selected_indexes['RGF'] = range(1,num_queries-1,2)
	elif data_name == 'magic04':
		if 'noise' in options['file_middle_name']:
			selected_indexes['AEWAF'] = range(0,10,2)+range(10,20,2)+[22,28]
			selected_indexes['RAEWAF'] = range(0,21,7) + range(21,num_queries)
			selected_indexes['REWAF'] = range(1,num_queries-2)
			selected_indexes['AGF'] = range(0,16)
			selected_indexes['RAGF'] = [0,1] + range(8,num_queries,2)
			selected_indexes['RGF'] = range(2,num_queries, 2)
		else:
			selected_indexes['AEWAF'] = range(0,10)+range(10,num_queries-1,5)
			selected_indexes['RAEWAF'] = range(0,21,7) + range(21,num_queries)
			selected_indexes['REWAF'] = range(1,num_queries-2)
			selected_indexes['AGF'] = range(0,20)
			selected_indexes['RAGF'] = range(0,4) + range(4,num_queries,3)
			selected_indexes['RGF'] = range(1,num_queries, 2)
	elif data_name == 'mushrooms':
		if 'noise' in options['file_middle_name']:
			selected_indexes['AEWAF'] = range(0,18,2)+range(18,num_queries-1)
			selected_indexes['RAEWAF'] = range(0,16,3) + range(16,num_queries-2,2)
			selected_indexes['REWAF'] = range(1,num_queries-1,2)
			selected_indexes['AGF'] = range(0,19)
			selected_indexes['RAGF'] = range(0,4) + range(4,num_queries-2,3)
			selected_indexes['RGF'] = range(1,num_queries-1, 2)
		else:
			selected_indexes['AEWAF'] = range(0,18,2)+range(18,num_queries-1,4)
			selected_indexes['RAEWAF'] = range(0,10) + range(10,num_queries-1,2)
			selected_indexes['REWAF'] = range(1,num_queries-1,2)
			selected_indexes['AGF'] = range(0,num_queries-2,2)
			selected_indexes['RAGF'] = range(0,4) + range(4,num_queries,3)
			selected_indexes['RGF'] = range(1,num_queries-1, 2)
	elif data_name == 'spambase':
		if 'noise' in options['file_middle_name']:
			selected_indexes['AEWAF'] = range(0,14)+[20,28]
			selected_indexes['RAEWAF'] = range(0,21,3) + range(21,num_queries)
			selected_indexes['REWAF'] = range(2,num_queries-2,2)
			selected_indexes['AGF'] = range(0,18)
			selected_indexes['RAGF'] = [0, 1, 3, 5, 7, 10, 13, 16, 19, 21, 22, 24, 26, 28]
			selected_indexes['RGF'] = range(1,num_queries-1, 2)
		else:
			selected_indexes['AEWAF'] = range(0,18,2)+range(18,num_queries-1,3)
			selected_indexes['RAEWAF'] = range(0,21,3) + range(21,num_queries)
			selected_indexes['REWAF'] = range(1,num_queries-2,2)
			selected_indexes['AGF'] = range(0,14)
			selected_indexes['RAGF'] = [0, 1, 3, 5, 7, 10, 13, 16, 19, 21, 22, 24, 26, 28]
			selected_indexes['RGF'] = range(1,num_queries-1, 2)
	elif data_name == 'svmguide1':
		if 'noise' in options['file_middle_name']:
			selected_indexes['AEWAF'] = range(1,num_queries-1)
			selected_indexes['RAEWAF'] = range(1,21,7) + range(21,num_queries)
			selected_indexes['REWAF'] = range(1,num_queries-2,2)
			selected_indexes['AGF'] = range(0,25)
			selected_indexes['RAGF'] = [0, 1, 3, 5, 7, 10, 13, 16, 19, 21, 22, 24, 26, 28]
			selected_indexes['RGF'] = range(1,num_queries-1, 2)
		else:
			selected_indexes['AEWAF'] = range(1,18)+range(18,num_queries-1,4)
			selected_indexes['RAEWAF'] = range(1,21,7) + range(21,num_queries)
			selected_indexes['REWAF'] = range(1,num_queries-2)
			selected_indexes['AGF'] = range(0,25)
			selected_indexes['RAGF'] = [0, 1, 3, 5, 7, 10, 13, 16, 19, 21, 22, 24, 26, 28]
			selected_indexes['RGF'] = range(1,num_queries-1, 2)
	elif data_name == 'w8a_pro':
		if 'noise' in options['file_middle_name']:
			selected_indexes['AEWAF'] = range(0,18)+range(18,num_queries-1)
			selected_indexes['RAEWAF'] = range(0,21,5) + range(21,26)
			selected_indexes['REWAF'] = range(2,29,2)   
			selected_indexes['AGF'] = range(0,18)+range(18,24)
			selected_indexes['RAGF'] = [0,1,10,11] + range(11,22,2)
			selected_indexes['RGF'] = range(3,num_queries-2)
		else:
			selected_indexes['AEWAF'] = range(0,18)+range(18,num_queries-1,4)
			selected_indexes['RAEWAF'] = range(0,21,7) + range(21,num_queries)
			selected_indexes['REWAF'] = range(1,num_queries-2)   
			selected_indexes['AGF'] = range(0,18)+range(18,num_queries-1,4)
			selected_indexes['RAGF'] = range(0,21,7) + range(21,num_queries)
			selected_indexes['RGF'] = range(1,num_queries-2)    
	else:
		selected_indexes['AEWAF'] = range(0,num_queries)
		selected_indexes['RAEWAF'] = range(0,num_queries)
		selected_indexes['REWAF'] = range(0,num_queries)
		selected_indexes['AGF'] = range(0,num_queries)
		selected_indexes['RAGF'] = range(0,num_queries)
		selected_indexes['RGF'] = range(0,num_queries)
	options['selected_indexes'] = selected_indexes
if __name__ == '__main__':
	options={}        
	input_dir = '../results/'
	output_dir = '../results/'
	
	options['output_file_extension'] = '.pdf'
	options['output_dir'] = output_dir
	options['file_middle_name'] = 'noise'
   	
   	options['num_noisy_experts_list'] = [0,2,5,7]

	normalize_functions={}
	normalize_functions['a8a'] = 'l0'
	normalize_functions['magic04'] = 'std_scale'
	normalize_functions['mushrooms'] = 'min_max_scale'
	normalize_functions['spambase'] = 'std_scale'
	normalize_functions['svmguide1'] = 'std_scale'
	normalize_functions['w8a_pro'] = 'l0'    

	#
	data_names = ['a8a','magic04','mushrooms','spambase','svmguide1','w8a_pro']
	for data_name in data_names:
		print data_name
		options['data_name'] = data_name
		for num_noisy_experts in options['num_noisy_experts_list']
		
			output_file_name = options['output_dir']+data_name+'_'+str(num_noisy_experts)+'_'+normalized_function[data_name]
			options['output_file_name'] = output_file_name
			
			print data_full_path
			result_que = pickle.load(open(output_file_name+'.txt','r'))  
			# print results_que              
			selected_indexes = {}
			options['num_queries'] = len(result_que['REWAF']['que']) 
			
			SetSelectedIndexes(data_name,options)
			options['normalized_function'] = normalize_functions[data_name]
			# options['output_file_name'] = 
			PlotVariedQuery(result_que,options)