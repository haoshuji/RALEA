from sklearn import preprocessing
from sklearn.preprocessing import normalize

def NormalizeData(X,normalized_function):		
	if normalized_function == 'min_max_scale': # normalized to [0,1] for each feature
		print 'min_max_scale normalized'
		min_max_scaler = preprocessing.MinMaxScaler()
		X_minmax = min_max_scaler.fit_transform(X)
		X = X_minmax				
	elif normalized_function == 'l2':
		print 'l2 normalized'
		XNormalized = normalize(X,axis=1,norm='l2')
		X = XNormalized
	elif normalized_function == 'std_scale': # mean=0, std=1, for each feature
		print 'std_scale normalized'
		X_scaled = preprocessing.scale(X,axis=0)
		X = X_scaled
	else:
		print 'l0 normalized'
	return X