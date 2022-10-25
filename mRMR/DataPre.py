import numpy as np

def discretize(data, dataset_name, data_pre):
	X = np.array(data)
	[n_sample, n_feature] = X.shape

	for ith_feat in range(n_feature):
		xi = X[:,ith_feat]
		mean = xi.mean()
		std = xi.std()

		for ith_sample in range(n_sample):
			if data_pre == 1:
				# if dataset_name == 'HDR' or dataset_name == 'SIX':
				X[ith_sample, ith_feat] = 1 if X[ith_sample, ith_feat] > mean \
					else -1
			else:
				# elif dataset_name == 'ARR':
				X[ith_sample, ith_feat] = 1 if X[ith_sample, ith_feat] > mean + std \
					else -1 if X[ith_sample, ith_feat] < mean - std \
					else 0




	return X


def DataPreprocessing(data, dataset_name, data_pre):

	# 我改成了所有数据都会进行数据预处理
	# if dataset_name == "HDR" or dataset_name == 'ARR' or dataset_name == 'SIX':
	data = discretize(data, dataset_name, data_pre)
	return data
