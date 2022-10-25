import os
import sys
import time
import numpy as np
import pandas as pd
from sklearn import metrics
from LoadData import load
from DataPre import DataPreprocessing
from MICriterion import Mutual_Info, mRMR_sel, MaxRel_sel
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.model_selection import LeaveOneOut

if __name__ == "__main__":

	clf_name = sys.argv[1]
	if not (clf_name == 'NB' or clf_name == 'SVM' or clf_name == 'LDA'):
		print('first argument is classfier name')
		exit()

	dataset = sys.argv[2]

	algthm_name = sys.argv[3]
	if not (algthm_name == 'mRMR' or algthm_name == 'MaxRel'):
		print('third argument is selection of algorithm')
		exit()

	max_feanum = sys.argv[4]

	data_method = sys.argv[5]

	# CV or LOOCV
	Cv_Loo = sys.argv[6]

	MAX_FEANUM = int(max_feanum)

	dir_path = './Dataset/' + dataset + '/'

	datafile = dir_path + dataset + '.csv'

	if clf_name == 'NB':
		clf = BernoulliNB()
	elif clf_name == 'SVM':
		clf = SVC(kernel='linear', C=1)
	elif clf_name == 'LDA':
		clf = LDA()
	else:
		raise Exception('Incorrect setting of classifer: {}'.format(clf_name))


	X, y = load(datafile, is_shuffle=False if dataset == 'ARR' else True)
	data_pre = int (data_method)
	X = DataPreprocessing(X, dataset, data_pre)
	n_sample = X.shape[0]

	# Run mRMR algorithm
	error_mean = []
	feat_ind = []
	costtime = []
	num_feat = X.shape[1]
	rel_array = np.zeros(num_feat)
	red_array = np.zeros(num_feat)
	for ith_feat in range(num_feat):
		xi = X[:,ith_feat]
		rel_array[ith_feat] = (Mutual_Info(xi, y))

	for i in range(MAX_FEANUM):
		scores = 0
		t0 = time.clock()
		if i == 0:
			print("Select 1st features from X")
			feat_ind.append(np.argsort(rel_array)[-1])
		else:
			print("Select %d features from X" % (i+1))
			if algthm_name == 'mRMR':
				feat_ind = mRMR_sel(X, feat_ind, rel_array, red_array)
			elif algthm_name == 'MaxRel':
				feat_ind = MaxRel_sel(X, y, feat_ind, rel_array)

		t1 = time.clock()-t0
		costtime.append(t1)
		print(feat_ind)
		print(t1, 'seconds')
		feat_sel_X = X[:, feat_ind]

		if Cv_Loo == 'CV':
			scores = cross_val_score(clf, feat_sel_X, y, cv=10)
			scores = 1 - scores
			error_mean.append(scores.mean())
		else:
			loo = LeaveOneOut()
			scores = 0
			for train, test in loo.split(feat_sel_X):
				ith_test = feat_sel_X[test,:]
				ith_train = feat_sel_X[train,:]
				ith_predict = y[test]
				ith_label = np.delete(y,test)
				clf.fit(ith_train,ith_label)
				scores += clf.score(ith_test,ith_predict)
			error_mean.append(1-scores/n_sample)

		print("error mean %f" % error_mean[i])

	feat_ind.append(0)
	print("select features are：", feat_ind)
	feat_ind.sort()
	print("After sorted, selected features are：", feat_ind)
	df = pd.read_csv(datafile, skipinitialspace=True, usecols=feat_ind)
	print(df.shape)
	df = df.values
	df = np.insert(df, 0, feat_ind, axis=0)
	np.savetxt("mrmrdata.csv", df, delimiter=",")
