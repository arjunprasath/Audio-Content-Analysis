import h5py
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import pdb 
import cv2
from skimage.feature import hog
from skimage import data, color, exposure
from sklearn import svm
from skimage.transform import rotate
from skimage import transform as tf
from sklearn.decomposition import PCA	
from sklearn.decomposition import IncrementalPCA
import random
import cPickle
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import SGDClassifier
from random import randint
import scipy.io
from sklearn import preprocessing
#classifier = SGDClassifier()
ntrain = 50000
nhalf =  40000
itrain = 0

print 'DFT Difference PC= 100'

matname = 'X2.mat'
clasname = 'cover_dft2.pkl'
n_components = 100
mat = scipy.io.loadmat('myFile.mat')
songData = mat['songData'][:,1:17000]
labels = mat['songData'][:,0]

songScaled = preprocessing.scale(songData)

ipca = IncrementalPCA(n_components=n_components, batch_size=2000)
X_ipca = ipca.fit_transform(songScaled,labels)


def getTrainData(X_ipca,labels):
	i = 0
	X_pos = []
	y = []
	while(i<15500):
		label = labels[i]
		j =0
		temp = []
		while(labels[i] == label):
			temp.append(X_ipca[i])
			j=j+1
			i=i+1
		
		for x1 in range(len(temp)/2):
			for x2 in range(len(temp)):
				if x1 != x2:
					
					X_pos.append(np.concatenate((temp[x1],temp[x2]),axis=0))
	itrain = i
	print labels[i]				

	k = 0
	X_neg = []
	songRand = np.copy(X_ipca)
	np.random.shuffle(songRand)
	while(k < len(X_pos)):
		m = randint(0,17999)
		n = randint(0,17999)
		if labels[m] != labels[n]:
			X_neg.append(np.concatenate((X_ipca[m], X_ipca[n]),axis=0))
			
			k = k+1 
	
	dic = {'X_pos': X_pos, 'X_neg': X_neg, 'X_ipca':X_ipca, 'labels':labels}
	scipy.io.savemat(matname,dic)

def getTrainData2(X_ipca,labels):
	i = 0
	X_pos = []
	y = []
	while(i<15000):
		label = labels[i]
		j =0
		temp = []
		while(labels[i] == label):
			temp.append(X_ipca[i])
			j=j+1
			i=i+1
		
		for x1 in range(len(temp)/2):
			for x2 in range(len(temp)):
				if x1 != x2:
					
					X_pos.append(abs(temp[x1]-temp[x2]))
	itrain = i
	print labels[i]				

	k = 0
	X_neg = []
	songRand = np.copy(X_ipca)
	np.random.shuffle(songRand)
	while(k < len(X_pos)):
		m = randint(0,15000)
		n = randint(0,15000)
		if labels[m] != labels[n]:
			X_neg.append(abs(X_ipca[m] - X_ipca[n]))
			
			k = k+1 
	
	dic = {'X_pos': X_pos, 'X_neg': X_neg, 'X_ipca':X_ipca, 'labels':labels}
	scipy.io.savemat(matname,dic)
		



#getTrainData(X_ipca,labels)



#ipca = IncrementalPCA(n_components=200, batch_size=1000)







def train_classifier():
	

	global ntrain
	global classifier
	global n_components
	all_classes = np.array([0, 1])
	batch_size = 10000
	mat2 = scipy.io.loadmat(matname)
	
	#for batch in range(0, ntrain, batch_size):
	for batch in range(0, 1):
		
		#epochp =  mat2['X_pos'][batch:batch+batch_size,:]
		#epochn =  mat2['X_neg'][batch:batch+batch_size,:]
		epochp = mat2['X_pos']
		epochn = mat2['X_neg']

		train_data = np.append(epochp,epochn,0)

		
		print train_data.shape
		labels = np.ones(len(epochp)).reshape(-1,1)
		labels = np.append(labels, np.zeros(len(epochn)).reshape(-1,1))
		
		trainshuff = np.zeros([train_data.shape[0],n_components+1])

		trainshuff[:,0] = labels
		trainshuff[:,1:] = train_data
		np.random.shuffle(trainshuff)

		train_data = trainshuff[:,1:]
		labels = trainshuff[:,0]
		
		#pdb.set_trace()
		print labels.shape
		#Train the Classifier
		classifier = svm.SVC()
		classifier.fit(train_data,labels)
		

		#classifier.fit(train_data,labels)



	with open(clasname, 'wb') as fid:
	    cPickle.dump(classifier, fid)


	# test_pos = mat2['X_pos'][:1000]
	# test_neg = mat2['X_neg'][:1000]
	# testset = np.append(test_pos,test_neg, 0)
	

	# exp_labels = np.append((np.ones(len(test_pos)).reshape(-1,1)).T, (np.zeros(len(test_neg)).reshape(-1,1)).T,1)
	
	# print classifier.score(testset,exp_labels.T)
	# pred_labels = classifier.predict(np.array(testset)).T.astype(np.float32)
	# mask = pred_labels==exp_labels
	# correct = np.count_nonzero(mask)
	# #pdb.set_trace()
	
	# print correct*100.0/pred_labels.size





def test_classifier3():
	#global labels
	ntests = 500
	with open(clasname, 'rb') as fid:
		clf_loaded = cPickle.load(fid)
	mat2 = scipy.io.loadmat(matname)
	labels = mat2['labels'].T
	avg = 0
	nlabels = 0
	i = 16200
	nqueries = 0
	
	while(nqueries < 500):
		label = labels[i]
		j =0
		temp = []

		
		while(labels[i] == label):
			temp.append(mat2['X_ipca'][i])
			j=j+1
			i=i+1
		for x1 in range(len(temp)):
			for x2 in range(1):
				test_pos =[]
				test_neg =[]
				query = []
				if x1 != x2 and len(temp) >= 2:
					test_pos.append(abs((temp[x1] - temp[x2])))
					#test_pos.append(np.concatenate((temp[x1], temp[x2]),axis=0))

					#print i
					test_neg.append(abs((temp[x1] - mat2['X_ipca'][randint(0,10000)])))
					#test_neg.append(np.concatenate((temp[x1], mat2['X_ipca'][randint(0,1000)]), axis=0))


					query = np.append(np.array(test_pos),np.array(test_neg),0)
					exp_labels = np.array([1,0])
					pred_labels = clf_loaded.predict(np.array(query))

					mask = pred_labels==exp_labels
					correct = np.count_nonzero(mask)
					#pdb.set_trace()
					nqueries += 1
					#print 'label=' +str(label)
					#print correct*100.0/pred_labels.size
					avg+=correct*100.0/pred_labels.size
			

		

		



	print avg/500

def test_classifier():
	#global labels
	ntests = 2000
	with open('cover3.pkl', 'rb') as fid:
		clf_loaded = cPickle.load(fid)
	mat2 = scipy.io.loadmat('X5.mat')
	labels = mat2['labels'].T
	avg = 0
	nlabels = 0
	i = 15500
	
	
	while(i < 15500+ntests):
		label = labels[i]
		j =0
		temp = []
		test_pos =[]
		test_neg =[]
		
		while(labels[i] == label):
			temp.append(mat2['X_ipca'][i])
			j=j+1
			i=i+1
		for x1 in range(len(temp)):
			for x2 in range(len(temp)):
				if x1 != x2:
					test_pos.append(abs((temp[x1] - temp[x2])))
					test_neg.append(abs((temp[x1] - mat2['X_ipca'][randint(0,15500)])))

		testset = np.append(np.array(test_pos),np.array(test_neg),0)
		
		#pdb.set_trace()
		if len(test_pos) > 0:
			exp_labels = (np.ones(len(test_pos)).reshape(-1,1)).T
			exp_labels = np.append((np.ones(len(test_pos)).reshape(-1,1)).T, (np.zeros(len(test_neg)).reshape(-1,1)).T,1)
			pred_labels = clf_loaded.predict(np.array(testset)).T.astype(np.float32)
			mask = pred_labels==exp_labels
			correct = np.count_nonzero(mask)
			#pdb.set_trace()
			#print 'label=' +str(label)
			#print correct*100.0/pred_labels.size
			avg+=correct*100.0/pred_labels.size
			nlabels = nlabels+1


	print avg/nlabels

if __name__=="__main__":
	getTrainData2(X_ipca,labels)
	train_classifier()
	test_classifier3()
	# flip_id = np.fliplr(file['id'][0])
	# flip_cam= np.fliplr(file['image'][0])
	# print genFeatures(flip_id,flip_cam).shape
	# id_img = file['id'][1]
	# cam_img = file['image'][0]
	# print genFeatures(id_img,cam_img).shape
	# flip_id = flip_id.astype(np.uint8)
	# T,R,F = get_random_transform()
	# id_imgT = apply_transformation(id_img,T,R,F).astype(np.uint8)
	# id_img = id_img.astype(np.uint8)
	#cv2.imshow('Transformed',id_imgT)
	#cv2.imshow('normal',id_img)
	#cv2.waitKey(0)
	cv2.destroyAllWindows()
	#pdb.set_trace()









