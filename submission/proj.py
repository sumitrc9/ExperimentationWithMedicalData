import matplotlib.pyplot, os.path
import itertools
import time
import numpy as np
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import confusion_matrix, recall_score, make_scorer
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.tree import DecisionTreeClassifier

#Age, Sex, Cp, Trestbps, Chol, Fbs, Restecg, Thalach, Exang, Oldpeak, Slope, Ca, Thal, Target 


def plot_confusion_matrix(cm, classes,
                          title='Confusion matrix',
                          cmap=matplotlib.pyplot.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    
    matplotlib.pyplot.imshow(cm, interpolation='nearest', cmap=cmap)
    matplotlib.pyplot.title(title)
    matplotlib.pyplot.colorbar()
    tick_marks = np.arange(len(classes))
    matplotlib.pyplot.xticks(tick_marks, classes, rotation=45)
    matplotlib.pyplot.yticks(tick_marks, classes)

    fmt = 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        matplotlib.pyplot.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    matplotlib.pyplot.ylabel('True label')
    matplotlib.pyplot.xlabel('Predicted label')
    matplotlib.pyplot.tight_layout()



def get_test_train(fname,seed,datatype):
	'''
	Returns a test/train split of the data in fname shuffled with
	the given seed


	Args:
		fname: 		A str/file object that points to the CSV file to load, passed to 
					numpy.genfromtxt()
		seed:		The seed passed to numpy.random.seed(). Typically an int or long
		datatype:	The datatype to pass to genfromtxt(), usually int, float, or str


	Returns:
		train_X:	A NxD numpy array of training data (row-vectors), 75% of all data
		train_Y:	A Nx1 numpy array of class labels for the training data
		test_X:		A MxD numpy array of testing data, same format as train_X, 25% of all data
		test_Y:		A Mx1 numpy array of class labels for the testing data
	'''
	data = np.genfromtxt(fname,delimiter=',',dtype=datatype)
	np.random.seed(seed)
	shuffled_idx = np.random.permutation(data.shape[0])
	cutoff = int(data.shape[0]*0.75)
	train_data = data[shuffled_idx[:cutoff]]
	test_data = data[shuffled_idx[cutoff:]]
	train_X = train_data[:,:-1].astype(float)
	train_Y = train_data[:,-1].reshape(-1,1)
	test_X = test_data[:,:-1].astype(float)
	test_Y = test_data[:,-1].reshape(-1,1)
	return train_X, train_Y, test_X, test_Y


def load_heart():
	return get_test_train('heart.csv',seed=1567708903,datatype=float)


def get_folds(X_train, y_train, folds=5):
	kf = KFold(n_splits = folds, shuffle = True, random_state = 10)

	X_train_folds = []
	y_train_folds = []
	X_test_folds = []
	y_test_folds = []
	for train_index, test_index in kf.split(X_train):
	    X_train_folds.append(X_train[train_index])
	    X_test_folds.append(X_train[test_index])
	    y_train_folds.append(y_train[train_index])
	    y_test_folds.append(y_train[test_index])

	return X_train_folds, y_train_folds, X_test_folds, y_test_folds

def tune_rf(ub, x1, y1, x2, y2):
	#Random Forest
	#Baggging
	#Tune n_estimators, representing the number of bags or individual decision trees (passed in as ub)
	#criterion can either be (gini, entropy); 

	mean_acc = []
	std_acc = []
	mean_recall = []
	std_recall = []
	start = time.time()

	for i in range(1, ub):
		acc = []
		recall = []
		for fold in range(len(x1)):
			#set criterion to either gini or entropy
			clf = RandomForestClassifier(n_estimators=i, criterion='gini')
			clf.fit(x1[fold], y1[fold].ravel())
			acc.append(clf.score(x2[fold], y2[fold]))
			recall.append(recall_score(y2[fold], clf.predict(x2[fold])))
		mean_acc.append(np.mean(acc))
		mean_recall.append(np.mean(recall))
		std_acc.append(np.std(acc))
		std_recall.append(np.std(recall))
		print("n_estimatiors = ", i)

	end = time.time()
	print("Execution Time", end - start)

	fig1 = matplotlib.pyplot.figure()
	fig1.gca().set_title("5-Fold Cross Validation Hyperparmater Tuning Random Forest with Bagging using Entropy")
	fig1.gca().set_ylabel("Accuracy")
	fig1.gca().set_xlabel("n_estimators")
	fig1.gca().plot(np.arange(1,ub), mean_acc, color='b', label='Validation Accuracy')
	fig1.gca().errorbar(np.arange(1,ub), mean_acc, yerr=std_acc)
	fig1.gca().legend(loc='lower right')
	matplotlib.pyplot.show()

	fig2 = matplotlib.pyplot.figure()
	fig2.gca().set_title("5-Fold Cross Validation Hyperparmater Tuning Random Forest with Bagging using Entropy")
	fig2.gca().set_ylabel("Recall")
	fig2.gca().set_xlabel("n_estimators")
	fig2.gca().plot(np.arange(1,ub), mean_recall,color='b', label='Validation Recall')
	fig2.gca().errorbar(np.arange(1,ub), mean_recall, yerr=std_recall)
	fig2.gca().legend(loc='lower right')
	matplotlib.pyplot.show()

def tune_ada_boost(ub, x1, y1, x2, y2):
	# #Tune n_estimators, representing number of iterations for boosting (represente with ub)
	
	mean_acc = []
	std_acc = []
	mean_recall = []
	std_recall = []

	start = time.time()

	for i in range(1, ub):
		acc = []
		recall = []
		for fold in range(len(x1)):
			#set base estimator
			clf = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=5), n_estimators=i)
			clf.fit(x1[fold], y1[fold].ravel())
			acc.append(clf.score(x2[fold], y2[fold]))
			recall.append(recall_score(y2[fold], clf.predict(x2[fold])))
		mean_acc.append(np.mean(acc))
		mean_recall.append(np.mean(recall))
		std_acc.append(np.std(acc))
		std_recall.append(np.std(recall))
		print("n_estimatiors = ", i)

	end = time.time()

	fig1 = matplotlib.pyplot.figure()
	fig1.gca().set_title("5-Fold Cross Validation AdaBoost Classifier, Max Depth = 5")
	fig1.gca().set_ylabel("Accuracy")
	fig1.gca().set_xlabel("n_estimators")
	fig1.gca().plot(np.arange(1,ub), mean_acc, color='b', label='Validation Accuracy')
	fig1.gca().errorbar(np.arange(1,ub), mean_acc, yerr=std_acc)
	fig1.gca().legend(loc='lower right')
	matplotlib.pyplot.show()

	fig2 = matplotlib.pyplot.figure()
	fig2.gca().set_title("5-Fold Cross Validation AdaBoost Classifier, Max Depth = 5")
	fig2.gca().set_ylabel("Recall")
	fig2.gca().set_xlabel("n_estimators")
	fig2.gca().plot(np.arange(1,ub), mean_recall,color='b', label='Validation Recall')
	fig2.gca().errorbar(np.arange(1,ub), mean_recall, yerr=std_recall)
	fig2.gca().legend(loc='lower right')
	matplotlib.pyplot.show()
	
	print("Execution Time", end - start)

def tune_svm(c, x1, y1, x2, y2):
	#Tune type of Kernel (rbf, poly (adjust degree))
	#Tune Slack parameter C for SVMs (regularization) (passed in as c)
	
	mean_acc = []
	std_acc = []
	mean_recall = []
	std_recall = []

	start = time.time()

	for i in range(1, c):
		acc = []
		recall = []
		for fold in range(len(x1)):
			#set kernel function to 'rbf' or 'poly' w/ degree
			clf = SVC(C=i, kernel='rbf', degree=3)
			clf.fit(x1[fold], y1[fold].ravel())
			acc.append(clf.score(x2[fold], y2[fold]))
			recall.append(recall_score(y2[fold], clf.predict(x2[fold])))
		mean_acc.append(np.mean(acc))
		mean_recall.append(np.mean(recall))
		std_acc.append(np.std(acc))
		std_recall.append(np.std(recall))
		print("C = ", i)

	end = time.time()

	fig1 = matplotlib.pyplot.figure()
	fig1.gca().set_title("5-Fold Cross Validation SVM Classifier RBF Kernel")
	fig1.gca().set_ylabel("Accuracy")
	fig1.gca().set_xlabel("Slack Parameter (C)")
	fig1.gca().plot(np.arange(1,c), mean_acc, color='b', label='Validation Accuracy')
	#fig1.gca().errorbar(np.arange(1,c), mean_acc, yerr=std_acc)
	fig1.gca().legend(loc='lower right')
	matplotlib.pyplot.show()

	fig2 = matplotlib.pyplot.figure()
	fig2.gca().set_title("5-Fold Cross Validation SVM Classifier RBF Kernel")
	fig2.gca().set_ylabel("Recall")
	fig2.gca().set_xlabel("Slack Parameter (C)")
	fig2.gca().plot(np.arange(1,c), mean_recall,color='b', label='Validation Recall')
	#fig2.gca().errorbar(np.arange(1,c), mean_recall, yerr=std_recall)
	fig2.gca().legend(loc='lower right')
	matplotlib.pyplot.show()
	
	print("Execution Time", end - start)

def tune_nn(i_lb, i,ub, j_lb, j_ub, train_x, train_y):
	# #Hyper parmeters that will be tuned include the number of perceptrons in each of the hidden layers 

	start = time.time()

	mlp = MLPClassifier()
	parameters = {'hidden_layer_sizes':[(i,j) for i in range(i_lb,i_ub) for j in range(j_lb,j_ub)]}
	clf = GridSearchCV(estimator=mlp, param_grid=parameters)
	clf.fit(train_x, train_y.ravel())

	mean_acc = clf.cv_results_['mean_test_score']
	#np.savetxt('acc.out', mean_acc)
	
	params2 = clf.best_params_
	

	fig1 = matplotlib.pyplot.figure()
	fig1.gca().set_title("Impact of Hidden Layer Size on Accuracy")
	fig1.gca().set_ylabel("Accuracy")
	fig1.gca().set_xlabel("Perceptrons in Hidden Layer")
	fig1.gca().plot(np.arange(0,len(mean_acc)), mean_acc,color='r')
	
	matplotlib.pyplot.savefig('nn_accuracy.png')

	mlp = MLPClassifier()
	clf = GridSearchCV(estimator=mlp, param_grid=parameters, scoring=make_scorer(recall_score))
	clf.fit(train_x, train_y.ravel())
	
	mean_recall = clf.cv_results_['mean_test_score']
	#np.savetxt('recall.out', mean_recall)
	params = clf.best_params_
	
	print(params2)
	print(params)

	fig2 = matplotlib.pyplot.figure()
	fig2.gca().set_title("Impact of Hidden Layer Size on Accuracy")
	fig2.gca().set_ylabel("Accuracy")
	fig2.gca().set_xlabel("Perceptrons in Hidden Layer")
	fig2.gca().plot(np.arange(0,len(mean_acc)), mean_acc,color='r')

	matplotlib.pyplot.savefig('nn_recall.png')
	end = time.time()

	print("Time", end-start)
	#return mean_acc, mean_recall


def main():
	#load data set
	data = load_heart()
	fit_data = data[1].ravel()


	#TUNING PARMETERS
	#Alls using 5-fold cross validation

	#Get Validation Set
	x1, y1, x2, y2 = get_folds(data[0], data[1])

	#Random Forest
	#Baggging
	#tune_rf(31, x1, y1, x2, y2)
		
	# #Boosting
	#tune_ada_boost(31, x1, y1, x2, y2)

	# #Support Vector Classifer
	#tune_svm(301, x1, y1, x2, y2)
	

	# #Neural Network
	#acc_nn, recall_nn = tune_nn(10, 101, 10, 101, data[0], data[1])
	
	

	#TESTING

	#Classificatin with best hyperparmaters
	svc = SVC(C=260, kernel='rbf')
	nn = MLPClassifier(hidden_layer_sizes=(74,70))
	boost = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=5), n_estimators=9)
	rf = RandomForestClassifier(n_estimators=9, criterion='entropy')
	
	# change to classifier of choice
	start = time.time()
	svc.fit(data[0], data[1])
	end = time.time()
	pred = svc.predict(data[2])
	truth = data[3]
	# Compute confusion matrix
	cnf_matrix = confusion_matrix(truth, pred)
	class_names=['Disease', 'No Disease']
	np.set_printoptions(precision=2)

	# Plot confusion matrix
	matplotlib.pyplot.figure()
	plot_confusion_matrix(cnf_matrix, classes=class_names,
	                      title='Confusion Matrix Support Vector Classifier')
	matplotlib.pyplot.show()
	print("Training Time: ", end-start)

	#Confidence Intervals
	#Get 30 Folds to apply central limit theorm
	x1, y1, x2, y2 = get_folds(data[0], data[1], 30)

	acc = []
	recall = []
	

	for fold in range(len(x1)):
		#set clf to classifier with best hyperparmeters
		clf = SVC(C=260, kernel='rbf')
		clf.fit(x1[fold], y1[fold].ravel())
		acc.append(clf.score(data[2], data[3]))
		recall.append(recall_score(data[3], clf.predict(data[2])))
	
	print("Mean Acc: ", np.mean(acc))
	print("Mean Recall: ", np.mean(recall))
	print("Std Acc: ", np.std(acc))
	print("Std Recall: ", np.std(recall))


	#Fiding best neural net params)

	# acc_max = max(acc_nn)
	# recall_max = max(recall_nn)

	# diffs = []
	
	# for i in range(len(acc)):
	# 	acc_nn[i] = acc_max - acc_nn[i]
	# 	recall_nn[i] = recall_max - recall_nn[i]
	# 	diffs.append(acc_nn[i] + recall_nn[i])

	# print(diffs.index(min(diffs)))



if __name__ == '__main__':
	main()