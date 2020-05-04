ReadMe for 4641 Final Project:

Software Depdencies:
Install Python and Anaconda Packages
Using time and itertools library from python standard libraries
From Anaconda 3 need sklearn, numpy, and matplotlib
Sklearn Librearies used include:
sklearn.svm.SVC
sklearn.neural_network.MLPClassifier
sklearn.ensemble.RandomForestClassifier
sklearn.ensemble.AdaBoostClassifier
sklearn.metrics.confusion_matrix,
sklearn.metrics.recall_score
sklearn.metrics.make_scorer
sklearn.model_selection.KFold
skelearn.model_slectionGridSearchCV
sklearn.tree.DecisionTreeClassifier

Raw Data file: heart.csv

To run program type the following command:
python proj.py

How to recreate graphs and figures (description of program file):
Alter the main method to recreate experiements:

load_heart() method splits the data into training and testing data sets

Hyperparmeters for Tuning experiments:
get_fold() splits the training data into folds that can be used for cross validation (default k=5)

tune_rf() perfroms the experiments on the random forests with bagging algorithm
method takes in the folds and an integer for max estimators
method creates random forests for each fold in the range of estimators between 1 and the value passed in
at each value for n_estimators, the method calculates the average accuraccy and recall.
a plot of mean validation accuracy and mean validation recall with respect to the number of estimators with standard deviation error bars is shown
the criterion field of the random forest must be set manually to either gini or entropy


tune_ada_boost() perfroms the experiments on the ada-boost classifier
method takes in the folds and an integer for max estimators
method creates ada-boost classifiers for each fold in the range of estimators between 1 and the value passed in
at each value for n_estimators, the method calculates the average accuraccy and recall.
a plot of mean validation accuracy and mean validation recall with respect to the number of estimators
the base_estimator field of the classifier must be set manually to a base estimator of choice (displayed figures use a DecisionTreeClassifier with wax depth of either 1,3, or 5)


tune_svm() perfroms the experiments on the support vector classifier
method takes in the folds and an integer for max value of the slack parameter
method creates SVC for each fold in the range of C between 1 and the value passed in
at each value for C, the method calculates the average accuraccy and recall.
a plot of mean validation accuracy and mean validation recall with respect to the the value of C
the kernel field of the support vector classifier must be set manually to either rbf or poly (degree used were 3 and 6)


tune_nn() perfroms the experiments on neural nets
method takes in the folds and bounds for the number of nodes in the first and second hidden layers
method creates neural nets for each fold for each combination of hidden layer sizes 
since the search space is much larger for neural nets compared to other algorithms, sklearn's GridSearchCV library is used
This library performs corss validation with 5-folds and provides arrays of mean accuracy and mean recall
a plot of mean validation accuracy and mean validation recall with respect to the number of estimators
the horizontal axis value of the plot represents (i_ub - i_lb)(i - i_lb) + (j - j_lb) where i is number of perceptrons in the first layer and j is the number of perceptrons in the second layer. 
if a wide span of layer combinations are tested, there is a bit of helper code at the end of the main method which can identify the index of the best hyperparameters

All tuning algorithms print the running time of the experiment to the console.

Next section of the main method cretes the classifiers with the best hyper parmeters.
These classifers are tested on the testing data and a confusion matrix is shown
Confusion matrix is displayed is a helper method called plot_confusion_matrix()
Note: the classifier for which the confusion matrix is displayed must be changed manually.
Amount of training time for the classifier of choice is outputted to the console

The final part of the main methods caluculates the confidence intervals:
It folds the traing data 30 times so that the central limit theorem can be applied.
It then performs cross validation on the test data to calcuate mean adn standard deviataion of accuracy and recall.
Note: the classifier for which the means and standard deviations are calculated must be changed manually.
