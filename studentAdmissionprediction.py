import numpy as np 
import pandas as pd 
import math
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model.logistic import _logistic_loss
from sklearn.model_selection import train_test_split
from math import sqrt
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc


# load the dataset
df = pd.read_csv("C:/Users/irmri/OneDrive/Documents/academic_projects/coursera/StudentAdmissionPrediction-master/ex2data1.txt")
df.columns=['x1','x2','y']

# visulaizing the dataset
markers = ['o', '+']
labels = ["Admitted", "Not Admitted"]
for k, m in enumerate(markers):
    i = (df.y == k)
    plt.scatter(df.x1[i], df.x2[i], marker=m, label=labels[k])
    plt.legend(loc=0)
plt.show()

alpha = 0.01
iterations = 10000
Lamda = 1
X = df[['x1','x2']]
[m,n] = X.shape
initial_theta = np.array([-1,2,3], dtype ="float")
X.insert(loc=0, column='ones', value=np.ones(len(X)))
mean = X.mean(axis=0)
sd = X.std(axis=0)
J_history =[]

# feature standardization
def normalizeFeatures(mean, sd, features):
		features['x1'] = (features.x1 - mean.x1)/sd.x1
		features['x2'] = (features.x2 - mean.x2)/sd.x2
		return features

x = normalizeFeatures(mean, sd, X)

# sigmoid function
def sigmoid(z):
	return 1/ (1 + np.exp(-z))

#loss function 
def costFunction(initial_theta, X , y):
	m=len(y)
	predictions = sigmoid(np.dot(X,initial_theta))
	error = (y * np.log(predictions)) + ((1-y)*np.log(1-predictions))
	cost = -1/m * sum(error)
	return cost

#gradient descent
def gradientDescent(alpha, iterations, theta, X, y):
	for i in range(iterations):
		z = X.dot(theta.T)
		h_theta = 1 / (1 + np.exp(-z))
		for index in range(len(theta)):
			theta[index] = theta[index] - (alpha/m) * ((h_theta - y).dot(X[X.columns[index]]) + Lamda*(theta[index]))
			cost = costFunction(theta, X, y)
			J_history.append(cost)
	return theta

# Split the dataset into training and test data
X_train, X_test, y_train, y_test = train_test_split(x, df.y, test_size=0.2)

def trainModel(X_train, y_train):
    optimized_theta = gradientDescent(alpha,iterations, initial_theta, X_train, y_train)
    plt.plot(J_history)
    plt.xlabel("Iteration")
    plt.ylabel("$J(\Theta)$")
    plt.title("Cost function using Gradient Descent")
    plt.show()
    return optimized_theta

def predictModel(theta, X):
	predictions = sigmoid(np.dot(X,theta))  # Get predictions
	return predictions


print("Training model .....")
weights = trainModel(X_train, y_train)
y_pred_LR = predictModel(weights,X_test).round()
print("********Results for logistic regression using current model**********")
CM_LR = confusion_matrix(y_test,y_pred_LR)
print("Confusion Matrix")
print(CM_LR)
accuracy_LR = accuracy_score(y_test,y_pred_LR)
print("Accuracy of predictions :",accuracy_LR)
precision_LR = precision_score(y_test,y_pred_LR)
print("Precison of predictions :",precision_LR)
recall_LR = recall_score(y_test,y_pred_LR)
print("Recall :",recall_LR)

fpr,tpr,thresholds = roc_curve(y_test,y_pred_LR)
plt.plot(fpr,tpr,label= "Logistic Regression")
plt.xlabel("False Positive Rate(FPR)")
plt.ylabel("True Positive Rate(TPR)")
plt.legend()
plt.show()
print("AUC for Logistic Regression ",auc(fpr,tpr))
print("*********************************************************************")




model = LogisticRegression()
model.fit(X_train,y_train)
print("******** Results for logistic regression using scikit learn library **********")
y_pred = model.predict(X_test)
CM = confusion_matrix(y_test,y_pred)
print("Confusion Matrix")
print(CM)
accuracy = accuracy_score(y_test,y_pred)
print("Accuracy of predictions :",accuracy)
precision = precision_score(y_test,y_pred)
print("Precison of predictions :",precision)
recall = recall_score(y_test,y_pred)
print("Recall :",recall)
print("*********************************************************************")
	



