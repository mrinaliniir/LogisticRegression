import numpy as np 
import pandas as pd 
import math
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model.logistic import _logistic_loss
from sklearn.model_selection import train_test_split
from math import sqrt
from sklearn.metrics import mean_squared_error
#from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score 

# load the dataset
df = pd.read_csv("C:/Users/irmri/OneDrive/Desktop/coursera/machine-learning-ex2/ex2/ex2data1.txt")
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

def normalizeFeatures(mean, sd, features):
		features['x1'] = (features.x1 - mean.x1)/sd.x1
		features['x2'] = (features.x2 - mean.x2)/sd.x2
		return features

x = normalizeFeatures(mean, sd, X)

def sigmoid(z):
	return 1/ (1 + np.exp(-z))

def costFunction(initial_theta, X , y):
	m=len(y)
	predictions = sigmoid(np.dot(X,initial_theta))
	error = (y * np.log(predictions)) + ((1-y)*np.log(1-predictions))
	cost = -1/m * sum(error)
	return cost

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

# Calculate the mean squared error
def rmse(X_train, y_train, theta):
    predicted_value = predictModel(theta, X_train)
    actual_value = y_train
    mean_error = np.sqrt(np.mean((predicted_value - actual_value)**2))
    return mean_error

print("Training model .....")
weights = trainModel(X_train, y_train)
print("********Results for logistic regression using current model**********")
RMSE_train = rmse(X_train, y_train, weights)
print("RMSE train :", RMSE_train)
RMSE_test = rmse(X_test, y_test, weights)
print("RMSE test :", RMSE_test) 
print(weights)
print("******************")


model = LogisticRegression()
model.fit(X_train,y_train)
print("******** Results for logistic regression using scikit learn library **********")
y_pred = model.predict(X_test)
RMSE2 = sqrt(mean_squared_error(y_test, y_pred))
print("RMSE :", RMSE2)
print(model.coef_)
print(model.intercept_)
print("******************")
	



