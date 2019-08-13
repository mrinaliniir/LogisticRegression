import numpy as np 
import pandas as pd 
import math
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model.logistic import _logistic_loss

# load the dataset
df = pd.read_csv("C:/Users/irmri/OneDrive/Desktop/coursera/machine-learning-ex2/ex2/ex2data1.txt")
df.columns=['x1','x2','y']

# visulaizing the dataset
markers = ['o', '+']
for k, m in enumerate(markers):
    i = (df.y == k)
    plt.scatter(df.x1[i], df.x2[i], marker=m)

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
#print(x)
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
	error = (-y * np.log(predictions)) - ((1-y)*np.log(1-predictions))
	cost = 1/m * sum(error)
	return cost

def gradientDescent(alpha, iterations, theta, X, y):
	for i in range(iterations):
		z = X.dot(theta.T)
		h_theta = 1 / (1 + np.exp(-z))
		#print(h_theta)
		for index in range(len(theta)):
			theta[index] = theta[index] - (alpha/m) * ((h_theta - y).dot(X[X.columns[index]]) + Lamda*(theta[index]))
			cost = costFunction(theta, X, df.y)
			J_history.append(cost)
	return theta

theta = gradientDescent(alpha,iterations, initial_theta, x, df.y)
print(theta)
plt.plot(J_history)
plt.xlabel("Iteration")
plt.ylabel("$J(\Theta)$")
plt.title("Cost function using Gradient Descent")
plt.show()



#############Sckitlearn ###################
model = LogisticRegression()
model.fit(x,df.y )
print(model.coef_)
print(model.intercept_)
