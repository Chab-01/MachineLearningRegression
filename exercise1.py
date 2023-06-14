import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np

data = pd.read_csv("GPUbenchmark.csv", header=None).values
X = data[:,:6] #Select first six columns
Y = data[:, -1] #Select all values from last columns, that is why -1 is used
Xmin = X.min(axis=0) #Find smallest value in each column
Xmax = X.max(axis=0) #Find largest value in each column

#Task 1
Xn = (X - np.mean(X, axis=0)) / np.std(X, axis=0) #Normalizing values of X into variable Xn

#Task 2
def plotXiWithY():#Plots the all the normalized columns of X with the Y values, creating 6 plots in total
  plt.figure(1)
  for i in range(len(X[0])):
    plt.subplot(231+i)
    plt.scatter(Xn[:,i], Y, s=5) # Selects all values from the i-th column and plots it with the Y values
    plt.xlabel("Normalized X values")
    plt.ylabel("Y values")

#Task 3
Bval = [2432, 1607, 1683, 8, 8, 256] #Given in the assignment
BvalNormalized = (Bval - np.mean(X, axis=0)) / np.std(X, axis=0) #Normalizing Bval usin X
BvalExtended = np.hstack((np.ones((1, 1)), BvalNormalized.reshape(1, -1))).flatten() #Extends BvalNormalized so we can use this for prediction
Xe = np.hstack((np.ones((len(Xn), 1)), Xn)) #Extended matrix of Xn
XeT = np.transpose(Xe) 
predParams = np.dot(np.dot(np.linalg.inv(np.dot(XeT, Xe)), XeT), Y) #Holds prediction parameter values to get a prediction
predBval = np.dot(BvalExtended, predParams) #Multiply pred with extended Bval to get prediction
XePred = np.dot(Xe, predParams) #Get predicted Y values by multiplying Xe with the prediction parameters to get predicted Y values
                                #That we later compare with the real Y values to get the cost

#Task 4
def cost(Xe, y, predParams):
  cost = np.dot(Xe, predParams) - y
  return (cost.T.dot(cost)) / Xe.shape[0]

print(f"Predicted value: {predBval}, Cost = {cost(Xe, Y, predParams)}")
normal_eq_cost = cost(Xe, Y, predParams)

#Task 5
def gradientDescent(X, Y, N, learningRate):
  m = len(Y)
  costs = []
  theta = np.zeros(X.shape[1])

  for i in range(N):
    pred = np.dot(X, theta)
    theta = theta -(1/m)*learningRate*(X.T.dot((pred - Y)))
    costs.append([i,  cost(X, Y, theta)])

  return theta, costs

def plotGradAndPredictedBenchMark():
  plt.figure(2)
  theta, costs = gradientDescent(Xe, Y, 1000, 0.05) #We find 1000 iterations, and learning rate = 0.05, to fall within 1%
  costs = np.asanyarray(costs)
  prediction = np.dot(theta, BvalExtended)
  plt.scatter(costs[:, 0], costs[:, 1], s=5)
  plt.xlabel("Iterations")
  plt.ylabel("Cost")
  print(f"Predicted value after gradient: {prediction}")

plotXiWithY()
plotGradAndPredictedBenchMark()
plt.show()