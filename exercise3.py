import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Task 1
data = pd.read_csv("banknote_authentication.csv", header=None).values
x = data[:,:4]
y = data[:,-1]

np.random.shuffle(data)
dataLength = len(data) #Length of the data in this case 1372
trainingSetSize = round(int(0.8*dataLength)) #80% of the length which in this case is 1097
testSetSize = dataLength - trainingSetSize #Remaining 20%  which is 275
index = np.arange(dataLength) #We arrange the indices for the data
np.random.shuffle(index) #We shuffle the indices randomly

#Task 2
xTrain = x[index[:trainingSetSize]] 
yTrain = y[index[:trainingSetSize]] 
xTest = x[index[:testSetSize]] 
yTest = y[index[:testSetSize]] 

#Task 3 and 4
xTrainNormalized = (xTrain - np.mean(xTrain)) / np.std(xTrain)
xTestNormalized = (xTest - np.mean(xTest)) / np.std(xTest)

def sigmoid(x): 
  return 1/(1+np.exp(-x))

def cost(x, y, theta): #Calculate cost using sigmoid
    lengthY = len(y)
    pred = sigmoid(np.dot(x, theta))
    cost = (-1/lengthY) * np.sum(y*np.log(pred) + (1-y)*np.log(1-pred))
    return cost

def gradientDescent(x, y, learningRate, iterations): #Similar gradient from first exercise, only this one uses sigmoid
    yLength = len(y)
    costs = []
    theta = np.zeros(xTrainNormalized.shape[1])

    for i in range(iterations):
        h = sigmoid(np.dot(x, theta))
        grad = (1/yLength) * np.dot(x.T, h-y)
        theta -= learningRate * grad
        costs.append(cost(x, y, theta))
    return theta, costs

def trainAndPlot():
  iterations = 1000
  learningRate = 0.1
  theta, costs = gradientDescent(xTrainNormalized, yTrain, learningRate, iterations)

  predictions1 = np.round(sigmoid(np.dot(xTrainNormalized, theta))) #We use sigmoid to help get predictions
  trainErrors = np.sum(predictions1 != yTrain) #We take the sum where predicted values are not the same as actual values
  trainAccuracy = (1 - (trainErrors/len(yTrain))) * 100 #We get the accuracy in percentage

  predictions2 = np.round(sigmoid(np.dot(xTestNormalized, theta))) 
  testErrors = np.sum(predictions2 != yTest)
  testAccuracy = (1 - (testErrors/len(yTest))) * 100

  print(f"Iterations: {iterations}, LearningRate: {learningRate}")
  print(f"Training errors: {trainErrors}, Accuracy: {trainAccuracy:.2f}%")
  print(f"Test errors: {testErrors}, Test Accuracy: {testAccuracy:.2f}%")
  plt.plot(costs)
  plt.xlabel("Iterations")
  plt.ylabel("Cost")

trainAndPlot()  
plt.show()

