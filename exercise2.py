import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv("secret_polynomial.csv", header=None).values
x = data[:,0] 
y = data[:, -1] 
degrees = [1,2,3,4,5,6]

Xn = (x - np.mean(x, axis=0)) / np.std(x, axis=0)
Yn = (y - np.mean(y, axis=0)) / np.std(y, axis=0)

dataLength = len(data) #Length of the data in this case 300
trainingSetSize = int(0.8*dataLength) #80% of the length which in this case is 240
testSetSize = dataLength - trainingSetSize #Remaining 20%  which is 60
index = np.arange(dataLength) #We arrange the indices for the data, 0-299

for i in range(10): #We run simulation 10 times
  np.random.shuffle(index) #We shuffle the indices randomly
  xTrain = x[index[:trainingSetSize]] #We fetch 240 x values with the random indices
  yTrain = y[index[:trainingSetSize]] #We fetch 240 y values with the random indices
  xTest = x[index[:testSetSize]] #We fetch 60 x values with the random indices
  yTest = y[index[:testSetSize]] #We fetch 60 y values with the random indices

  for j, d in enumerate(degrees): 
    p = np.polyfit(xTrain, yTrain, d) #We fit the training data
    yPredTrain = np.polyval(p, xTrain) #We get pred values to use in trainMSE calculation
    yPredTest = np.polyval(p, xTest) #We get pred values to use in testMSE calculation

    trainMSE = np.mean((yTrain - yPredTrain)**2) #Calculate trainMSE
    testMSE = np.mean((yTest - yPredTest)**2) #Calculate testMSE

    xp = np.linspace(np.min(x), np.max(x), 100) #We create the polynomial regression line
    yp = np.polyval(p, xp)

    plt.subplot(231+j)
    plt.scatter(xTrain, yTrain, c="green",s=15)
    plt.scatter(xTest, yTest, c="red",s=15)
    plt.plot(xp, yp, c="blue", label="Regression line") 
    plt.title(f"Degree {d}, TrainingMSE = {trainMSE:.2f},\nTestMSE = {testMSE}")
    plt.legend(["Training data", "Test data", "Polynomial regression line"])

plt.show()

