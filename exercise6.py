import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

data = pd.read_csv("cars-mpg.csv").values
x = data[:,1:]
y = data[:,0]
dataLength = len(data)
trainingSetSize = round(0.8*dataLength)
validationSetsize = dataLength - trainingSetSize
index = np.arange(dataLength)
np.random.seed(1)
np.random.shuffle(index)

trainSet = data[index[:trainingSetSize]]
validationSet = x[index[:validationSetsize]]

def cost(X, y, beta):
    y_pred = np.dot(X, beta)
    mse = np.mean((y - y_pred) ** 2)
    return mse

def forward_selection(X, y, testX, testY):
  features = X.shape[1] #Amount of features, in this case it is 7
  linreg = LinearRegression()
  remainingFeatures = list(range(features)) #A list that contains so many features as there are columns in the dataset
  best = []

  for i in range(features):
    costs = []
    for j in remainingFeatures: #iterates through features and adding it to selectedFeatures
      selectedFeatures = X[:, best + [j]]  # We store the +j feature
      coef = linreg.fit(selectedFeatures, y).coef_ # Get the coefficients
      costs.append(mean_squared_error(y, linreg.predict(selectedFeatures)))

    z = remainingFeatures[np.argmin(costs)] #finds index of the feature in remainingFeatures with lowest cost
    best.append(z)
    remainingFeatures.remove(z) #Removes the index of z from the remaingFeatures

    #evaluate model using test set
    selectedFeatures = X[:, best]
    linreg.fit(selectedFeatures, y)
    testPred = linreg.predict(testX[:, best])
    testMSE = mean_squared_error(testY, testPred)
    print(f"Selected features: {best}, Test MSE: {testMSE:.2f}")
  return best

best = forward_selection(trainSet[:,1:], trainSet[:,0], validationSet, validationSet[:,0]) 
print(f"Best model: {best}")
print(f"Most important feature: {best[0]}")
