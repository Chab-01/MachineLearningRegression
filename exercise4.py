import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

data = pd.read_csv("microchips.csv", header=None).values
x = data[:, 0]
y = data[:, 1]
dataWithoutLabels = data[:,:2]
labels = data[:, 2]
dataLength = len(data)

def sigmoid(x): 
  return 1/(1+np.exp(-x))

def cost(x, y, theta): #Calculate cost using sigmoid
    yLength = len(y)
    pred = sigmoid(np.dot(x, theta))
    cost = (-1/yLength) * np.sum(y*np.log(pred) + (1-y)*np.log(1-pred))
    return cost

def gradientDescent(x, y, learningRate, iterations): #Similar gradient from first exercise, only this one uses sigmoid
    yLength = len(y)
    costs = []
    theta = np.zeros(x.shape[1])

    for i in range(iterations):
        h = sigmoid(np.dot(x, theta))
        grad = (1/yLength) * np.dot(x.T, h-labels)
        theta -= learningRate * grad
        costs.append([i,cost(x, labels, theta)])
    return theta, costs

#Task 3    
def mapFeature(X1,X2,D):
  one = np.ones([len(X1),1])
  Xe = np.c_[one,X1,X2] #Start with [1,X1,X2]
  for i in range(2,D+1):
    for j in range(0,i+1):
      Xnew = X1**(i-j)*X2**j #type (N)
      Xnew = Xnew.reshape(-1,1) #type (N,1) required by append
      Xe = np.append(Xe,Xnew,1) #axis = 1 ==> append column
  return Xe

#Task 1
plt.figure(1)
plt.title("Microship data")
colors = ['green' if i == 1 else 'red' for i in labels]
plt.scatter(x, y, s=12, c=['green' if i == 1 else 'red' for i in labels])

#Task 2
learningRate = 0.1
iterations = 5000
Xe = np.c_[np.ones((dataWithoutLabels.shape[0], 1)), x, y, x**2, x*y, y**2]
theta, costs = gradientDescent(Xe, y, learningRate, iterations)
costs = np.asanyarray(costs)
predictions = np.round(sigmoid(np.dot(Xe, theta))) 
errors = np.sum(predictions != labels)
print("Learning rate:", learningRate)
print("Number of iterations:", iterations)

plt.figure(2)
plt.subplot(121)
plt.xlabel("Iterations")
plt.ylabel("Cost")
plt.plot(costs[:,0],costs[:,1])

h = .01 #step size in the mesh
x_min, x_max = x.min()-0.1, x.max()+0.1
y_min, y_max = y.min()-0.1, y.max()+0.1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h)) # Mesh Grid
x1,x2 = xx.ravel(), yy.ravel() #Turn to two Nx1 arrays
XXe = mapFeature(x1,x2,2) #Extend matrix for degree 2
p = sigmoid( np.dot(XXe, theta) ) #classify mesh ==> probabilities
classes = p>0.5 #round off probabilities
clz_mesh = classes.reshape(xx.shape) #return to mesh format
cmap_light = ListedColormap(["#FFAAAA", "#AAFFAA", "#AAAAFF"]) #mesh plot
cmap_bold = ListedColormap(["#FF0000", "#00FF00", "#0000FF"]) #colors
plt.subplot(122)
plt.title(f"Decision boundary, Errors = {errors}")
plt.pcolormesh(xx, yy, clz_mesh, cmap=cmap_light)
plt.scatter(x, y ,c=labels, marker=".",cmap=cmap_bold)

#Task 4
Xe = mapFeature(x, y, 5) #Extend matrix for degree 5
theta, costs = gradientDescent(Xe, y, learningRate, iterations)
costs = np.asanyarray(costs)
predictions5 = np.round(sigmoid(np.dot(Xe, theta))) 
error = np.sum(predictions5 != labels)

plt.figure(3)
plt.subplot(121)
plt.xlabel("Iterations")
plt.ylabel("Cost")
plt.plot(costs[:,0],costs[:,1])

h = .01 #step size in the mesh
x_min, x_max = x.min()-0.1, x.max()+0.1
y_min, y_max = y.min()-0.1, y.max()+0.1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h)) #Mesh Grid
x1,x2 = xx.ravel(), yy.ravel() #Turn to two Nx1 arrays
XXe = mapFeature(x1,x2,5) #Extend matrix for degree 5
p = sigmoid( np.dot(XXe, theta) ) #classify mesh ==> probabilities
classes = p>0.5 #round off probabilities
clz_mesh = classes.reshape(xx.shape) # return to mesh format
cmap_light = ListedColormap(["#FFAAAA", "#AAFFAA", "#AAAAFF"]) #mesh plot
cmap_bold = ListedColormap(["#FF0000", "#00FF00", "#0000FF"]) #colors
plt.subplot(122)
plt.title(f"Decision boundary, Errors = {error}")
plt.pcolormesh(xx, yy, clz_mesh, cmap=cmap_light)
plt.scatter(x, y ,c=labels, marker=".",cmap=cmap_bold)

plt.show()
