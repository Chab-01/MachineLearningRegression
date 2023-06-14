import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

data = pd.read_csv("microchips.csv", header=None).values
x = data[:, 0]
y = data[:, 1]
dataWithoutLabels = data[:,:2]
labels = data[:, 2]
C = [10000.0, 1.0]
def mapFeature(X1,X2,D, ones):
  one = np.ones([len(X1),1])
  if ones:
    Xe = np.c_[one,X1,X2] #Start with [1,X1,X2]
  else: 
    Xe = np.c_[X1,X2]
  for i in range(2,D+1):
    for j in range(0,i+1):
      Xnew = X1**(i-j)*X2**j #type (N)
      Xnew = Xnew.reshape(-1,1) #type (N,1) required by append
      Xe = np.append(Xe,Xnew,1) #axis = 1 ==> append column
  return Xe

#Task 1
logreg = LogisticRegression(solver="lbfgs", C=C[0], tol=1e-6,max_iter=1000) #Used for a high C value

for i in range(9):
  plt.figure(1)
  plt.subplot(331+i)

  Xe = mapFeature(x, y, i+1, False)
  logreg.fit(Xe, labels)
  yPred = logreg.predict(Xe)
  errors = np.sum(yPred != labels)

  h = .01 #step size in the mesh
  x_min, x_max = x.min()-0.1, x.max()+0.1
  y_min, y_max = y.min()-0.1, y.max()+0.1
  xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h)) #Mesh Grid
  x1,x2 = xx.ravel(), yy.ravel() #Turn to two Nx1 arrays
  xyMesh = mapFeature(x1, x2, i+1, False)
  classes = logreg.predict(xyMesh)
  clz_mesh = classes.reshape(xx.shape) # return to mesh format
  cmap_light = ListedColormap(["#FFAAAA", "#AAFFAA", "#AAAAFF"]) #mesh plot
  cmap_bold = ListedColormap(["#FF0000", "#00FF00", "#0000FF"]) #colors
  plt.pcolormesh(xx, yy, clz_mesh, cmap=cmap_light)
  plt.scatter(x, y ,c=labels, marker=".",cmap=cmap_bold)

  plt.title(f"Degree: {i+1}, Errors: {errors}")

#Task 2
logreg2 = LogisticRegression(solver="lbfgs", C=C[1], tol=1e-6,max_iter=1000) #Used for a low C
for i in range(9):
  plt.figure(2)
  plt.subplot(331+i)

  Xe = mapFeature(x, y, i+1, False)
  logreg2.fit(Xe, labels)
  yPred = logreg2.predict(Xe)
  errors = np.sum(yPred != labels)

  h = .01 #step size in the mesh
  x_min, x_max = x.min()-0.1, x.max()+0.1
  y_min, y_max = y.min()-0.1, y.max()+0.1
  xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h)) #Mesh Grid
  x1,x2 = xx.ravel(), yy.ravel() #Turn to two Nx1 arrays
  xyMesh = mapFeature(x1, x2, i+1, False)
  classes = logreg2.predict(xyMesh)
  clz_mesh = classes.reshape(xx.shape) #return to mesh format
  cmap_light = ListedColormap(["#FFAAAA", "#AAFFAA", "#AAAAFF"]) #mesh plot
  cmap_bold = ListedColormap(["#FF0000", "#00FF00", "#0000FF"]) #colors
  plt.pcolormesh(xx, yy, clz_mesh, cmap=cmap_light)
  plt.scatter(x, y ,c=labels, marker=".",cmap=cmap_bold)

  plt.title(f"Degree: {i+1}, Errors: {errors}")

#Task 3
plt.figure(3)
errorsReg = []
errorsUnReg = []

for i in range(9):
  for j in C:
    Xe = mapFeature(x, y, i, False)
    logreg = LogisticRegression(solver="lbfgs", C=j, tol=1e-6,max_iter=1000)
    scores = cross_val_score(logreg, Xe, labels) #Get the error rates
    amountOfErrors = round((1 - np.mean(scores)) * len(labels)) #Here we get the actual number of errors
    
    if j == C[0]:
      errorsUnReg.append(amountOfErrors) #We append to the list depending on what the C value is
    if j == C[1]:
      errorsReg.append(amountOfErrors) 

plt.plot(range(1,10), errorsUnReg, label="Unregularized") #we plot from 1-9 representing the degrees and the errors on the y axis
plt.plot(range(1,10), errorsReg, label="Regularized")
plt.ylabel("Number of errors")
plt.xlabel("Degree")
plt.legend()


plt.show()