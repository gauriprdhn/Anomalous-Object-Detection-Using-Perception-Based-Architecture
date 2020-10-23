import numpy as np 
from sklearn.model_selection import train_test_split

class DataSet:
  def __init__(self,X,y,numLabels,split_size = 0.20,random_state=42):
      self.X = X
      self.y = y
      self.numLabels = numLabels
      self.split_size = split_size
      self.random_state = random_state
  
  def __normalize__(self,X):
      X = np.asarray(X)
      X = X.astype("float32")
      X = np.true_divide(X,np.max(X))
      return X
  def __oneHotEncoding__(self,labels):
    numImageSamples = len(labels)
    encodedLabels = np.zeros((numImageSamples,self.numLabels))
    for i in range(0,numImageSamples,1):
        encodedLabels[i,labels[i]] = 1
    return encodedLabels

  def train_test_split(self):
      X_train, X_test,y_train, y_test = train_test_split(self.X,self.y, test_size=self.split_size, random_state=self.random_state)
      if type(self.X) is list and type(self.y) is list:
        X_train = self.__normalize__(X_train)
        X_test = self.__normalize__(X_test)
        y_train = self.__oneHotEncoding__(y_train)
        y_test = self.__oneHotEncoding__(y_test)
      return X_train, X_test, y_train , y_test