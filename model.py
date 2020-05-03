## Linear Regression
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt

data = pd.read_csv('salary_data/data.csv')
X = data.iloc[:,0].values
y = data.iloc[:,1].values

X = X.reshape(-1,1)

class LinearRegression:
    def __init__(self):
        self.weight = np.zeros(1)
        self.bias = np.array(1.0)

    def predict(self,X):
        return np.dot(self.weight,X.T) + self.bias

    def computeCost(self,X,y):
        m,_ = X.shape
        pred = self.predict(X)
        return  ((pred - y)**2).mean()/2    

    def gradientDescent(self,X,y,alpha=0.05,epochs=5000):
        for e in range(epochs):
            m,_ = X.shape
            y_hat = self.predict(X)
            self.weight -= alpha * ((y_hat - y)*X[:,0]).mean()
            self.bias -= alpha * (y_hat - y).mean()
            J = self.computeCost(X,y)
        return self.weight,self.bias

    def train(self,X,y):
        m = X.shape
        self.weight,self.bias = self.gradientDescent(X,y,alpha=0.05,epochs=500)

model = LinearRegression()
model.train(X,y)
preds = model.predict(X)

plt.plot(X,y,'.',X,preds,'-')
plt.xlabel("Years of Experience")
plt.ylabel("Salary in $")
plt.title("Salary Prediction (scratch lr)")
plt.show()

pickle.dump(open("checkpoint.pkl","wb"))