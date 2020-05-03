## Linear Regression from Scratch

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle

data = pd.read_csv("./salary_data/data.csv")
# print(data.head(4))
x = data.iloc[:,0].values.reshape(-1,1)
y = data.iloc[:,1].values

class LinearRegression():

    def predict(self,X,theta):
        return np.dot(X,theta)

    def computeCost(self,X,y,theta):
        prediction = self.predict(X,theta)
        return ((prediction - y)**2).mean()/2

    def gradientDescent(self,X,y,theta,alpha=0.01,epochs=1000):
        for e in range(epochs):
            preds = self.predict(X,theta)
            theta[0] -= alpha * ((preds - y)).mean()
            theta[1] -= alpha * ((preds - y)*X[:,1]).mean()
            J = self.computeCost(X,y,theta)
        return theta

    def train(self,X,y):
        m= X.shape
        theta = np.zeros(2)
        X = np.column_stack((np.ones(m),X))
        theta = self.gradientDescent(X,y,theta,alpha=0.005,epochs=5000)
        preds = self.predict(X,theta)
        print(theta)
        return preds

model = LinearRegression()
preds = model.train(x,y)

plt.scatter(x,y,color='r',alpha=0.5)
plt.title("Salary Prediction scratch")
plt.xlabel("Experience in years")
plt.ylabel("Salary in $")
plt.plot(x,preds)
plt.show()

pickle.dump(model,open('checkpt.pkl','wb'))

test_model = pickle.load(open("checkpt.pkl","rb"))
pred = test_model.train(x,y)