import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

data = pd.read_csv('./salary-data/Salary_Data.csv')



X = data.iloc[:,0].values
y = data.iloc[:,-1].values

X = X.reshape(-1,1) # to remove rank 1 martix
y = y.reshape(-1,1) # to remove rank 1 matrix

print("Dataset shape: " + str(data.shape))
print("X shape: " + str(X.shape))
print("Y shape: " + str(y.shape))

x_train,x_test,y_train,y_test = train_test_split(X,y,train_size=0.8,test_size=0.2,random_state=10)

lm = LinearRegression()
lm.fit(x_train,y_train)
y_pred = lm.predict(x_train)
print("R2 Score for train_data: " + str(r2_score(y_pred,y_train)))

#plotting
plt.scatter(x_train,y_train,color='red',alpha=0.5)
plt.plot(x_train,y_pred)
plt.xlabel('Experience in years')
plt.ylabel('Salary in $')
plt.title('Salary Prediction on Train Data')
plt.show()

y_pred_test = lm.predict(x_test)
print("R2 Score for test data: " + str(r2_score(y_pred_test,y_test)))

#plotting
plt.scatter(x_test,y_test,color='red',alpha=0.5)
plt.plot(x_test,y_pred_test)
plt.xlabel('Experience in years')
plt.ylabel('Salary in $')
plt.title('Salary Prediction on Test Data')
plt.show()

print("Enter Experience in years: ")
experience = [int(input())]
# print(salary)
# print(type(salary))
experience_arr = np.array([experience])
# print(experience.shape)
sal_pred = lm.predict(experience_arr)
# sal_pred = sal_pred.reshape(1,0)
print(f"Estimated Pay for {experience} years experience: $ {sal_pred} | Accuracy: {r2_score(y_pred_test,y_test)*100}%")