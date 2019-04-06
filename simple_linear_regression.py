#Simple Linear Regression

#importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing the data_set
dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values


#splitting the dataset into train and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)

"""#feature scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)""""

#Fittin the training set to the regressor
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

#Predicting the test set values
y_pred = regressor.predict(X_test)

#visualising the training set results
plt.scatter(X_train, y_train, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Salary vs Experience(Training Set)')
plt.xlabel('Years Of Experience')
plt.ylabel('Salary')
plt.show()

#visualising the test set results
plt.scatter(X_test, y_test, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Salary vs Experience(Test Set)')
plt.xlabel('Years Of Experience')
plt.ylabel('Salary')
plt.show()
