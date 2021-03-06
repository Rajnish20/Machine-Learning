#Support Vector Regression 

#importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing the data_set
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values


#splitting the dataset into train and test set
"""from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)"""

#feature scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y= StandardScaler()
X = sc_X.fit_transform(X)
y = sc_y.fit_transform(y.reshape(-1,1))




#Fitting the Regression to the dataset
from sklearn.svm import SVR
regressor = SVR(kernel = 'rbf')
regressor.fit(X, y)



#Predecting The truth or bluff from Polynomial linear Regression
y_pred = sc_y.inverse_transform(regressor.predict(0.4))


#visualising the  regression results
plt.scatter(X, y, color = 'red')
plt.plot(X, regressor.predict(X), color = 'blue')
plt.title('Truth or Bluff')
plt.xlabel('Levels')
plt.ylabel('Salaries')
plt.show()

#visualising the  regression results(with higher resolution)
"""X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'red')
plt.plot(X, regressor.predict(X), color = 'blue')
plt.title('Truth or Bluff')
plt.xlabel('Levels')
plt.ylabel('Salaries')
plt.show()"""

