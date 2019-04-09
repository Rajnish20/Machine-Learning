# Regression Template

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

"""#feature scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)""""



#Fitting the Regression to the dataset
#Create your regressor here



#Predecting The truth or bluff from Polynomial linear Regression
y_pred = regressor.predict(6.5)


#visualising the  regression results
plt.scatter(X, y, color = 'red')
plt.plot(X, regressor.predict(X), color = 'blue')
plt.title('Truth or Bluff')
plt.xlabel('Levels')
plt.ylabel('Salaries')
plt.show()

#visualising the  regression results(with higher resolution)
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'red')
plt.plot(X, regressor.predict(X), color = 'blue')
plt.title('Truth or Bluff')
plt.xlabel('Levels')
plt.ylabel('Salaries')
plt.show()




