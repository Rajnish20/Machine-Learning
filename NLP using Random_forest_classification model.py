# Natural Language Processing

#Importing the libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Importing the dataset
dataset = pd.read_csv('Restaurant_Reviews.tsv', delimiter = '\t', quoting = 3)

#cleaning the text
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords 
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()
corpus = []
for i in range(0, 1000):
    review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])
    review = review.lower()
    review = review.split()
    review = [ps.stem(word) for word in review if not word in stopwords.words('english')]
    #using set in case if we have a bigger sentence or a book that we have to remove irrelevant text from
    review = [word for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review) # Back to the original review but clean
    corpus.append(review)
    
# Creating the Bag of words through Tokenization
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 1500)
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:, 1].values

#splitting the dataset into train and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)


#Fitting Random_Forest_Classification to the Training Set
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0 )
classifier.fit(X_train, y_train)


#Predecting the results of Test Set
y_pred = classifier.predict(X_test)

#Creating the Confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

#Analysing the different Performance metric of model
Accuracy = (86 + 62)/200
Precision = 62 / (11 + 62)
Recall = 62 / (62 + 41)
F1_score = 2 * Precision * Recall/(Precision + Recall)