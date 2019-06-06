# Natural Language Processing

#Importing the libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Importing the dataset
dataset = pd.read_csv('Restaurant_Reviews.tsv', delimiter = '\t', quoting = 3)

#cleaning the text
import re
review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][0])
review = review.lower()