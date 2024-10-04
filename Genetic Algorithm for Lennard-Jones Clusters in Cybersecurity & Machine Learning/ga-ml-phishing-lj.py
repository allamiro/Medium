# Importing necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from deap import base, creator, tools, algorithms
import random
import numpy as np

# Load the dataset from the .arff file
# (Make sure you have extracted the file locally and update the path to match your system)
from scipy.io import arff
import io
import requests

# Download and load the dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00327/Training%20Dataset.arff"
response = requests.get(url)
data, meta = arff.loadarff(io.StringIO(response.text))
# Convert the dataset to a pandas DataFrame
df = pd.DataFrame(data)
print(df.head(10))




# Assign feature and target columns
X = df.iloc[:, :-1]  # All features except the last column
y = df.iloc[:, -1].apply(lambda x: 1 if x == b'1' else 0)  # Target column, converting byte strings to integer labels

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)






