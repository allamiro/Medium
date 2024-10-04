# Importing necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from deap import base, creator, tools, algorithms
import random
import numpy as np

# Load the dataset (replace the file path with your dataset path)
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00327/PhishingData.arff"
data = pd.read_csv(url)


