# Import necessary libraries for running the entire workflow, including data loading, preprocessing, GA optimization, and evaluation
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from deap import base, creator, tools, algorithms
import random
import numpy as np
from scipy.io import arff
import io
import requests
import matplotlib.pyplot as plt

# Load the dataset from the .arff file using the UCI dataset URL
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00327/Training%20Dataset.arff"
response = requests.get(url)
data, meta = arff.loadarff(io.StringIO(response.text))

# Convert the dataset to a pandas DataFrame
df = pd.DataFrame(data)

# Assign feature and target columns
X = df.iloc[:, :-1]  # All features except the last column
y = df.iloc[:, -1].apply(lambda x: 1 if x == b'1' else 0)  # Target column, converting byte strings to integer labels

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Define the GA optimization problem
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("attr_bool", random.randint, 0, 1)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, n=X_train.shape[1])
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# Function to evaluate the performance of the individual feature set
def eval_phishing(individual):
    # Create a feature mask based on the individual
    mask = np.array(individual, dtype=bool)
    X_selected = X_train.iloc[:, mask]
    
    # Train a classifier (Random Forest)
    clf = RandomForestClassifier(n_estimators=50, random_state=42)
    clf.fit(X_selected, y_train)
    
    # Evaluate performance on test set
    X_test_selected = X_test.iloc[:, mask]
    y_pred = clf.predict(X_test_selected)
    accuracy = accuracy_score(y_test, y_pred)
    return (accuracy,)

toolbox.register("evaluate", eval_phishing)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
toolbox.register("select", tools.selTournament, tournsize=3)

# Define the Genetic Algorithm process
def run_genetic_algorithm():
    random.seed(42)
    population = toolbox.population(n=50)
    NGEN = 20
    CXPB = 0.5
    MUTPB = 0.2

    best_fitness_per_gen = []
    
    for gen in range(NGEN):
        print(f"Generation {gen}")
        
        # Evaluate the individuals
        fitnesses = map(toolbox.evaluate, population)
        for ind, fit in zip(population, fitnesses):
            ind.fitness.values = fit
        
        # Record best fitness of this generation
        top_individual = tools.selBest(population, 1)[0]
        best_fitness_per_gen.append(top_individual.fitness.values[0])
        
        # Select the next generation
        offspring = toolbox.select(population, len(population))
        offspring = list(map(toolbox.clone, offspring))
        
        # Apply crossover and mutation
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < CXPB:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values
        
        for mutant in offspring:
            if random.random() < MUTPB:
                toolbox.mutate(mutant)
                del mutant.fitness.values
        
        # Evaluate the new individuals
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
        
        population[:] = offspring
    
    # Return the best solution and fitness progression
    top_individual = tools.selBest(population, 1)[0]
    return top_individual, best_fitness_per_gen

# Run the genetic algorithm to find the best feature set
best_individual, fitness_progression = run_genetic_algorithm()
print("Best individual (feature selection mask):", best_individual)

# Evaluate the best feature set on the test set and calculate various performance metrics
best_mask = np.array(best_individual, dtype=bool)
X_train_selected = X_train.iloc[:, best_mask]
X_test_selected = X_test.iloc[:, best_mask]

# Train the final model using RandomForest on the best feature set
final_clf = RandomForestClassifier(n_estimators=50, random_state=42)
final_clf.fit(X_train_selected, y_train)
y_pred_final = final_clf.predict(X_test_selected)

# Calculate performance metrics
accuracy = accuracy_score(y_test, y_pred_final)
precision = precision_score(y_test, y_pred_final)
recall = recall_score(y_test, y_pred_final)
f1 = f1_score(y_test, y_pred_final)
roc_auc = roc_auc_score(y_test, y_pred_final)

# Plot the fitness progression across generations
plt.plot(fitness_progression)
plt.title('Fitness Progression Across Generations')
plt.xlabel('Generation')
plt.ylabel('Best Fitness (Accuracy)')
plt.show()

# Output performance metrics
performance_results = {
    'Accuracy': accuracy,
    'Precision': precision,
    'Recall': recall,
    'F1 Score': f1,
    'AUC-ROC': roc_auc
}

import ace_tools as tools; tools.display_dataframe_to_user(name="Performance Results", dataframe=pd.DataFrame([performance_results]))
