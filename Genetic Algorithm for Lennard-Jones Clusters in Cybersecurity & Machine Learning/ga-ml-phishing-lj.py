# Import necessary libraries
import pandas as pd
import numpy as np
from scipy.io import arff
import io
import requests

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    auc,
)
from sklearn.ensemble import RandomForestClassifier

from deap import base, creator, tools
import random

import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")

# Load the dataset from the .arff file using the UCI dataset URL
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00327/Training%20Dataset.arff"
response = requests.get(url)
data, meta = arff.loadarff(io.StringIO(response.text))

# Convert the dataset to a pandas DataFrame
df = pd.DataFrame(data)

# Preprocess the target variable (convert byte strings to integers)
df["Result"] = df["Result"].apply(lambda x: 1 if x == b"1" else 0)

# Assign feature and target columns
X = df.drop("Result", axis=1)
y = df["Result"]

# Ensure features are numeric (convert byte strings to integers)
X = X.applymap(lambda x: int(x) if isinstance(x, bytes) else x)

# Split the dataset into training, validation, and testing sets with stratification
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.25, stratify=y_temp, random_state=42
)
# Now, X_train and y_train: 60%, X_val and y_val: 20%, X_test and y_test: 20%

# Define the GA optimization problem
creator.create("FitnessMax", base.Fitness, weights=(1.0,))  # Maximize F1-score
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("attr_bool", random.randint, 0, 1)
# Initialize individual with random 0s and 1s for feature selection
toolbox.register(
    "individual",
    tools.initRepeat,
    creator.Individual,
    toolbox.attr_bool,
    n=X_train.shape[1],
)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# Evaluation function using cross-validation on the training data
def eval_phishing(individual):
    mask = np.array(individual, dtype=bool)
    if np.count_nonzero(mask) == 0:
        return (0.0,)  # Avoid models with no features
    X_selected = X_train.iloc[:, mask]
    clf = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42)
    # Use stratified 5-fold cross-validation
    skf = StratifiedKFold(n_splits=5)
    scores = cross_val_score(clf, X_selected, y_train, cv=skf, scoring="f1")
    return (scores.mean(),)

toolbox.register("evaluate", eval_phishing)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
toolbox.register("select", tools.selTournament, tournsize=3)

# Genetic Algorithm process
def run_genetic_algorithm():
    random.seed(42)
    population = toolbox.population(n=50)
    NGEN = 20
    CXPB = 0.5  # Crossover probability
    MUTPB = 0.2  # Mutation probability

    best_fitness_per_gen = []
    avg_fitness_per_gen = []

    for gen in range(NGEN):
        print(f"Generation {gen}")

        # Evaluate the individuals
        fitnesses = list(map(toolbox.evaluate, population))
        for ind, fit in zip(population, fitnesses):
            ind.fitness.values = fit

        # Record statistics
        top_individual = tools.selBest(population, 1)[0]
        best_fitness = top_individual.fitness.values[0]
        mean_fitness = np.mean([ind.fitness.values[0] for ind in population])
        best_fitness_per_gen.append(best_fitness)
        avg_fitness_per_gen.append(mean_fitness)

        print(f"Best F1 Score in generation {gen}: {best_fitness:.4f}")

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
        fitnesses = list(map(toolbox.evaluate, invalid_ind))
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        population[:] = offspring

    # Return the best solution and fitness progression
    top_individual = tools.selBest(population, 1)[0]
    return top_individual, best_fitness_per_gen, avg_fitness_per_gen

# Run the genetic algorithm to find the best feature set
best_individual, best_fitness_progression, avg_fitness_progression = run_genetic_algorithm()
print("\nBest individual (feature selection mask):")
print(best_individual)

# Evaluate the best feature set on the validation set and calculate various performance metrics
best_mask = np.array(best_individual, dtype=bool)
X_train_selected = X_train.iloc[:, best_mask]
X_val_selected = X_val.iloc[:, best_mask]

# Train the final model using RandomForest on the best feature set
final_clf = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42)
final_clf.fit(X_train_selected, y_train)
y_pred_val = final_clf.predict(X_val_selected)

# Calculate performance metrics on validation set
accuracy_val = accuracy_score(y_val, y_pred_val)
precision_val = precision_score(y_val, y_pred_val)
recall_val = recall_score(y_val, y_pred_val)
f1_val = f1_score(y_val, y_pred_val)
roc_auc_val = roc_auc_score(y_val, y_pred_val)

print("\nPerformance on Validation Set:")
print(f"Accuracy: {accuracy_val:.4f}")
print(f"Precision: {precision_val:.4f}")
print(f"Recall: {recall_val:.4f}")
print(f"F1 Score: {f1_val:.4f}")
print(f"AUC-ROC: {roc_auc_val:.4f}")

# Finally, evaluate on the test set
X_test_selected = X_test.iloc[:, best_mask]
y_pred_test = final_clf.predict(X_test_selected)

# Calculate performance metrics on test set
accuracy_test = accuracy_score(y_test, y_pred_test)
precision_test = precision_score(y_test, y_pred_test)
recall_test = recall_score(y_test, y_pred_test)
f1_test = f1_score(y_test, y_pred_test)
roc_auc_test = roc_auc_score(y_test, y_pred_test)

print("\nPerformance on Test Set:")
print(f"Accuracy: {accuracy_test:.4f}")
print(f"Precision: {precision_test:.4f}")
print(f"Recall: {recall_test:.4f}")
print(f"F1 Score: {f1_test:.4f}")
print(f"AUC-ROC: {roc_auc_test:.4f}")

# Plot the fitness progression across generations and save to file
plt.figure(figsize=(10, 6))
plt.plot(best_fitness_progression, label="Best Fitness")
plt.plot(avg_fitness_progression, label="Average Fitness")
plt.title("Fitness Progression Across Generations")
plt.xlabel("Generation")
plt.ylabel("F1 Score")
plt.legend()
plt.grid(True)
plt.savefig("fitness_progression.png")  # Save the plot to a file
plt.show()

# Plot ROC curve for the test set and save to file
y_pred_proba_test = final_clf.predict_proba(X_test_selected)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba_test)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc_test:.4f})")
plt.plot([0, 1], [0, 1], "k--")  # Diagonal line
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.title("Receiver Operating Characteristic - Test Set")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend(loc="lower right")
plt.grid(True)
plt.savefig("roc_curve_test_set.png")  # Save the plot to a file
plt.show()

# Output performance metrics
performance_results_val = {
    "Accuracy": accuracy_val,
    "Precision": precision_val,
    "Recall": recall_val,
    "F1 Score": f1_val,
    "AUC-ROC": roc_auc_val,
}

performance_results_test = {
    "Accuracy": accuracy_test,
    "Precision": precision_test,
    "Recall": recall_test,
    "F1 Score": f1_test,
    "AUC-ROC": roc_auc_test,
}

print("\nValidation Performance Metrics:")
print(performance_results_val)

print("\nTest Performance Metrics:")
print(performance_results_test)
