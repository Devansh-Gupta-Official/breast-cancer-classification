# Breast Cancer Classification

This repository contains a Jupyter Notebook (classifier.ipynb) that demonstrates the classification of breast cancer using Support Vector Machines (SVM).

## Introduction
This notebook utilizes the breast cancer dataset from sklearn.datasets to perform classification using an SVM model. The dataset consists of various features characterizing cell nuclei present in breast images, aiming to classify tumors into malignant or benign.

## Dataset
The notebook begins by loading the dataset and creating a Pandas DataFrame. It showcases the structure of the dataset, displaying key features like mean radius, mean texture, mean perimeter, mean area, and more.

## Data Visualization
### Pair Plot
Visualizes relationships between select features using Seaborn's pairplot function, coloring data points based on their target labels.

### Count Plot
Displays the count of each target label in the dataset using Seaborn's countplot.

### Scatter Plots
Two scatter plots visualize relationships between 'mean area' and 'mean smoothness', one colored by target label.

### Heatmap
Displays a heatmap of feature correlations using Seaborn's heatmap.

## Model Training
The notebook moves on to train an SVM model using the dataset, splitting it into training and test sets.

## Evaluating the Model
### Confusion Matrix
Displays a heatmap of the confusion matrix and provides classification metrics like precision, recall, and accuracy.

## Model Improvement
### Normalization
Demonstrates data normalization using Min-Max scaling and its impact on model performance.

### Grid Search
Utilizes Grid Search with cross-validation to optimize hyperparameters for the SVM model.

