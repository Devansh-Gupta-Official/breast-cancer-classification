# Breast Cancer Classification

This repository contains a Jupyter Notebook (classifier.ipynb) that demonstrates the classification of breast cancer using Support Vector Machines (SVM).

## Introduction
This notebook utilizes the breast cancer dataset from sklearn.datasets to perform classification using an SVM model. The dataset consists of various features characterizing cell nuclei present in breast images, aiming to classify tumors into malignant or benign.

## Dataset
The notebook begins by loading the dataset and creating a Pandas DataFrame. It showcases the structure of the dataset, displaying key features like mean radius, mean texture, mean perimeter, mean area, and more.

## Importing Libraries and Data
This part typically involves importing necessary libraries for data manipulation, visualization, and model building. Here, the code snippet loads breast cancer data from sklearn's dataset module and converts it into a DataFrame for easier manipulation using Pandas.

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
This section involves preparing the data for training, splitting it into training and testing sets, and fitting an SVM (Support Vector Machine) classifier to the training data using sklearn.

## Evaluating the Model
### Confusion Matrix
The trained model is used to predict outcomes on the test set, and a confusion matrix along with a classification report is generated to assess the model's performance in terms of precision, recall, F1-score, and accuracy.

## Model Improvement
### Normalization
This part focuses on enhancing the model's performance through normalization. Min-max scaling is applied to normalize the training and testing datasets. The SVM model is then retrained on the normalized data to observe potential improvements.

### Grid Search
Grid search, a hyperparameter tuning technique, is employed here. It's utilized to find the best hyperparameters for the SVM model. The code defines a parameter grid containing different values for 'C' and 'gamma' to find the optimal combination for the SVM kernel.

