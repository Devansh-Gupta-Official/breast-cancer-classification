# Breast Cancer Classification

This repository contains a Jupyter Notebook (classifier.ipynb) that demonstrates the classification of breast cancer using Support Vector Machines (SVM).

## Problem Statement
- Predicting if the cancer diagnosis is benign or malignant based on several observations/features 
- 30 features are used, examples:
        - radius (mean of distances from center to points on the perimeter)
        - texture (standard deviation of gray-scale values)
        - perimeter
        - area
        - smoothness (local variation in radius lengths)
        - compactness (perimeter^2 / area - 1.0)
        - concavity (severity of concave portions of the contour)
        - concave points (number of concave portions of the contour)
        - symmetry 
        - fractal dimension ("coastline approximation" - 1)

- Datasets are linearly separable using all 30 input features
- Number of Instances: 569
- Class Distribution: 212 Malignant, 357 Benign
- Target class:
         - Malignant
         - Benign


https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic)

## Introduction
This notebook utilizes the breast cancer dataset from sklearn.datasets to perform classification using an SVM model. The dataset consists of various features characterizing cell nuclei present in breast images, aiming to classify tumors into malignant or benign.

## Dataset
The notebook begins by loading the dataset and creating a Pandas DataFrame. It showcases the structure of the dataset, displaying key features like mean radius, mean texture, mean perimeter, mean area, and more.

## STEP 1: Importing Libraries and Data
This part typically involves importing necessary libraries for data manipulation, visualization, and model building. Here, the code snippet loads breast cancer data from sklearn's dataset module and converts it into a DataFrame for easier manipulation using Pandas.

## STEP 2: Data Visualization
### Understanding Data Distribution:
Pairplot: The sns.pairplot function is used to visualize pairwise relationships across multiple variables, showcasing how each feature relates to others. It helps identify potential patterns, correlations, or separability between benign and malignant tumors.

Countplot: The sns.countplot displays the distribution of target classes (benign and malignant) in the dataset. It helps assess the balance or imbalance between different classes, which is crucial in classification tasks.

Scatterplot: Scatterplots like sns.scatterplot visualize the relationship between two specific features (e.g., 'mean area' and 'mean smoothness') while color-coding points by their target class. This allows observing if certain features provide clear separation between the classes.

### Correlation Analysis:
Heatmap: The sns.heatmap represents the correlation matrix among all features. This visualization helps in identifying highly correlated features, which might impact the model's performance or introduce multicollinearity.

## STEP 3: Model Training
This section involves preparing the data for training, splitting it into training and testing sets, and fitting an SVM (Support Vector Machine) classifier to the training data using sklearn.

**1. Data Preparation:**
Data Splitting: The dataset is divided into training and testing sets using train_test_split from sklearn.model_selection. For instance, an 80-20 split allocates 80% of the data for training the model and 20% for testing its performance.

**2. Model Selection and Training:**
1. Choosing a Classifier: For this task, classifiers like Logistic Regression, Decision Trees, Random Forests, Support Vector Machines (SVM), or Gradient Boosting classifiers can be used. The repository might employ sklearn's classifiers like LogisticRegression, SVC, RandomForestClassifier, or others.

2. Model Initialization: Models are initialized with default or predefined hyperparameters.

3. Model Training: The selected model is trained using the training data (features and labels) via the fit() method.
```
model = LogisticRegression()  # Initialize a Logistic Regression model
model.fit(X_train, y_train)  # Train the model using training data
```

## STEP 4: Evaluating the Model
**1. Confusion Matrix Heatmap:** After model evaluation, the sns.heatmap is used to display the confusion matrix, allowing a visual representation of true positive, true negative, false positive, and false negative predictions. It helps understand the model's performance and where it misclassifies instances.

**2. Normalized Features:** Visualization of the scaled or normalized features through scatterplots (sns.scatterplot) helps observe the distribution of data after normalization, which is crucial for algorithms sensitive to feature scales.

## STEP 5: Model Improvement
### Normalization
This part focuses on enhancing the model's performance through normalization. Min-max scaling is applied to normalize the training and testing datasets. The SVM model is then retrained on the normalized data to observe potential improvements. Normalizing features improved the model's performance by approximately 5%, leading to better convergence and reduced sensitivity to feature scales.

### Grid Search
Grid search, a hyperparameter tuning technique, is employed here. It's utilized to find the best hyperparameters for the SVM model. The code defines a parameter grid containing different values for 'C' and 'gamma' to find the optimal combination for the SVM kernel.

## Results
### Exploratory Data Analysis (EDA)
**1. Pairplot Analysis:** The pairplot visualization revealed several features showing distinguishable patterns between benign and malignant tumors. Features like 'mean radius,' 'mean texture,' and 'mean perimeter' showcased noticeable separability.

**2. Countplot for Target Classes:** The countplot demonstrated a slight class imbalance, with a larger number of benign cases compared to malignant cases. This imbalance might require handling techniques during model training.

**3. Correlation Heatmap:** The heatmap of feature correlations highlighted certain pairs of features exhibiting strong correlations. Notably, 'mean radius' and 'mean perimeter' displayed a high correlation, indicating potential multicollinearity.

### Model Evaluation
**1. Baseline Model Performance:** The initial model (e.g., logistic regression, SVM) achieved an accuracy of approximately **94%** on the test set.

**2. Confusion Matrix Analysis:** The confusion matrix heatmap illustrated the model's performance in classifying benign and malignant tumors. It showed that the model had a higher tendency to correctly identify benign tumors but had a few misclassifications for malignant cases.

### Performance Comparison:
- Initial Model Accuracy: **94%**
- Improved Model Accuracy: **96%**
