# Project Narrative: Chronic Kidney Disease (CKD) Prediction

## 1. Project Overview

This project aimed to develop and evaluate machine learning models to predict the presence of Chronic Kidney Disease (CKD) based on a dataset containing various health indicators. The primary goal was to build a robust classification model that can accurately identify individuals at risk of CKD.

## 2. Data Loading and Initial Exploration

The project began by loading the dataset, named `kidney_disease_dataset.csv`, from Google Drive into a pandas DataFrame. Initial exploration involved displaying the first few rows (`data.head()`) to understand the data structure and content, and examining the data types and non-null counts using `data.info()`. We confirmed that the dataset contains 2304 entries with 9 columns, and importantly, no missing values across any of the features.

## 3. Exploratory Data Analysis (EDA)

Exploratory Data Analysis was conducted to gain insights into the data distribution and relationships between variables. Key EDA steps included:

*   **Descriptive Statistics:** Generating descriptive statistics (`data.describe()`) to understand the central tendency, dispersion, and shape of the numerical features.
*   **Missing Value Check:** Explicitly verifying the absence of missing values (`data.isnull().sum()`).
*   **Visualizations:**
    *   Histograms were created for all numerical columns (`Age`, `Creatinine_Level`, `BUN`, `Diabetes`, `Hypertension`, `GFR`, `Urine_Output`, `CKD_Status`, `Dialysis_Needed`) to visualize their distributions.
    *   A scatter plot of 'Creatinine\_Level' versus 'BUN' was generated to explore the relationship between these two key indicators.
    *   A box plot of 'GFR' by 'CKD\_Status' was created to visualize the distribution of Glomerular Filtration Rate (GFR) for individuals with and without CKD. This visualization clearly showed a lower GFR in the CKD group.
*   **Correlation Analysis:** A correlation matrix was calculated and visualized as a heatmap to understand the linear relationships between numerical variables. This analysis highlighted strong negative correlations between 'GFR' and both 'Creatinine\_Level' and 'BUN', which is clinically expected as reduced kidney function (lower GFR) leads to higher levels of creatinine and BUN.

## 4. Data Preprocessing

Data preprocessing steps were performed to prepare the data for machine learning models:

*   **Feature Scaling:** Numerical features (`Age`, `Creatinine_Level`, `BUN`, `GFR`, `Urine_Output`) were scaled using `StandardScaler` to ensure that no single feature dominates the model training due to its scale. Binary variables (`Diabetes`, `Hypertension`, `CKD_Status`, `Dialysis_Needed`) were not scaled as they were already in a suitable format (0 or 1).
*   **Outlier Handling:** Outliers in the scaled numerical columns were identified using the Interquartile Range (IQR) method and handled by capping the values at the calculated lower and upper bounds (Q1 - 1.5\*IQR and Q3 + 1.5\*IQR). This approach mitigates the influence of extreme values without removing data points.
*   **Data Splitting:** The preprocessed data was split into training (80%) and testing (20%) sets using `train_test_split`, with 'CKD\_Status' as the target variable. Stratification was applied to ensure that the proportion of CKD cases was maintained in both the training and testing sets.

## 5. Model Selection and Training

Several classification models suitable for binary classification were selected and trained on the training data:

*   **Logistic Regression:** A simple linear model for its efficiency and interpretability.
*   **Support Vector Machine (SVM):** Effective for finding a clear margin of separation.
*   **Random Forest:** An ensemble method known for its robustness and handling of non-linear relationships.
*   **Gradient Boosting (GradientBoostingClassifier):** Another powerful ensemble method.
*   **K-Nearest Neighbors (KNN):** A non-parametric instance-based learning algorithm.

Each model was instantiated with `random_state=42` for reproducibility (where applicable) and trained on the `X_train` and `y_train` datasets.

## 6. Model Evaluation (Initial)

The performance of each trained model was evaluated on the held-out testing data (`X_test`, `y_test`) using common classification metrics: Accuracy, Precision, Recall, and F1-score.

The initial evaluation results were:

*   **Logistic Regression:** Accuracy: 0.8330, Precision: 0.8435, Recall: 0.8255, F1-score: 0.8344
*   **Support Vector Machine:** Accuracy: 0.9610, Precision: 0.9578, Recall: 0.9660, F1-score: 0.9619
*   **Random Forest:** Accuracy: 1.0000, Precision: 1.0000, Recall: 1.0000, F1-score: 1.0000
*   **Gradient Boosting:** Accuracy: 1.0000, Precision: 1.0000, Recall: 1.0000, F1-score: 1.0000
*   **K-Nearest Neighbors:** Accuracy: 0.8850, Precision: 0.8889, Recall: 0.8851, F1-score: 0.8870

Based on the F1-score, Random Forest and Gradient Boosting appeared to be the best performing models, achieving perfect scores. However, perfect scores on a test set can sometimes indicate overfitting or potential data leakage.

## 7. Hyperparameter Tuning and Cross-Validation

To potentially improve the performance of the best performing model (Random Forest) and obtain a more reliable estimate of its performance, hyperparameter tuning was performed using `GridSearchCV` with 5-fold cross-validation.

The parameter grid for Random Forest included `n_estimators`, `max_depth`, `min_samples_split`, and `min_samples_leaf`. `GridSearchCV` was fitted to the training data (`X_train`, `y_train`) using 'f1' as the scoring metric and `n_jobs=-1` for parallel processing.

The best hyperparameters found for Random Forest were: `{'max_depth': None, 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 100}`.

Cross-validation was also performed on the Random Forest model using KFold with 5 splits, shuffling, and a random state. The cross-validation results for different metrics were:

*   **F1-score:** Mean = 0.9995, Std = 0.0011
*   **Accuracy:** Mean = 0.9995, Std = 0.0011
*   **Precision:** Mean = 1.0000, Std = 0.0000
*   **Recall:** Mean = 0.9989, Std = 0.0021

These cross-validation scores further support the high performance of the Random Forest model on the training data.

It's important to note that Random Forest models do not directly use L1/L2 regularization in the same way as linear models like Logistic Regression. Overfitting in Random Forests is typically managed through hyperparameters like `max_depth`, `min_samples_split`, and `min_samples_leaf`, which were included in the tuning process.

## 8. Final Model and Results

A final Random Forest model was instantiated with the best hyperparameters found through `GridSearchCV` and trained on the entire training dataset. This final model was then evaluated on the held-out testing dataset.

The final performance metrics for the tuned Random Forest model on the test set were:

*   **Accuracy:** 1.0000
*   **Precision:** 1.0000
*   **Recall:** 1.0000
*   **F1-score:** 1.0000

## 9. Conclusion and Future Work

The project successfully explored the Chronic Kidney Disease dataset, performed necessary data preprocessing, and trained and evaluated several classification models. The Random Forest model, both initially and after hyperparameter tuning, achieved perfect performance metrics on the test set.

While the perfect scores are promising, they warrant further investigation to rule out potential overfitting or data characteristics that lead to such high separability.

Future work could include:

*   **Further Data Investigation:** A deeper dive into the dataset to understand why the models achieve perfect scores on the test set and confirm the absence of data leakage.
*   **Model Interpretability:** Exploring techniques to understand which features are most important for the Random Forest model's predictions.
*   **Deployment:** If the perfect scores are validated and the model's reliability is confirmed, the next step could involve deploying the model to predict CKD risk on new, unseen data.
*   **Exploring Other Algorithms:** Although Random Forest performed perfectly, evaluating other algorithms with more extensive tuning or ensemble methods could provide alternative perspectives or potentially more robust models in different scenarios.

This project demonstrates a standard machine learning workflow, from data loading and exploration through preprocessing, modeling, evaluation, and tuning, culminating in a high-performing predictive model for CKD status based on the provided dataset.
