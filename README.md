# Customer-Churn-Prediction

🧾 Project Overview – Customer Churn Prediction

Objective:
Build a machine learning classification model to predict whether a customer will churn (i.e., stop using the service), using demographic, account, and service-related data.

📊 Dataset Description

    Dataset Name: Customer_data

    Features Include:

        Demographics: Gender, SeniorCitizen, Tenure, etc.

        Services: InternetService, PhoneService, OnlineSecurity, etc.

        Account Info: Contract type, MonthlyCharges, TotalCharges

    Target Variable: Churn (Yes/No)

# Distribution of Customer Churn

<img width="616" height="443" alt="image" src="https://github.com/user-attachments/assets/e901a3b2-454d-41b4-b7ad-fd5f7d607179" />

This bar plot shows the distribution of customers who churned vs. those who stayed.

    No (Non-Churned): Majority of customers stayed with the company.

    Yes (Churned): A smaller portion of customers left.

    This imbalance highlights the need to handle class imbalance during model training to avoid biased predictions.
    
# Total Charges vs. Churn Status

<img width="625" height="448" alt="image" src="https://github.com/user-attachments/assets/29017e0c-2b3d-4870-874b-cfa96a9b7703" />

This histogram shows the distribution of TotalCharges grouped by churn status:

    Customers with lower total charges are more likely to churn (orange bars are higher on the left).

    Customers who stayed (No) are spread across a wider range of total charges, especially in higher amounts.

# Correlation Matrix

<img width="595" height="465" alt="image" src="https://github.com/user-attachments/assets/db1d2b87-1521-4de3-9c7d-de275db25447" />

This heatmap shows the correlation between numerical features:

    TotalCharges is highly correlated with tenure (0.83) and MonthlyCharges (0.65), which is expected since it’s a product of both over time.

    SeniorCitizen has weak correlations with all other variables.

🧹 Data Preprocessing

    Dropped customerID column as it doesn't contribute to prediction.

    Label Encoding:
    Converted categorical columns (like gender, InternetService, Contract, etc.) into numerical values using LabelEncoder, including the target column Churn.

    Feature Scaling:
    Standardized the feature values using StandardScaler for better model performance.

🤖 Model Development – Logistic Regression

    Data Split:
    Split the dataset into training (70%) and testing (30%) sets using train_test_split.

    Model:
    Trained a LogisticRegression model.

    Results:

        Accuracy: 79.91%

        Precision/Recall (Churn = 1):

            Precision: 0.64

            Recall: 0.53

            F1-Score: 0.58

<img width="530" height="392" alt="image" src="https://github.com/user-attachments/assets/1e26ed57-5c44-4879-a52b-9002afff953c" />

    The model performs well for predicting non-churn (Churn=0) but can be further improved for churned customers (Churn=1) using techniques like SMOTE, feature selection, or trying other models like Random Forest or XGBoost.

# Decision Tree Algorithm
    Training: Used standardized training data from earlier preprocessing.

    Results:

        Accuracy: 72.42%

        Performance on Churn Class (1):

            Precision: 0.48

            Recall: 0.50

            F1-Score: 0.49

<img width="524" height="391" alt="image" src="https://github.com/user-attachments/assets/af1f3dd9-6563-4eac-a65d-713e79579934" />

    The Decision Tree model shows lower accuracy and recall for churned customers compared to Logistic Regression. It may be prone to overfitting and needs tuning (e.g., max_depth, min_samples_split) or alternative models like Random Forest for better performance.

# Random Forest Algorithm

    Training: Performed on standardized data.

    Results:

        Accuracy: 79.05%

        Performance on Churn Class (1):

            Precision: 0.63

            Recall: 0.48

            F1-Score: 0.55

    The Random Forest model performs comparably to Logistic Regression in terms of accuracy but still struggles to recall churned customers. Hyperparameter tuning (e.g., via GridSearchCV) may further improve performance.    

# Support Vector Machine

    Training: Performed on standardized data.

    Results:

        Accuracy: 79.05%

        Performance on Churn Class (1):

            Precision: 0.63

            Recall: 0.48

            F1-Score: 0.55

    The Random Forest model performs comparably to Logistic Regression in terms of accuracy but still struggles to recall churned customers. Hyperparameter tuning (e.g., via GridSearchCV) may further improve performance.

# Summary of Accuracies

<img width="313" height="91" alt="image" src="https://github.com/user-attachments/assets/46b5b8f6-3623-474c-8a49-aedecdc94257" />

Logistic Regression achieved the highest accuracy, closely followed by SVM and Random Forest.
Decision Tree had the lowest accuracy and would benefit from tuning or ensemble methods.

# Hyperparameter Tuning (GridSearchCV)

Used GridSearchCV to optimize model parameters and improve performance through cross-validation (cv=5).
🔹 Logistic Regression

    Best Parameters:
    {'C': 0.1, 'solver': 'liblinear', 'max_iter': 100}

    Best Cross-Validated Accuracy: 80.37%

    C=0.1 applies stronger regularization. The liblinear solver works well for small datasets.

🔸 Decision Tree

    Best Parameters:
    {'criterion': 'gini', 'max_depth': 5, 'min_samples_leaf': 2, 'min_samples_split': 2}

    Best Cross-Validated Accuracy: 78.50%

    A shallow tree with max_depth=5 and more constrained splits helped avoid overfitting.

🌲 Random Forest

    Best Parameters:
    {'n_estimators': 200, 'max_depth': 10, 'min_samples_split': 10, 'min_samples_leaf': 1}

    Best Cross-Validated Accuracy: 80.72%

    A deeper ensemble model with many estimators and controlled splits produced the best overall accuracy.

📈 Support Vector Machine (SVM)

    Best Parameters:
    {'C': 0.1, 'kernel': 'linear', 'gamma': 'scale'}

    Best Cross-Validated Accuracy: 79.93%

    The linear kernel with small regularization worked best; suitable for linearly separable data.

Conclusion:

Random Forest gave the best cross-validated accuracy, but Logistic Regression and SVM were close behind with simpler and more interpretable models.

