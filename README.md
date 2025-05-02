## Bank Customer Churn Prediction

This project is from a Kaggle challenge called [Playground Series - Season 4, Episode 1](https://www.kaggle.com/competitions/playground-series-s4e1).

The goal is to predict if a customer will leave a bank or not. This is called customer churn. We are given a dataset with information about each customer, like their credit score, age, country, gender, how long they’ve been with the bank, and their account details. The column we want to predict is called `Exited`. It shows 1 if the customer left, and 0 if they stayed. We train a machine learning model using this data and make predictions for new customers.

## Data Loading and Initial Look

We used `pandas` to load the `train.csv` and `test.csv` files and inspected the number of rows, columns, and feature types. We checked for missing values and created a table describing each feature's type, range, and outliers.

## Data Visualization

We compared the distribution of each feature across the two classes: customers who stayed (`Exited = 0`) and those who left (`Exited = 1`). Histograms were used for numerical features, and bar plots for categorical ones. This helped identify that age, balance, number of products, and geography were important features for predicting churn.

## Data Cleaning and Preparation for Machine Learning

We removed non-useful columns like IDs and names. Categorical variables (`Gender`, `Geography`) were one-hot encoded. Numerical features were scaled using `StandardScaler` to prepare for modeling. We visualized each feature before and after scaling to confirm the transformation worked correctly.

## Machine Learning

We used a Random Forest Classifier. The data was split into training, validation, and test sets. The model was trained on the training set, and its performance was evaluated using accuracy and a classification report. Finally, predictions were made on the Kaggle test set, and the results were saved to `submission.csv`.

## Files

- `final.ipynb`: Complete notebook with code, graphs, and explanations  
- `README.md`: Project overview and instructions  


## Libraries Used

- pandas  
- numpy  
- seaborn  
- matplotlib  
- scikit-learn  



