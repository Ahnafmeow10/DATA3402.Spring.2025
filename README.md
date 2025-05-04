## Bank Customer Churn Prediction

This project is from a Kaggle challenge called [Playground Series - Season 4, Episode 1](https://www.kaggle.com/competitions/playground-series-s4e1).

This project is part of the Kaggle Playground Series - Season 4, Episode 1. The goal is to predict customer churn — whether a customer will leave a bank or not. The dataset includes features such as credit score, age, country, gender, tenure, account balance, number of products, credit card ownership, activity status, and estimated salary. The target variable, "Exited", indicates whether a customer has left (1) or stayed (0). By training a machine learning model on this data, we aim to identify customers who are likely to leave, helping the bank take proactive steps to improve customer retention.


## Data Loading and Initial Look

We used `pandas` to load the `train.csv` and `test.csv` files and inspected the number of rows, columns, and feature types. We checked for missing values and created a table describing each feature's type, range, and outliers.

## Data Visualization

We compared the distribution of each feature across the two classes: customers who stayed (`Exited = 0`) and those who left (`Exited = 1`). Histograms were used for numerical features, and bar plots for categorical ones. This helped identify that age, balance, number of products, and geography were important features for predicting churn.
![image](https://github.com/user-attachments/assets/50a247c0-eec5-4de6-8391-c65db22544f8)
![image](https://github.com/user-attachments/assets/eecf5ede-3073-4d00-88f7-b7c340276436)
![image](https://github.com/user-attachments/assets/a9d087f9-5514-4691-b703-b27e9b0d8f9f)
![image](https://github.com/user-attachments/assets/117d76bb-e26f-4b02-ae6a-032a1498c006)
![image](https://github.com/user-attachments/assets/3851043a-e4c1-4660-9271-4b58a9012641)
![image](https://github.com/user-attachments/assets/1e486b10-8228-4639-a23e-a3d4bd1c3944)
![image](https://github.com/user-attachments/assets/960dcac3-4d2e-480e-aa95-cfc8e07c2024)
![image](https://github.com/user-attachments/assets/ee30e1d0-f768-4ec8-9ced-e3dd2d2ea78f)






## Data Cleaning and Preparation for Machine Learning

We removed non-useful columns like IDs and names. Categorical variables (`Gender`, `Geography`) were one-hot encoded. Numerical features were scaled using `StandardScaler` to prepare for modeling. We visualized each feature before and after scaling to confirm the transformation worked correctly.

![image](https://github.com/user-attachments/assets/046cc83e-49a6-4325-9d4c-2b20eb7ba46d)
![image](https://github.com/user-attachments/assets/467e9d0c-67cb-4ad6-a7f1-f5f3822f1130)
![image](https://github.com/user-attachments/assets/a782dbd9-3c40-4f78-bcfe-4bf744224b90)
![image](https://github.com/user-attachments/assets/2ca022bc-d743-4a6a-bdbe-0b59983af898)
![image](https://github.com/user-attachments/assets/9598f24f-2f1a-420b-96fc-d8b12265ec3b)
![image](https://github.com/user-attachments/assets/10a2c0b7-2c5d-4620-af09-d887dd5903a6)






## Machine Learning

We used a Random Forest Classifier. The data was split into training, validation, and test sets. The model was trained on the training set, and its performance was evaluated using accuracy and a classification report. Finally, predictions were made on the Kaggle test set, and the results were saved to `submission.csv`.

Validation Accuracy: 0.8590991718844678

Classification Report:
               precision    recall  f1-score   support

           0       0.88      0.95      0.91     19517
           1       0.73      0.54      0.62      5238

    accuracy                           0.86     24755
   macro avg       0.81      0.74      0.77     24755
weighted avg       0.85      0.86      0.85     24755
Hold-out Test Accuracy: 0.8572063338180643

The model achieved a **validation accuracy of 85.9%**, a **hold-out test accuracy of 85.7%**, and a **weighted F1-score of 85%**, demonstrating consistent overall performance. It performs well in predicting retained customers but has lower recall for churned customers, indicating room for improvement in identifying churn.


## Files

- `final.ipynb`: Complete notebook with code, graphs, and explanations  
- `README.md`: Project overview and instructions  


## Libraries Used

- pandas  
- numpy  
- seaborn  
- matplotlib  
- scikit-learn  



