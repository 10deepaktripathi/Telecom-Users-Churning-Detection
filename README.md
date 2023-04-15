# Telecom-Users-Churning-Detection

Below are the 2 main business objective i am trying to solve with this project-

1st: Predict behavior to retain telecom customers. Analyze all relevant customer data and develop focused customer retention programs. 

2nd: Identify which all attributes are important in deciding if a customer will  leave the service or not


INTRODUCTION

I will develop a classification model which can predict if a customer can leave the service in the future or not. To verify the performance of the model I will 
use following metrics.

Classification Metrics

• Accuracy

• Precision 

• Recall 

• F1-Score 

• Confusion Matrix 

Here, only accuracy is not a good matric to verify the performance of the model as data is highly imbalanced.
For our case, I think Precision is important, which says among total predicted for a category what percentage were correct predictions. This is due to that 
fact that if our model predicts someone will not leave the service and later if that persons leaves then it will be a loss for the company. 
Recall also seems a good metric which says of total data available for a certain category, what percentage has returned correctly. This is because if there are 
many people who will leave the service, but our model is not able to identify them, it will again be a big loss for the company. 
Since we want precision and recall both be high, I will check F1-core as well, which is the harmonic mean of both.
I will also use confusion matrix which can give detailed analysis of how model is performing by drawing a comparison between actual and predicted values in form of a matrix



Summary on what all I did:

• Upon loading the data, I performed few initial checks and found that data was highly imbalanced.

• It contained 5986 rows and 22 columns.

• It did not contain any null values.

• There were 3 numeric and 19 categorical columns.

• Used chi square test to check which all categorical columns are impacting categorical target ‘’Churn”.

• Used Annova test to check which all numerical columns are impacting categorical target ‘’Churn”.

• Used correlation and VIF to verify if there is any collinearity in data

• Tried standard scaling on numeric features and one hot encoding with categorical columns.

• Did a comparative study among multiple models. first with all columns and then with only selected columns (selected after chi square and annova test).

• Did hyperparameter tunning to get the best model possible.

• Tried deep learning model MLP on this data to check how does it perform.

• Eventually used Logistic regression and Gradient boosting to predict the test data and validate its performance.

• Used to embedded technique to get important features. I used Gradient boosting to get the important features


Please go through the notebook which is self explainaory.
