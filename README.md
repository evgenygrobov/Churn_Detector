# Prediction of customer churn
### (supervised  machine learning modeling on binary classification target) 

---

### Project description:
Churn, or attrition, is a metric that measures how much business you’ve lost. This loss can be in:
- percent of customers – this is called customer churn and measures how many customers you’ve lost in a given period;
- or in revenue – this is called revenue churn and measures the money you’ve lost in a given period. It’s important to note that revenue churn doesn’t only measure the money lost due to customers leaving you. 

---

### Data

The data was taken here https://www.kaggle.com/mathchi/churn-for-bank-customers. It is open source public Kaggle website.
***Target is a binary classificition labled as class 1 and class 0 for customers who left and retained in bank respectively. Predictors are: [CreditScore', 'Gender', 'Age', 'Tenure', 'Balance', 'NumOfProducts','HasCrCard', 'IsActiveMember', 'EstimatedSalary', 'Exited']***
       

I wish to perform supervised learning on a bynary classification problem to determine if an existng customer is going to leave the Bank. The Customer churn dataset consists of 10000 unique observations, each labeled as loyal customer (or not exiting) (0) or one who is not loyal and leaving the bank (1). The data also contains a number of predictors (12), each of which is either a social  metrics, or a customer experience history with the Bank.

It is might be interesting to see some inside from the data. First, lets take a look at the ratio of the customer churn.

![](https://github.com/evgenygrobov/Customer-churn-prediction/blob/main/images/pie_chart.png)


***The ratio is not much awfull. 1:4 of left and retained customer.***

---

**We can observe Age, Balance and IsActiveMember corespond to the target label.**

![](https://github.com/evgenygrobov/Customer-churn-prediction/blob/main/images/correl.png)

On the scatter plot you can see orange crosses = class 1(churn) and blue circles = class 0(retained customers).

![](https://github.com/evgenygrobov/Customer-churn-prediction/blob/main/images/AGE%7CBalance.png)

Bar plots depict the number of products distribution amongst customer class 0 and class 1.

![](https://github.com/evgenygrobov/Customer-churn-prediction/blob/main/images/Custome%7CProducts.png)

---

## Machine learning modeling

### Choosing model to perform churn prediction.
Evaluating model performance can tell us if our approach is working. TWo diagnistic tools that help in interpretation of probabolistic forecast for binary
classification predictive mideling problems are ROC curves and Precision Recall scores.

![](https://github.com/evgenygrobov/Customer-churn-prediction/blob/main/images/AllModelsROC.png)

---

Random Forest algorith yielded higher accurancy and roc_auc score amongst other models. Looks like RF is a good candidate to predict churn.

Lets first plot the confusion matrix to take a look at model perfomance.

![](https://github.com/evgenygrobov/Customer-churn-prediction/blob/main/images/MatrixBefore.png)


---

Sensitivity=TP/TP+FN, Specificit=1-FP/TN

Sensitivity=82.4%, Specificity=97.9%. That is great, however we see some classes were missclassified as False Negative.

Were Missed= 245/386=63.4%, Total error=275/1875=14.6%. This is not well. 


Since we mostly interested in label 1 predicition which is actuall churn, we should scrutiny on the Precision and Recall.

Precision = TP/TP + FP, whereas  Recall= TP/FN + TP.

Precision = 82.4%, Recall = 36.5%

In this scenario, I build Customer Churn Detector where positive class is churn. I needed to decide how to oprimize the model perfomance this is because false positive(loyal clients that are flagged as possible churn) are more acceptible than false negative(churn not detected).

Lets see Precision_Recall curve 

![]()

---

***Optimal threshold for ROC Curve***. 
The curve is useful to understand the trade-off in the true-positive rate and false-positive rate for different thresholds. One approach would be to test the model with each threshold returned from the call roc_auc_score() and select the threshold with the largest G-Mean value. Given that we have already calculated the Sensitivity (TPR) and the complement to the Specificity when we calculated the ROC Curve, I can calculate the G-Mean for each threshold directly: ***gmeans = sqrt(tpr * (1-fpr))***


![](https://github.com/evgenygrobov/Customer-churn-prediction/blob/main/images/ROCbest.png)


## Lets zoom it up!


![](https://github.com/evgenygrobov/Customer-churn-prediction/blob/main/images/ROCbestzoom.png)


***We can see the optimal threshold is a large black dot and it appears to be closest to the top-left of the plot.***


***Optimal Threshold for Precision-Recall Curve***
Unlike the ROC Curve, a precision-recall curve focuses on the performance of a classifier on the positive (minority class) only. Precision is the ratio of the number of true positives divided by the sum of the true positives and false positives. It describes how good a model is at predicting the positive class. Recall is calculated as the ratio of the number of true positives divided by the sum of the true positives and the false negatives. Recall is the same as sensitivity.
The naive approach to finding the optimal threshold would be to calculate the F-score for each threshold. 


![](https://github.com/evgenygrobov/Customer-churn-prediction/blob/main/images/F-score.png)

---

***Threshold setting and tunning***(Now I can tune models hypermaremters to adjust default decision boundary when appropriate.)


---

### Confusion matrix


![]()




---

## Conclusion

A good model will have a high level of true positive and true negatives, because these results indicate where the model has got the right answer.
The target of the project is to predict customer churn from the bank, espessialy for ***class 1*** labeled customers. 
The models obviousely learnt how to disitinguish between classes and detect churn. If i would be the owner of the project I will pick model that yields better balance between TP and TN ***class1*** . 


P.S. Of cause the final decision is after business what trade off between TPs,TNs to pick and which model meets the goals the best.

