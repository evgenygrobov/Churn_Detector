# Prediction of customer churn
### (supervised tree based machine learning modeling on binary classification target) 

---

### Project description:
Churn, or attrition, is a metric that measures how much business you’ve lost. This loss can be in:
- percent of customers – this is called customer churn and measures how many customers you’ve lost in a given period;
- or in revenue – this is called revenue churn and measures the money you’ve lost in a given period. It’s important to note that revenue churn doesn’t only measure the money lost due to customers leaving you. 

---

### Data

The data was taken here https://www.kaggle.com/mathchi/churn-for-bank-customers. It is open source public Kaggle website.
***Target is a binary classificition labled as class 1 and class 0 for customers who left and retained in bank respectively. Predictors are: [CreditScore', 'Gender', 'Age', 'Tenure', 'Balance', 'NumOfProducts','HasCrCard', 'IsActiveMember', 'EstimatedSalary', 'Exited']***
       

I wish to perform supervised learning on a bynary classification problem to determine if an existng customer is going to leave the Bank. The Customer churn dataset consists of 10000 unique observations, each labelled as loyal customer (or not exiting) (0) or one who is not loyal and leaving the bank (1). The data also contains a number of predictors (12), each of which is either a social  metrics, or a customer experience history with the Bank.

It is might be interesting to see some inside from the data. First, lets take a look at the ratio of the customer churn.

![](https://github.com/evgenygrobov/Customer-churn-prediction/blob/main/images/pie_chart.png)


***The ratio is not much awfull. 1:4 of left and retained customer.***

---

**We could notice that Age and Balance are the most relevant features to label.**

![](https://github.com/evgenygrobov/Customer-churn-prediction/blob/main/images/correl.png)

On the scatter plot you can see orange crosses = class 1(churn) and blue circles = class 0(retained customers).

![](https://github.com/evgenygrobov/Customer-churn-prediction/blob/main/images/AGE%7CBalance.png)

Bar plots depict the number of products distribution amongst customer class 0 and class 1.

![](https://github.com/evgenygrobov/Customer-churn-prediction/blob/main/images/Custome%7CProducts.png)

---

***Breifly i could conclude that major group of customers who left the bank(40%) were middle age beetwen 36 and 52 and had account balance>$80K.***


## Machine learning modeling

Evaluating model performance can tell us if our approach is working. Two diagnostic tools that help in the interpretation of probabilistic forecast for binary (two-class) classification predictive modeling problems are ROC Curves and Precision-Recall curves.
I will use repeated cross-validation to evaluate the models, with three repeats of 10-fold cross-validation. The mode performance will be reported using the mean ROC area under curve (ROC AUC) averaged over repeats and all folds.

---

***ROC_AUC_score*** (it turns out Logistic Regression, K-neighbor and Decision Tree models are doing a poor job by misclassifying label)


![](https://github.com/evgenygrobov/Customer-churn-prediction/blob/main/images/ROC_allmodel.png)

---

***Threshold for ROC Curve***. The curve is useful to understand the trade-off in the true-positive rate and false-positive rate for different thresholds


![](https://github.com/evgenygrobov/Customer-churn-prediction/blob/main/images/ROCbest.png)


One approach would be to test the model with each threshold returned from the call roc_auc_score() and select the threshold with the largest G-Mean value. Given that we have already calculated the Sensitivity (TPR) and the complement to the Specificity when we calculated the ROC Curve, I can calculate the G-Mean for each threshold directly: ***gmeans = sqrt(tpr * (1-fpr))***

![](https://github.com/evgenygrobov/Customer-churn-prediction/blob/main/images/ROCbestzoom.png)


***We can see that the point for the optimal threshold is a large black dot and it appears to be closest to the top-left of the plot.***

***Optimal Threshold for Precision-Recall Curve***
Unlike the ROC Curve, a precision-recall curve focuses on the performance of a classifier on the positive (minority class) only. Precision is the ratio of the number of true positives divided by the sum of the true positives and false positives. It describes how good a model is at predicting the positive class. Recall is calculated as the ratio of the number of true positives divided by the sum of the true positives and the false negatives. Recall is the same as sensitivity.
As in the previous section, the naive approach to finding the optimal threshold would be to calculate the F-measure for each threshold. We can achieve the same effect by converting the precision and recall measures to F-measure directly; for example:

![](https://github.com/evgenygrobov/Customer-churn-prediction/blob/main/images/F-score.png)

---

***Threshold setting and tunning***

---

### Classification report

![](https://github.com/evgenygrobov/Customer-churn-prediction/blob/main/images/class_report.png)

---

***Models prediction***

![](https://github.com/evgenygrobov/Customer-churn-prediction/blob/main/images/Models%20prediction.png)


---

## Conclusion

A good model will have a high level of true positive and true negatives, because these results indicate where the model has got the right answer. - 276 predicted customers will left the bank, that were actually left (TPs);
- 117 predicted retain with the bank that were actually exiting (FNs);
- 266 predicted left that were actually not (FPs); and
- 1341 predicted retain that were actually retain (TNs).

So in summary, out of 2000 test cases, we observed (considering a “positive” result as being a churn and a “negative” one being not churn).

It is a trade off here. Final decision after business what to prefer here either  more acccurate prediction on customers who are leaving  and save them or more accurate prediction on customer who is going to retain. 

