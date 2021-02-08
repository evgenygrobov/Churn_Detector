# Churn Detector
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

### EDA

It is might be interesting to see some inside from the data. First, lets take a look at the ratio of the customer churn.

![](https://github.com/evgenygrobov/Customer-churn-prediction/blob/main/images/pie_chart.png)


***The ratio is not much awfull. 1:4 of left and retained customer.***

---

**We can observe Age, Balance and IsActiveMember corespond to the target label.**

![](https://github.com/evgenygrobov/Customer-churn-prediction/blob/main/images/correl.png)

On the scatter plot you can see orange crosses = class 1(churn) and blue circles = class 0(retained customers).
And the box plots on the right. 

![](https://github.com/evgenygrobov/Customer-churn-prediction/blob/main/images/AGE%7CBalance.png)

Box plots depict the Age and Balance  distribution amongst customer class 0 and class 1.

---

## Machine learning modeling

### Choosing model to perform churn prediction.
Evaluating model performance can tell us if our approach is working. Two diagnistic tools that help in interpretation of probabolistic forecast for binary
classification predictive mideling problems are ROC curves and Precision Recall scores.

![](https://github.com/evgenygrobov/Customer-churn-prediction/blob/main/images/AllModelsROC.png)

---

Random Forest yielded higher accurancy, precision/recall and roc_auc score amongst other models. Looks like RF is a good candidate to predict churn.

Lets first plot the confusion matrix to take a look at model perfomance.

![](https://github.com/evgenygrobov/Customer-churn-prediction/blob/main/images/MatrixBefore.png)


---

Sensitivity=TP/TP+FN, Specificit=1-FP/TN

Sensitivity=82.4%, Specificity=97.9%. That is great, however we see some classes were missclassified as False Negative.

Were Missed= 245/386=63.4%, Total error=275/1875=14.6%. This is not well. 


Since we mostly interested in label 1 predicition which is actuall churn, we should scrutiny on the Precision and Recall.

Precision = TP/TP + FP, whereas  Recall= TP/FN + TP.

Precision = 82.4%, Recall = 36.5%

A good model will have a high level of true positive and true negatives, because these results indicate where the model has got the right answer.

Lets see Precision_Recall curve 

![](https://github.com/evgenygrobov/Customer-churn-prediction/blob/main/images/Threshold_Prec_Recall.png)

---

We know that class labels are imbalanced with the ratio 1:4. I need to set optimal decision threshold to keep RF from missclassifying labels.
I can set the new decision threshold with the RF roba_predict() method.


![](https://github.com/evgenygrobov/Customer-churn-prediction/blob/main/images/MatrixAfter.png)

Now we see neither errors, nor missed customers.

---
## Conclusion

The target of the project is to build Customer Churn Detector and predict probability of churn. 
In this scenario, I needed to decide how to oprimize the model perfomance, this is because false positive(loyal clients that are flagged as possible churn) are more acceptible than false negative(churn not detected). Random Forest is very powerfil tool with banch of hyperparameters to tune.
There are may be many other scenarios, with more compliated goals. The final decision of how model should be optimized is after businesses, of cause.

