# Prediction of customer churn
### (supervised tree based machine learning modeling on binary classification target) 

---

### Project description:
Churn, or attrition, is a metric that measures how much business you’ve lost. This loss can be in:
- percent of customers – this is called customer churn and measures how many customers you’ve lost in a given period;
- or in revenue – this is called revenue churn and measures the money you’ve lost in a given period. It’s important to note that revenue churn doesn’t only measure the money lost due to customers leaving you. 

---

## Data

The data was taken here https://www.kaggle.com/mathchi/churn-for-bank-customers. It is open source public Kaggle website.
***Target is a binary classificition labled as class 1 and class 0 for customers who left and retained in bank respectively. Predictors are: [CreditScore', 'Gender', 'Age', 'Tenure', 'Balance', 'NumOfProducts','HasCrCard', 'IsActiveMember', 'EstimatedSalary', 'Exited']***
       

I wish to perform supervised learning on a bynary classification problem to determine if an existng customer is going to leave the Bank. The Customer churn dataset consists of 10000 unique observations, each labelled as loyal customer (or not exiting) (0) or one who is not loyal and leaving the bank (1). The data also contains a number of predictors (12), each of which is either a social  metrics, or a customer experience history with the Bank.

It is might be interesting to see some inside from the data. First, lets take a look at the ratio of the customer churn.

![](https://github.com/evgenygrobov/Customer-churn-prediction/blob/main/images/pie_chart.png)


The ratio is not much awfull. 1:4 of left and retained customer. 

---

We could notice that Age and Balance are the most relevant features to label.

![](https://github.com/evgenygrobov/Customer-churn-prediction/blob/main/images/correl.png)

On the scatter plot you can see orange crosses = class 1(churn) and blue circles = class 0(retained customers).

![](https://github.com/evgenygrobov/Customer-churn-prediction/blob/main/images/AGE%7CBalance.png)

Bar plots depict the number of products distribution amongst customer class 0 and class 1.

![](https://github.com/evgenygrobov/Customer-churn-prediction/blob/main/images/Custome%7CProducts.png)

---

***Breifly i could conclude that major group of customers who left the bank(40%) were middle age beetwen 36 and 52 and had account balance>$80K.***


## Machine learning modeling

Evaluating model performance can tell us if our approach is working — this turns out to be helpful.Monitoring model performance on a validation set is an excellent way to get feedback on whether what you’re doing is working. It’s also a great tool for comparing different models — ultimately, our aim is to build better, more accurate models that will help us make better decisions in real world applications.

---

## 1.Base data set

---

### ROC_AUC score

![](https://github.com/evgenygrobov/Customer-churn-prediction/blob/main/images/ROC_allmodel.png)

---

**Precision-Recall is a useful measure of success of prediction when the classes are very imbalanced. In information retrieval, precision is a measure of result relevancy, while recall is a measure of how many truly relevant results are returned.**

### Classification report

![](https://github.com/evgenygrobov/Customer-churn-prediction/blob/main/images/CLASSReportbase.png)


---

**A good model will have a high level of true positive and true negatives, because these results indicate where the model has got the right answer. A good model will also have a low level of false positives and false negatives, which indicate where the model has made mistakes. These four numbers can tell us a lot about how the model is doing and what we can do to help. Often, it’s helpful to represent them as a confusion matrix.**

### Best result: XGBC model confusion matrix.

![](https://github.com/evgenygrobov/Customer-churn-prediction/blob/main/images/CMbase.png)

---

The columns of this matrix represent what our model has predicted — no customer churn on the left and customer churn on the right. The rows represent what each instance that the model predicted actually was — no customer churn on the top and customer churn on the bottom. The number in each position tells us the number of each situation that was observed when comparing our predictions to the actual results.

---

## Summary for base(imbalanced) data set:

So in summary, out of 2000 test cases, we observed(considering a “positive” result as being a churn and a “negative” one being not churn ):
- 276 predicted customers will left the bank, that were actually left (TPs);
- 117 predicted retain with the bank that were actually exiting (FNs);
- 266 predicted left that were actually not (FPs); and
- 1341 predicted retain that were actually retain (TNs).

---

## 2. Resampled(over) data set.(SMOT)

Skewed datasets are not uncommon. And they are tough to handle. This is because if the dataset is skewed, such as in our example, a 1:4 ratio of Positives to Negatives occur. I have decided to apply resampling technics such as SMOT and setting some hyperparameters cleverly to using libraries that contain different versions of the usual algorithms which internally handle imbalance themselves. 

---

### ROC_AUC score

---

![](https://github.com/evgenygrobov/Customer-churn-prediction/blob/main/images/SMOT_ROC.png)


---

### Classification report

---

![](https://github.com/evgenygrobov/Customer-churn-prediction/blob/main/images/SMOT%20class_report.png)


---

### Confusion matrix

---

![](https://github.com/evgenygrobov/Customer-churn-prediction/blob/main/images/SMOT_conf_plot.png)


---

## Conclusion

A good model will have a high level of true positive and true negatives, because these results indicate where the model has got the right answer. A good model will also have a low level of false positives and false negatives, which indicate where the model has made mistakes. These four numbers can tell us a lot about how the model is doing and what we can do to help. Often, it’s helpful to represent them as a confusion matrix.

So in summary, out of 2000 test cases, we observed (considering a “positive” result as being a churn and a “negative” one being not churn).

It is a trade off here. Final decision after business what to prefer here either  more acccurate prediction on customers who are leaving  and save them or more accurate prediction on customer who is going to retain. 

