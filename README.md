# Customer churn out prediction

---

### Project description:
Churn, or attrition, is a metric that measures how much business you’ve lost. This loss can be in:
- percent of customers – this is called customer churn and measures how many customers you’ve lost in a given period;
- or in revenue – this is called revenue churn and measures the money you’ve lost in a given period. It’s important to note that revenue churn doesn’t only measure the money lost due to customers leaving you. 

---

## Data

Lets take a look at data more detailed.The data was taken here https://www.kaggle.com/mathchi/churn-for-bank-customers. It is open source public Kaggle website.

I wish to perform supervised learning on a bynary classification problem to determine if an existng customer is going to leave the Bank. The Customer churn dataset consists of 10000 unique observations, each labelled as loyal customer (or not exiting) (0) or one who is not loyal and leaving the bank (1). The data also contains a number of predictors (12), each of which is either a social  metrics, or a customer experience history with the Bank.

It is might be interesting to see some inside from the data. First, lets take a look at the ratio of the customer churn.

![](https://github.com/evgenygrobov/Customer-churn-prediction/blob/main/images/pie_chart.png)


The ratio is not much awfull. 1:4 of left and retained customer. EDA analysis reaveled what features are among the important answering who is leaving the Bank.

![](https://github.com/evgenygrobov/Customer-churn-prediction/blob/main/images/AGE%7CBalance.png)

Products. How many leaving without having a whole spectr of bank products.

![](https://github.com/evgenygrobov/Customer-churn-prediction/blob/main/images/NumOfProd.png)

---

## Machine learning

Skewed datasets are not uncommon. And they are tough to handle. This is because if the dataset is skewed, such as in our example, a 1:4 ratio of Positives to Negatives occur. I have decided to apply resampling technics such as SMOT and setting some hyperparameters cleverly to using libraries that contain different versions of the usual algorithms which internally handle imbalance themselves. 

## BaseLined machine learning.

---

### ROC_AUC score

![](https://github.com/evgenygrobov/Customer-churn-prediction/blob/main/images/ROC_AUC_base_score.png)

---

### Classification report

![](https://github.com/evgenygrobov/Customer-churn-prediction/blob/main/images/CLASSReportbase.png)


---

### Confusion matrix 

![](https://github.com/evgenygrobov/Customer-churn-prediction/blob/main/images/CMbase.png)


---

## SMOT  oversampling.

---

### ROC_AUC score

---

<image roc

---

### Classification report

---

<image class report

---

### Confusion matrix

---

<image cinfusion matrix

---

## Conclusion





