# Loan Delinquency Prediction

## Introduction
<p align="justify">A loan delinquency prediction is essential for financial institutions and lenders to identify potential risks
and minimize losses. By accurately predicting the likelihood of a borrower defaulting on a loan, lenders
can take proactive measures to prevent delinquency and safeguard their investments. </p>

<p align="justify">The primary motivation for conducting a loan delinquency prediction is to improve the credit risk
assessment process. With the advent of big data and advanced analytics techniques, lenders can now
leverage a vast amount of information to create predictive models that can forecast the likelihood of
delinquency. By using machine learning algorithms and statistical techniques, lenders can build models
that take into account a variety of factors, such as borrower credit history, loan amount, and repayment
term, to provide more accurate predictions. Another is to improve overall loan portfolio management. By
identifying high-risk borrowers early on, lenders can take proactive measures to minimize delinquencies
and defaults. This can include offering alternative repayment plans, providing financial education
resources, or implementing stricter loan approval criteria. Ultimately, this can help lenders to better
manage their loan portfolio and improve their overall profitability. </p>

## Dataset
### [LendingClub Issued Loans](https://drive.google.com/file/d/1IWwhz41P_gGFczo2G5D6iH4dpsxFb2mg/view?usp=share_link)
<p align="justify"> The dataset was distributed by LendingClub which is a US peer-to-peer lending company. It contains over
750,000 rows and 72 columns and the data spans from 2007 to 2017. There are several attributes
including ID, loan amount, delinquency, loan status, and other information related to loan repayment. In
addition to this, there are other features which include credit score, funded amount, annual income,
employment information, etc. </p>

### Census Data
<p align="justify"> The census dataset contains over 500 features. It will be used together with the loan data to draw further
insights using spatial plots and to analyze patterns within the dataset. The Census Data API is used to
query data about the boundaries/shapes of a particular region. </p>

## EDA
<p align="justify"> Standard EDA techniques are applied to visualize the relationship between the Loan Status and
other attributes related to a customer. We developed visualizations for an easy understanding between
various factors/relationships that might affect slow repayment of the loan and investigate the impact of
regional economic conditions like poverty/median income on loan repayment. Loan Status is
geographically visualized based on the regional information of the customers.
We used Seaborn and Matplotlib to plot visualizations between various variables. </p>

<p align="justify"> Not surprisingly, the number of higher-grade loans exceeds that of lower-grade loans [Fig 3]. The lending data shows lower
grade loans tend to have higher default rates [Fig 4]. The default rates were unusually high for the years
2007-2009. This likely indicates the impact of the 2008 Global Economic crisis [Fig 5].
In order to derive additional insights, we enriched the lending data with Census data from ACS
(2015-2019) and visualized the relationship between default rates of a given geographical region with the
estimated Median Income of that particular region. Default rates are skewed around 8-10% for most US
Zip codes [Fig 6]. On careful observation, we can identify that the low-income regions (southern states of
Mississippi, Oklahoma, parts of Arizona/New Mexico, and the Appalachian region) have higher default
rates [Fig 7]. The lower-income counties of Western Massachusetts have proportionally high default rates
[Fig 8]. </p>

## Modelling
<p align="justify"> we implemented classification models using loan data to predict Loan Delinquency
Probability. We only considered a few factors for the modeling like loan amount, term, interest rate,
subgrade, number of credit lines, number of mortgage accounts, loan purpose, verification type,
application type, and home ownership. Random Forest, XGBoost, and Naive Bayes models were applied
and their performance was evaluated using metrics such as accuracy, precision, recall, and F1-score.
The XGBoost model provides an accuracy of 88.93%, the random forest (with 100 estimators) gives
88.89% and the Gaussian Naive Bayes Model gives 86.40% of accurate classification. The XGBoost
model gives the best accuracy on the testing data. </p>
