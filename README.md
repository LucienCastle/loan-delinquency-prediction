# Customer Risk Assessment for Lending Clubs

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

<p align="justify"> Not surprisingly, the number of higher-grade loans exceeds that of lower-grade loans. The lending data shows lower
grade loans tend to have higher default rates. The default rates were unusually high for the years
2007-2009. This likely indicates the impact of the 2008 Global Economic crisis.
In order to derive additional insights, we enriched the lending data with Census data from ACS
(2015-2019) and visualized the relationship between default rates of a given geographical region with the
estimated Median Income of that particular region. Default rates are skewed around 8-10% for most US
Zip codes. </p>

[**Tableau Dashboard**](https://public.tableau.com/app/profile/sumit.patil5062/viz/USLoanDelinquencyAnalysis/Dashboard1)

## Modelling
<p align="justify"> We implemented classification models using loan data to predict Loan Delinquency
Probability. We only considered a few factors for the modeling like loan amount, term, interest rate,
subgrade, number of credit lines, number of mortgage accounts, loan purpose, verification type,
application type, and home ownership. Random Forest, XGBoost, and Naive Bayes models were applied
and their performance was evaluated using metrics such as accuracy, precision, recall, and F1-score.</p>

<p align="justify"> Moreover, we implemented Deep Neural Networks, to test if they excel at this task. The DNN was trained
for 20 epochs with a batch size of 32, a learning rate of 0.001, and an Adam optimizer</p>

![image](https://user-images.githubusercontent.com/47452095/233852341-56041af6-bcce-4540-91cf-be094db8b633.png)

## Metrics
- <p align="justify"> <b>Accuracy</b> measures the proportion of correct predictions made by the model out of all the predictions. It is calculated as the ratio of the number of correct predictions to the total number of predictions. </p>
- <p align="justify"> <b>Precision</b> measures the proportion of true positive predictions out of all the positive predictions made by the model. It is calculated as the ratio of true positives to true positives plus false positives. Precision is a useful metric when the cost of false positives is high. </p>
- <p align="justify"> <b>Recall</b> measures the proportion of true positive predictions out of all the actual positive instances. It is calculated as the ratio of true positives to true positives plus false negatives. Recall is a useful metric when the cost of false negatives is high. </p>
- <p align="justify"> <b>F1-score</b> is a harmonic mean of precision and recall, and it provides a balance between precision and recall. It is calculated as 2 * (precision * recall) / (precision + recall). F1-score is a useful metric when there is an uneven distribution of positive and negative instances in the dataset. </p>
- <p align="justify"> <b>ROC</b> (Receiver Operating Characteristic) curve is a graphical representation of the trade-off between true positive rate and false positive rate at various threshold settings. ROC score measures the area under the ROC curve, which indicates the model's ability to distinguish between positive and negative instances. ROC score is a useful metric when the cost of false positives and false negatives is relatively equal. </p>

## Results
<p align="jsutify"> The XGBoost model provides an accuracy of 88.93%, the random forest (with 100 estimators) gives
88.89% and the Gaussian Naive Bayes Model gives 86.40% of accurate classification. The XGBoost
model gives the best accuracy on the testing data. Whereas, ANN has the highest ROC score </p>

![image](https://user-images.githubusercontent.com/47452095/233852953-a4661e7c-ed8e-47a7-b99f-67537e56412e.png)


## Conclusion
A loan delinquency prediction project can have a significant return on investment for financial
institutions. By accurately predicting loan delinquency, lenders can take proactive measures to mitigate
losses and minimize the risk associated with lending money. This can result in a variety of financial
benefits, including:
- <p align="justify"> <b>Reduced loan losses:</b> Predicting loan delinquency can help lenders identify high-risk borrowers and take proactive measures to prevent loan defaults. This can significantly reduce the number of loans that go into default and ultimately lead to lower loan losses. </p>
- <p align="justify"> <b>Increased efficiency:</b> By automating the loan delinquency prediction process, lenders can reduce the amount of time and resources required to manage their loan portfolio. This can result in increased operational efficiency and lower costs. </p>
- <p align="justify"> <b>Improved customer satisfaction:</b> Accurately predicting loan delinquency can help lenders identify borrowers who are struggling to make payments and provide th em with additional support and resources to help them get back on track. This can improve customer satisfaction and loyalty. </p>

<p align="justify"> Overall, a loan delinquency prediction can provide financial institutions with a significant return on
investment by reducing loan losses, increasing efficiency, and improving customer satisfaction. </p>
