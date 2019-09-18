# Predicting-Ferry-Delays
This is the repository used for collaboration on the solution for the CANSSI National Case Study Competition 2019. More info can be found at http://www.canssi.ca/news-events/canssi-datathon-2019/

Idea
-
The problem outlined in the case is a classification problem, which means the output is a discrete value. In this case 1 (delayed) or 0 (not delayed).

A technique common for this type of problem is logistic regression. The idea behind it is similar to linear regression, but to activate the continuous output from that into a real number between 0 and 1 to represent a probability function. 

Assumptions
-
* The features we are using to predict the dependant variable are independant.
* Linear relationship
* Multivariate normality
* No or little multicollinearity
* No auto-correlation
* Homoscedasticity
* A note about sample size.  In Linear regression the sample size rule of thumb is that the regression analysis requires at least 20 cases per independent variable in the analysis.

Things to do:
- Try and finish the np tutorial and output a proper file in the competition format
- In Trip.Duration Column there are N/A values. For now we are excluding them but maybe there is another way to handle it.
- Research the bridge traffic dataset and the rules of linear regression to see how to clean and use it in the training data
- Research the two weather datasets to see how it can be used to help train the model


