TIme_Series_Analysis


We determine whether the association between the response and each term in the model is statistically significant by comparing the p-value for the term to 95% significance level to assess the null hypothesis.  The first two models in the TIme_Series_Analysis notebook has very hight AICs and BICs. and majority of their p-values are way higher than 0.05(95%). 

We also look at the  Coef of each term to see if in any way it's close to 100%. in this case, only the moving average of the ARIMA model  and the Beta 1 of the Garch model have a high Coef close to 100%. 

Therefore all the terms, except the moving average of the ARIMA model  and the Beta 1 of the Garch model are not statistically significant.  I cannot conclude that the coefficients are statistically significant. We may want to refit the model without the statistically insignificant terms. 



1. Based on your time series analysis, would you buy the yen now?
   Volatility and returns tend to cluster. In the Garch model, the volativity will increase so the return will increase as well. Wec can proceeed to do any transaction on the JPY/USD trade.

2. Is the risk of the yen expected to increase or decrease? 
  The risk of yen is expected to increase as the violativity increase.
    
3. Based on the model evaluation, would you feel confident in using these models for trading?
Not
Not to rely on the ARMA and ARIMA mode. But the Garch model can be part of reference.

Linear Regression Forecasting

A model that is underfit will have high training and high testing error while an overfit model will have extremely low training error but a high testing error.

In this case the out-of-sample RMSE is 30% lower than the in-sample RMSE. RMSE is typically lower for training data, but is higher in this case. , which means the model makes some minor incorrect assumptions about the data, an example of a little underfitting. But overall, the model has low variance in both the training and test data, so the modle is good enough.