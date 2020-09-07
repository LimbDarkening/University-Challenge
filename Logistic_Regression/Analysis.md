# Analysis
The aim for this investigation was to derive the historical importance of score to the success of a team. Obviously one needs to outscore their opponent, but by how much? When can we say that a team has accrued enough points to have a greater than 50% chance of winning the match? How has this score changed over the seasons?
## Methodology
A suitable technique for this line of enquiry is linear logistic regression. We begin by stating the sigmoid function that will model the probability of winning the match, as a function of a team’s score, as    

<a href="https://www.codecogs.com/eqnedit.php?latex=P(z)&space;=&space;\frac{1}{1&plus;e^{-z}}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?P(z)&space;=&space;\frac{1}{1&plus;e^{-z}}" title="P(z) = \frac{1}{1+e^{-z}}" /></a>

Where z is given by the linear combination,

<a href="https://www.codecogs.com/eqnedit.php?latex=z&space;=&space;\beta&space;_{0}&plus;\beta&space;_{1}x" target="_blank"><img src="https://latex.codecogs.com/gif.latex?z&space;=&space;\beta&space;_{0}&plus;\beta&space;_{1}x" title="z = \beta _{0}+\beta _{1}x" /></a>

where x is a teams score.

We proceed to derive the coefficients beta naught and beta one by maximising the log likelihood for a Bernoulli random variable, since the labels we are predicting are binary in nature. The likelihood can be stated as, 

<a href="https://www.codecogs.com/eqnedit.php?latex=LL(z)=&space;\sum_{i=1}^{n}&space;y^{i}\log&space;p(z^{i})&space;&plus;\left&space;(&space;1-&space;y^{i}\right&space;)\log\left&space;(&space;1-&space;p(z^{i})\right&space;)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?LL(z)=&space;\sum_{i=1}^{n}&space;y^{i}\log&space;p(z^{i})&space;&plus;\left&space;(&space;1-&space;y^{i}\right&space;)\log\left&space;(&space;1-&space;p(z^{i})\right&space;)" title="LL(z)= \sum_{i=1}^{n} y^{i}\log p(z^{i}) +\left ( 1- y^{i}\right )\log\left ( 1- p(z^{i})\right )" /></a>

The negation of this function was mininised using `scipy.optimize` and the BFGS methodology.

## Results

![full_fit](/Logistic_Regression/Full_fit.png)

Above we have the full fit of all team scores from 1994 to 2019. We find that a score over 171 points has a greater than 50% chance to win any match through out the modern history of University Challenge. 25% and 75% quartiles are also shown.

![full_fit](/Logistic_Regression/3_season_fit.png)
![full_fit](/Logistic_Regression/avg_score_VS_p_score.png)

To investigate any change over time, the loglikelihood function was maximised next on a per series basis. Plotted for context we have the first, the median and most recent fits shown above. We can see major deviation in the 1994 curve when compared to more recent series. For both the winning and losing teams the highest scores are disproportionately from the first season, whilst there is a comparative dearth in the lowest scores. This has resulted in the sigmoid being shifted, and providing a much larger P = 0.5 score. Whether the questions were easier, or the students smarter, in comparison to today’s classes remains to be seen, but they are scoring more points on average. 

Also provided is the time series of P = 0.5 score per series accompanied by the average winning score per series. Here we can clearly see the trend, in both metrics, of fewer points being needed to win a match. Interestingly, after a rapid decline through 2000 -2005 the metrics bounced back for a period of ~ 10 years. Was this deliberate? Did the producers want a more high scoring game to make better television? We have reached a similar nadir in 2019, so it will be interesting to see if scores bounce back in this coming year.  
