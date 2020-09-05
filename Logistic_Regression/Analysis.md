# Analysis
The aim for this investigation was to derive the historical importance of score to the success of a team. Obviously one needs to outscore their opponent, but by how much? When can we say that a team has accrued enough points to have a greater than 50% chance of winning the match?
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
