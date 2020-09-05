# Analysis
The aim for this investigation was to derive the historical importance of score to the success of a team. Obviously one needs to outscore their opponent, but by how much? When can we say that a team has accrued enough points to have a greater than 50% chance of winning the match?

A suitable technique for this line of enquiry is linear logistic regression. We begin by stating the sigmoid function that will model the probability of winning the match, as a function of a team’s score, as    

<img src="https://bit.ly/2F3RFto" align="center" border="0" alt="P\left ( z \right )=\frac{1}{1+e^{-z}}" width="183" height="65" />

Where Z is given by the linear combination,

<img src="https://bit.ly/3bt0kBP" align="center" border="0" alt=" \beta_{0} + \beta_{1} x" width="115" height="29" /> 

where X is a teams score.

We proceed to derive the coefficients beta naught and beta one by maximising the log likelihood for a Bernoulli random variable, since the labels we are predicting are binary in nature. The likelihood can be stated as, 

<img src="https://bit.ly/3lS4JCV" align="center" border="0" alt="LL(z)= \sum_{i=1}^{n} y^{i}\log p(z^{i}) +\left ( 1- y^{i}\right )\log\left ( 1- p(z^{i})\right )" width="575" height="75" />

The negation of this function was mininised using `SciPy.optimise` and the BFGS methodology. 
