# Analysis
The aim for this investigation was to derive the historical importance of score to the success of a team. Obviously one needs to outscore their opponent, but by how much? When can we say that a team has accrued enough points to have a greater than 50% chance of winning the match?

A suitable technique for this line of enquiry is linear logistic regression. We begin by stating the sigmoid function that will model the probability of winning the match, as a function of a team’s score, as    

<img src="https://bit.ly/2F3RFto" align="center" border="0" alt="P\left ( z \right )=\frac{1}{1+e^{-z}}" width="183" height="65" />

Where z is given by the linear combination,

<img src="https://bit.ly/3bt0kBP" align="center" border="0" alt=" \beta_{0} + \beta_{1} x" width="115" height="29" /> 
