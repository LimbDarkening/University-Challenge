# Normality

Given the scores for both the winners and losers, what distribution do they follow? It is clear that the score of a given match follows a binomial distribution, but when in summation with other matches this cannot be assumed.

We first start by checking for the normality of both distributions. We utilise the Chi Squared statistic to assess the goodness of fit, given by,

<a href="https://www.codecogs.com/eqnedit.php?latex=\chi&space;^{2}&space;=&space;\sum_{i=1}^{k}\frac{&space;\left&space;(&space;O_{i}&space;-E_{i}\right&space;)^{2}}{E_{i}}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\chi&space;^{2}&space;=&space;\sum_{i=1}^{k}\frac{&space;\left&space;(&space;O_{i}&space;-E_{i}\right&space;)^{2}}{E_{i}}" title="\chi ^{2} = \sum_{i=1}^{k}\frac{ \left ( O_{i} -E_{i}\right )^{2}}{E_{i}}" /></a>

here O is the observed frequency for bin i and E is the expected frequency for bin i. The expected frequency is calculated by,

<a href="https://www.codecogs.com/eqnedit.php?latex=E_{i}&space;=&space;N\left&space;(&space;F\left&space;(&space;L_{u}&space;\right&space;)&space;-&space;F\left&space;(L_{l}\right&space;)&space;\right&space;)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?E_{i}&space;=&space;N\left&space;(&space;F\left&space;(&space;L_{u}&space;\right&space;)&space;-&space;F\left&space;(L_{l}\right&space;)&space;\right&space;)" title="E_{i} = N\left ( F\left ( L_{u} \right ) - F\left (L_{l}\right ) \right )" /></a>

Where F is the cdf of our Gaussian distribution, N the number of data points, and L u/l the upper and lower bound of the bin we are generating the expected frequency for.

One of the limitations of this chi squared approach is that the validity of the technique is a function of the binning technique. If too few data points are grouped together our chi-squared value becomes unreliable. Here we used Sturges’ method for calculating bin edges and achieved a minimum of 10 data points per bin.

Once calculated we can then compare this statistic to the critical value from the chi square distribution with equivalent degrees of freedom, and a significance value of 0.05, to assess if we can reject the null hypothesis that the data is normally distributed.


# Results


