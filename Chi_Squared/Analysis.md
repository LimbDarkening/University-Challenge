# Normality

Given the scores for both the winners and losers, what distribution do they follow? It is clear that the score of a given match follows a binomial distribution, but when in summation with other matches this cannot be assumed.

We first start by checking for the normality of both distributions. We utilise the Chi Squared statistic to assess the goodness of fit, given by,

<a href="https://www.codecogs.com/eqnedit.php?latex=\chi&space;^{2}&space;=&space;\sum_{i=1}^{k}\frac{&space;\left&space;(&space;O_{i}&space;-E_{i}\right&space;)^{2}}{E_{i}}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\chi&space;^{2}&space;=&space;\sum_{i=1}^{k}\frac{&space;\left&space;(&space;O_{i}&space;-E_{i}\right&space;)^{2}}{E_{i}}" title="\chi ^{2} = \sum_{i=1}^{k}\frac{ \left ( O_{i} -E_{i}\right )^{2}}{E_{i}}" /></a>


