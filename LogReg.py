import pandas as pd
import numpy as np
from scipy import optimize
import matplotlib.pyplot as plt

#Select data
DATA = pd.read_csv('UC_database.txt')

RESULTS_T1 = (DATA['Team1_score'] > DATA['Team2_score']) * 1
RESULTS_T2 = np.logical_not(RESULTS_T1) * 1

RESULTS = RESULTS_T1.append(RESULTS_T2)
SCORE = DATA['Team1_score'].append( DATA['Team2_score'])

_dic = {'SCORE': SCORE,
        'RESULT': RESULTS}
INPUT = pd.DataFrame(_dic, columns = ['SCORE','RESULT'])

class Log_Regress():
    """ Performs Logistical Regression on a single feature."""
    def __init__(self):
        self.coeff = (0,0)
    
        
    def _sigma(self, coeff, x):
        """sigmoid function"""
        b0, b1 = coeff
        z = b0 + b1 * x
        return 1 / (1 + np.exp(-z))
    
    def _Log_like(self, coeff, data):
        """returns the log likelihood of the data and coefficent pairing"""
        s = self._sigma(coeff, data['SCORE'])
        LL = np.sum(data['RESULT']*np.log(s) + (1 - data['RESULT'])*np.log(1 - s))
        #negative so we can use optimise.minimise()
        return -LL   
    
    def fit(self, data):
        """Train the model with the provided data"""
        resmax = optimize.minimize(self._Log_like, self.coeff, args = data, tol = 1e-3)
        self.coeff = resmax.x
       
        
    def predict_score(self, p):
        """predicts score given probability"""
        b0, b1 = self.coeff
        return (np.log(p/(1-p))-b0) / b1

    def plot_fit(self, data):
        """plots theh fitted sigmoid to the data"""
        plt.scatter(data['SCORE'], data['RESULT'])
        x = np.linspace(0, 400, 400)
        plt.plot(x, self._sigma(self.coeff, x), c = 'r')
        plt.xlabel('Score')
        plt.ylabel('Probability')

        plt.show()
    
INST = Log_Regress()
ALL = INST.fit(INPUT)
INST.plot_fit(INPUT)
print(INST.coeff)





