import pandas as pd
import numpy as np
from scipy import optimize
import matplotlib.pyplot as plt

class Log_Regress():
    """ Performs Logistical Regression on a single feature."""
    def __init__(self):
        self.coeff = (0, 0)
        self.data = []

    def _sigma(coeff, x):
        """sigmoid function"""
        b0, b1 = coeff
        z = b0 + b1 * x
        return 1 / (1 + np.exp(-z))

    def _Log_like(coeff, data):
        """returns the log likelihood of the data and coefficent pairing"""
        s = Log_Regress._sigma(coeff, data['SCORE'])
        y = data['RESULT']
        LL = np.sum(y * np.log(s) + (1 - y)*np.log(1 - s))
        #negative so we can use optimise.minimise()
        return -LL

    def fit(self, data):
        """Train the model with the provided data"""
        resmax = optimize.minimize(Log_Regress._Log_like, self.coeff, args=data,
                                   tol=1e-3)
        self.coeff = resmax.x
        self.data = data

    def predict_score(self, p):
        """predicts score given probability"""
        b0, b1 = self.coeff
        return (np.log(p/(1-p))-b0) / b1

    def plot_fit(self):
        """plots theh fitted sigmoid to the data"""
        plt.scatter(self.data['SCORE'], self.data['RESULT'])
        _max = np.max(self.data['SCORE'])
        _min = np.min(self.data['SCORE'])
        x = np.linspace(_min, _max, 400)
        y = Log_Regress._sigma(self.coeff, x)
        plt.plot(x, y)
        plt.xlabel('Score')
        plt.ylabel('Probability')
        
        

        plt.show()

if __name__ == "__main__":
    #Select data
    DATA = pd.read_csv('UC_database.txt')
    
    RESULTS_T1 = (DATA['Team1_score'] > DATA['Team2_score']) * 1
    RESULTS_T2 = np.logical_not(RESULTS_T1) * 1
    
    RESULTS = RESULTS_T1.append(RESULTS_T2)
    SCORE = DATA['Team1_score'].append(DATA['Team2_score'])
    SEASON = DATA['Season'].append(DATA['Season'])
    _DIC = {'SCORE': SCORE,
            'RESULT': RESULTS,
            'SEASON': SEASON}
    INPUT = pd.DataFrame(_DIC, columns=['SCORE','RESULT','SEASON'])
    
    """    
    #Full fit of data
    INST  = Log_Regress()
    INST.fit(INPUT)
    INST.plot_fit()
    
    IQR = [0.25,0.5,0.75]
    SCORES = [INST.predict_score(prob) for prob in IQR]
    for i, bound in enumerate(SCORES):
        plt.plot([bound, bound], [0, IQR[i]], linestyle='-.', c='r')
        plt.plot([0, bound],[IQR[i], IQR[i]], linestyle='-.', c='r' )
        plt.text(0, IQR[i]+0.05, f'{IQR[i]*100}% = {int(bound)}')
    
    """
    """
    #start mid end fits
    labels = [1994, 2006, 2019]
    masks = [INPUT['SEASON'] == num for num in labels]
    avg_p = []
    for mask in masks:
        INST  = Log_Regress()
        INST.fit(INPUT[mask])
        avg_p.append(INST.predict_score(0.5))
        INST.plot_fit()
        
    plt.legend(labels)
    """
    """
    #Comparing avg to 50 prob graph
    avg = [np.mean(INPUT[mask]['SCORE']) for mask in masks]
    p_lab = 'Score to have a 50% probability of winning a match'
    avg_lab = 'Mean score'
    plt.plot(labels, avg_p, c='r', label=p_lab)
    plt.plot(labels, avg, c='b', label=avg_lab)
    plt.xlabel('Series')
    plt.ylabel('Score')
    plt.legend()
    plt.show()
    """


