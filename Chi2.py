import pandas as pd
import numpy as np
from scipy import optimize
import matplotlib.pyplot as plt
import scipy.stats as stats

def standardise(data):
    mean = np.mean(data)
    std = np.std(data)
    return (data-mean) / std

def expected_norm(bins, n):
    values = stats.norm.cdf(bins)
    delta = [values[i+1] - values[i] for i in range(len(values) - 1)]
    _sum = np.sum(n)
    return np.array(delta) * _sum

def check_if_normal(scores):
    
    n, bins, patch = plt.hist(scores, bins = 'auto')
    f_exp = expected_norm(bins, n)
    
    chisq, p = stats.chisquare(n, f_exp, ddof = 3)
    return chisq, p
    
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
    
    NUM_seasons = set(INPUT['SEASON'])
    COLS = ['W_CHISQ', 'W_p','L_CHISQ', 'L_p']
    TESTS = pd.DataFrame(None, index =set(INPUT['SEASON']),columns=COLS)
    
    WINNERS = INPUT['RESULT'] == 1
    LOSERS = np.logical_not(WINNERS)
    
    for season in NUM_seasons:
        WIN_MASK = np.logical_and(WINNERS, INPUT['SEASON'] == season)
        LOSS_MASK = np.logical_and(LOSERS, INPUT['SEASON'] == season)
        
        WIN_DATA = standardise(INPUT['SCORE'][WIN_MASK])
        LOSS_DATA = standardise(INPUT['SCORE'][LOSS_MASK])
        
        WIN_STATS = check_if_normal(WIN_DATA)
        LOSS_STATS = check_if_normal(LOSS_DATA)

        ROW = WIN_STATS + LOSS_STATS
        TESTS.loc[season] = ROW
        
    plt.clf()
    plt.plot(list(NUM_seasons), TESTS.W_p, c='r')
    plt.plot(list(NUM_seasons), TESTS.L_p, c='b')

    
    
    
    
                             