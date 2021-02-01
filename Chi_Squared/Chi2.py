import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

class Chi_Squared_test():
    
    @staticmethod
    def standardise(data):
        mean = np.mean(data)
        std = np.std(data)
        return (data-mean) / std
    
    @staticmethod
    def _expected_norm(bins, n):
        values = stats.norm.cdf(bins)
        delta = [values[i+1] - values[i] for i in range(len(values) - 1)]
        _sum = np.sum(n)
        return np.array(delta) * _sum

    def check_if_normal(self, scores):

        n, bins, patch = plt.hist(Chi_Squared_test.standardise(scores),
                                  bins = 'auto'
                                  )
        f_exp = Chi_Squared_test._expected_norm(bins, n)
        plt.clf()
        chisq, p = stats.chisquare(n, f_exp, ddof = 3)
        return chisq, p

    def normal(self, data):
        mean = np.mean(data)
        std = np.std(data)
        _min, _max = np.min(data), np.max(data)
        x = np.linspace(_min,_max, 100)
        y = stats.norm.pdf(x, mean, std)
        return x, y

if __name__ == "__main__":
    #Select data
    DATA = pd.read_csv('C:/Users/User/Documents/University-Challenge/UC_database.txt')

    RESULTS_T1 = (DATA['Team1_score'] > DATA['Team2_score']) * 1
    RESULTS_T2 = np.logical_not(RESULTS_T1) * 1

    RESULTS = RESULTS_T1.append(RESULTS_T2)
    SCORE = DATA['Team1_score'].append(DATA['Team2_score'])
    SEASON = DATA['Season'].append(DATA['Season'])
    _DIC = {'SCORE': SCORE,
            'RESULT': RESULTS,
            'SEASON': SEASON}
    INPUT = pd.DataFrame(_DIC, columns=['SCORE','RESULT','SEASON'])


    WIN = INPUT['RESULT'] == 1
    LOSS = np.logical_not(WIN)
    WINNERS = INPUT['SCORE'][WIN]
    LOSERS = INPUT['SCORE'][LOSS]

    INST = Chi_Squared_test()
    
    #Test Results
    WINNER_P = INST.check_if_normal(WINNERS)
    LOSERS_P = INST.check_if_normal(LOSERS)
    
    #Win Vs Loss
    plt.hist(WINNERS, density=True, label='Winners')
    plt.hist(LOSERS, alpha=0.6, density=True, label='Losers')

    x, y = INST.normal(LOSERS)
    plt.plot(x, y, c='r')
    plt.legend()
    plt.show()

    """
    #PER SERIES ANALYSIS
    NUM_seasons = set(INPUT['SEASON'])
    COLS = ['W_CHISQ', 'W_p','L_CHISQ', 'L_p']
    TESTS = pd.DataFrame(None, index =set(INPUT['SEASON']),columns=COLS)


    for season in NUM_seasons:
        WIN_MASK = np.logical_and(WIN, INPUT['SEASON'] == season)
        LOSS_MASK = np.logical_and(LOSS, INPUT['SEASON'] == season)



        WIN_STATS = INST.check_if_normal(INPUT['SCORE'][WIN_MASK])
        LOSS_STATS = INST.check_if_normal(INPUT['SCORE'][LOSS_MASK])

        ROW = WIN_STATS + LOSS_STATS
        TESTS.loc[season] = ROW

    plt.clf()
    plt.plot(list(NUM_seasons), TESTS.W_p, c='r')
    plt.plot(list(NUM_seasons), TESTS.L_p, c='b')
    """





