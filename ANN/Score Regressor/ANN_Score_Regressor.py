import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from collections import defaultdict
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split, cross_val_score

class cleaner():

    def __init__(self):
        self.string = 'C:/Users/User/Documents/University-Challenge/UC_database.txt'
        self.data = pd.read_csv(self.string)
        self.years = np.array(list(set(self.data['Season'])))
        self.uni_names = sorted(set(list(self.data['Team1'].append(self.data['Team2']))))
        self.run_avg_dict = defaultdict(list)

        self.season_avg_dict = self.season_avg()
        self.results = self.standardising()
        self.running_avgs = self.gen_running_avg()
        self.his_avgs = self.gen_his_avgs()
        self.cleaned = self.comp()


    def season_avg(self):
        '''Calculating season AVGs'''

        yr_masks = [self.data['Season'] == year for year in self.years]
        season_avg = [np.average(self.data[mask]['Team1_score'].append(
                                 self.data[mask]['Team2_score'])) \
                                 for mask in yr_masks]
        avg_dic = dict(zip(self.years, season_avg))
        return avg_dic

    def standardising(self):
        #Standardising scores based on season average
        avgs = [self.season_avg_dict.get(year) for year in self.data['Season']]
        res_dict = {'Team1_score': self.data['Team1_score']/avgs,
                    'Team2_score': self.data['Team2_score']/avgs
                    }
        #Generating running avgs
        results = pd.DataFrame(data=res_dict)
        return results

    def gen_running_avg(self):
        '''Calculates the running average of the last three scores. If only two
        scores returns their average, if only one it draws from normal distrubution
        with a mean of that score'''


        T1_tups = zip(self.data['Team1'], self.results['Team1_score'])
        T2_tups = zip(self.data['Team2'], self.results['Team2_score'])
        Match_tups = zip(T1_tups,T2_tups)

        c_matches = []
        for match in Match_tups:
            c_scores = []
            for team in match:
                name, score = team

                score_list = self.run_avg_dict[name]

                if len(score_list) == 0:
                    c_score = 1.0
                elif len(score_list) == 1:
                    c_score = score_list[0] #+ np.random.randn()/5
                elif len(score_list) == 2:
                    c_score = np.mean(score_list) #+ np.random.randn()/5
                else:
                    c_score = np.mean(score_list[-3:])

                self.run_avg_dict[name].append(score)
                c_scores.append(c_score)
            c_matches.append(c_scores)

        running_avgs = pd.DataFrame(c_matches, columns=['Team1RAVG_score',
                                                        'Team2RAVG_score']
                                    )

        return running_avgs
    def gen_his_avgs(self):
        names = self.uni_names
        his_dict = {name : np.mean(self.run_avg_dict[name]) for name in names }
        his_avg_dict = {'Team1_Hisavg' : [his_dict.get(name) for name in self.data['Team1']],
                        'Team2_Hisavg' : [his_dict.get(name) for name in self.data['Team2']]
                        }
        output = pd.DataFrame(his_avg_dict)

        return output
    def comp(self):
        cleaned = self.running_avgs.join(self.his_avgs).join(self.results)
        return cleaned

class rank_calculator(cleaner):
    '''This class trains a regression network to predict scores from cleaned
    data in a head to head match up. Using these predicted scores we simulate
    a season of UC with random draws.'''

    def __init__(self, participants):
        super().__init__()
        
        self.input_names = ['Team1RAVG_score',
                            'Team2RAVG_score',
                            'Team1_Hisavg',
                            'Team2_Hisavg']
        self.output_names = ['Team1_score',
                             'Team2_score']
        self.regr = self.train_regressor()
        self.participants = participants
    def train_regressor(self):
        '''Trainning the regression netowrk on the cleaned data'''

        X_train, X_test, y_train, y_test = train_test_split(self.cleaned[self.input_names],
                                                            self.cleaned[self.output_names],
                                                            test_size=0.33,
                                                            random_state=42)

        regr = MLPRegressor(solver='lbfgs', alpha=1e-5,
                            hidden_layer_sizes=(10,3),
                            random_state=56,
                            max_iter = 5000)
        regr.fit(X_train, y_train)

        predictions = regr.predict(X_test)

        T1P_win = predictions[:,0] > predictions[:,1]
        T1_win = y_test['Team1_score'] > y_test['Team2_score']
        correct_percent = (sum(np.logical_not(np.logical_xor(T1P_win, T1_win))) /
                           len(T1_win)) * 100
        print(f'Network produced a match result accuracy of {round(correct_percent, 1)}%')
        return regr

    def tournament_sim(self):
        '''Simulates a UC tournament with random seeding'''
        #Get new copy of team histories
        scores_dict = self.run_avg_dict.copy()

        def round_sim(*args):
            '''Formats the input to regression network for each round, and
            predicts results. Defaults to all participants for R1'''
            if args:
                participants = args[0]
            else:
                participants = self.participants

            np.random.shuffle(participants)
            cutoff = int(len(participants)/2)
            team1 = participants[:cutoff]
            team2 = participants[cutoff:]
            teamset = (team1, team2)
            
            round_data = []
            for teams in teamset:
                #Assumes no new teams
                his_avg  = [np.mean(scores_dict.get(team)) for team in teams]

                run_avg = []
                for team in teams:
                    score_list = scores_dict[team]
                    if len(score_list) == 0:
                        c_score = 1.0
                    elif len(score_list) == 1:
                        c_score = score_list[0]
                    elif len(score_list) == 2:
                        c_score = np.mean(score_list)
                    else:
                        c_score = np.mean(score_list[-3:])
                    run_avg.append(c_score)

                round_data.append(run_avg)
                round_data.append(his_avg)
                round_data.append(teams)
                out = pd.DataFrame(round_data).T
                       

            R_results = pd.DataFrame(self.regr.predict(out[[0,3,1,4]]),
                                     columns=['Team1_score',
                                              'Team2_score'])
            R_results['Team1'] = team1
            R_results['Team2'] = team2
            return R_results
        
        def winners(results, *args):
            '''Returns a list of winners and updates the score dictionary'''
            T1_win = results['Team1_score'] > results['Team2_score']
            T2_win = np.logical_not(T1_win)
            T1_win_score = zip(results['Team1'][T1_win],
                               results['Team1_score'][T1_win]
                               )
            T2_win_score = zip(results['Team2'][T2_win],
                               results['Team2_score'][T2_win]
                               )
            win_scores = [T1_win_score, T2_win_score]
            
            for win_score in win_scores:
                for pair in win_score:
                    name, score = pair
                    scores_dict[name].append(score)
                    
            winners = list(results['Team1'][T1_win].append(results['Team2'][T2_win]))
            
            if args: #SORT OUT HSL
                T1_loser = results[['Team1','Team1_score']][T2_win]
                T2_loser = results[['Team2','Team2_score']][T1_win]
                
                loserteams = T1_loser['Team1'].append(T2_loser['Team2'])
                loser_scores = T1_loser['Team1_score'].append(T2_loser['Team2_score'])
                losers = pd.DataFrame({'Losers' : loserteams,
                                       'Loser_score' : loser_scores})
                losers = losers.sort_values(by=['Loser_score']) 
                HSL = losers.iloc[-4:]['Losers']
                
                return winners, HSL
            
            return winners
        
                
        #FIRST ROUND ---------------------------------------------------------

        R1_results = round_sim() 
        R1_winners = winners(R1_results, True)
        
        #HIGHEST SCORING LOSER ROUND -----------------------------------------

        return R1_winners

participants = pd.read_csv('C:/Users/User/Documents/University-Challenge/ANN/Score Regressor/2020_Participants.txt',
                           names = ['Teams'])
main = rank_calculator(list(participants['Teams']))
z = main.tournament_sim()





