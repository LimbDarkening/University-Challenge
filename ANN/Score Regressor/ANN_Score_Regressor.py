import copy
import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split

class cleaner():
    '''Class imports the raw match data from past series of UC, standardises
    the values against seasonal averages, and calculates the historic and running
    averages as the time of the match.'''

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
        scores returns their average, if only one it returns that score.'''

        T1_tups = zip(self.data['Team1'], self.results['Team1_score'])
        T2_tups = zip(self.data['Team2'], self.results['Team2_score'])
        Match_tups = zip(T1_tups, T2_tups)

        c_matches = []
        for match in Match_tups:
            c_scores = []
            for team in match:
                name, score = team

                score_list = self.run_avg_dict[name]

                if len(score_list) == 0:
                    c_score = 1.0
                elif len(score_list) == 1:
                    c_score = score_list[0]
                elif len(score_list) == 2:
                    c_score = np.mean(score_list)
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
        his_dict = {name : np.mean(self.run_avg_dict[name]) for name in names}
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
        self.regr, self.mean, self.uncert = self.train_regressor()
        self.participants = participants

    def train_regressor(self):
        '''Trainning the regression netowrk on the cleaned data'''

        X_train, X_test, y_train, y_test = train_test_split(self.cleaned[self.input_names],
                                                            self.cleaned[self.output_names],
                                                            test_size=0.33,
                                                            random_state=42)

        regr = MLPRegressor(solver='lbfgs', alpha=1e-5,
                            hidden_layer_sizes=(10, 3),
                            random_state=56,
                            max_iter=5000)
        regr.fit(X_train, y_train)

        predictions = regr.predict(X_test)

        T1P_win = predictions[:, 0] > predictions[:, 1]
        T1_win = y_test['Team1_score'] > y_test['Team2_score']
        correct_percent = (sum(np.logical_not(np.logical_xor(T1P_win, T1_win))) /
                           len(T1_win)) * 100
        print(f'Network produced a match result accuracy of {round(correct_percent, 1)}%')

        deviations = (predictions[:, 0]-y_test['Team1_score']).append(
            predictions[:, 1]-y_test['Team2_score'])
        return regr, np.mean(deviations), np.std(deviations)

    def tournament_sim(self):
        '''Simulates a UC tournament with random seeding'''
        #Get new copy of team histories
        scores_dict = copy.deepcopy(self.run_avg_dict)

        def round_sim(*args):
            '''Formats the input to regression network for each round, and
            predicts results. Defaults to all participants for R1'''

            #Check if R1
            if args:
                participants = args[0]
            else:
                participants = self.participants

            qf = len(participants) == 8
            if not qf:
                np.random.shuffle(participants)


            cutoff = int(len(participants)/2)
            team1 = participants[:cutoff]
            team2 = participants[cutoff:]

            teamset = (team1, team2)
            round_data = []
            for teams in teamset:
                #Assumes no new teams
                his_avg = [np.mean(scores_dict.get(team)) for team in teams]

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


            raw_results = pd.DataFrame(self.regr.predict(out[[0, 3, 1, 4]]))
            Uncert = np.random.normal(self.mean, self.uncert, raw_results.shape)
            Results = raw_results + Uncert
            Results = Results * (Results > 0)
            Results = Results.rename(columns={0:'Team1_score',
                                              1:'Team2_score'}
                                     )
            Results['Team1'] = team1
            Results['Team2'] = team2
            return Results

        def winners_losers(results, round_rank, *args):
            '''Returns a list of winners and updates the score dictionary'''
            qf = len(results) == 4

            T1_win = results['Team1_score'] > results['Team2_score']
            T2_win = np.logical_not(T1_win)
            T1_score = zip(results['Team1'],
                           results['Team1_score']
                           )
            T2_score = zip(results['Team2'],
                           results['Team2_score']
                           )
            name_scores = [T1_score, T2_score]
            #Update Score dictionary
            for name_score in name_scores:
                for pair in name_score:
                    name, score = pair
                    scores_dict[name].append(score)

            T1_winner = results[['Team1', 'Team1_score']][T1_win]
            T2_winner = results[['Team2', 'Team2_score']][T2_win]

            winteams = T1_winner['Team1'].append(T2_winner['Team2'])
            win_scores = T1_winner['Team1_score'].append(T2_winner['Team2_score'])
            winners_df = pd.DataFrame({'winners' : winteams,
                                       'win_score' : win_scores})

            T1_loser = results[['Team1', 'Team1_score']][T2_win]
            T2_loser = results[['Team2', 'Team2_score']][T1_win]

            loserteams = T1_loser['Team1'].append(T2_loser['Team2'])
            loser_scores = T1_loser['Team1_score'].append(T2_loser['Team2_score'])
            losers = pd.DataFrame({'Losers' : loserteams,
                                   'Loser_score' : loser_scores})
            losers = losers.sort_values(by=['Loser_score'])
            Loser_dict = {name : round_rank for name in losers['Losers']}

            if qf:
                winners_df = winners_df.sort_values(by=['win_score'])
                Loser_dict = list(losers['Losers'])

            winners = list(winners_df['winners'])

            if args: #HSL list
                HSL = losers.iloc[-4:]['Losers']
                return winners, Loser_dict, list(HSL)

            
            return winners, Loser_dict

        #FIRST ROUND ---------------------------------------------------------

        R1_results = round_sim()
        R1_winners, rank_dict, Hsl = winners_losers(R1_results, 0, True)

        #HIGHEST SCORING LOSER ROUND -----------------------------------------

        HSL_results = round_sim(Hsl)
        HSL_winners, HSL_losers = winners_losers(HSL_results, 1)
        rank_dict.update(HSL_losers)

        R2_input = R1_winners + HSL_winners

        #SECOND ROUND --------------------------------------------------------

        R2_results = round_sim(R2_input)
        R2_winners, R2_losers = winners_losers(R2_results, 2)
        rank_dict.update(R2_losers)

        #QUARTER FINAL ROUND -------------------------------------------------

        S1QF_results = round_sim(R2_winners)
        S1QF_winners, S1QF_losers = winners_losers(S1QF_results, 3)

        S2QFT1 = [S1QF_winners[0], S1QF_winners[1], S1QF_losers[0], S1QF_losers[1]]
        S2QFT2 = [S1QF_winners[3], S1QF_winners[2], S1QF_losers[3], S1QF_losers[2]]
        S2QF_input = S2QFT1 + S2QFT2

        S2QF_results = round_sim(S2QF_input)
        S2QF_winners, S2QF_losers = winners_losers(S2QF_results, 3)


        double_win = list(set(S1QF_winners).intersection(S2QF_winners))
        double_loss = list(set(S1QF_losers).intersection(S2QF_losers))
        double_loss_dict = dict(zip(double_loss, [3, 3]))

        single_winS1 = list(set(S1QF_winners).intersection(S2QF_losers))
        single_winS2 = list(set(S2QF_winners).intersection(S1QF_losers))
        single_win = single_winS1 + single_winS2


        S3QF_results = round_sim(single_win)
        S3QF_winners, S3QF_losers = winners_losers(S3QF_results, 3)
        rank_dict.update(double_loss_dict)
        rank_dict.update(S3QF_losers)

        SF_input = double_win + S3QF_winners

        #SEMI FINAL ROUND -----------------------------------------------------

        SF_results = round_sim(SF_input)
        SF_winners, SF_losers = winners_losers(SF_results, 4)
        rank_dict.update(SF_losers)

        #FINAL ROUND ----------------------------------------------------------

        F_results = round_sim(SF_winners)
        F_winner, F_loser = winners_losers(F_results, 5)
        rank_dict.update(F_loser)
        rank_dict[F_winner[0]] = 6
        scores_dict = {}
        return rank_dict

participants = pd.read_csv('C:/Users/User/Documents/University-Challenge/ANN/Score Regressor/2020_Participants.txt',
                           names=['Teams'])
main = rank_calculator(list(participants['Teams']))
z = main.tournament_sim()

if __name__ == '__main__':

    loops = 20000
    output = [main.tournament_sim() for i in range(loops)]

    results = defaultdict(list)
    for dic in output:
        for key in z.keys():
            results[key].append(dic[key])

    np.save('20kNormSim2.npy', results)
