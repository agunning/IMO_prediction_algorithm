# IMO_prediction_algorithm
Algorithm for predicting various summary statistics (medal cuts, country performances) of an IMO, as well as some past data, using something approximating MCMC

The view of the algorithm currently tends to be that individual scores don't matter and country scores do for predicting performance, 
unsure if this is desirable behaviour 
usage: in a python shell

import prediction_alg_current

output=prediction_alg_current.predict(filename='IMO2024.csv',Trials=1000) #Works for any similary formatted file. 1000 trials should work for cutoffs, 10000 for country results.

output.printCuts()

output.printWinner() #Gives an estimation of which country is likely to win the IMO

output.CountryResults("AUS")

output.queryContestantScore(28) #should you know your score, this gives you your ranking

etc.
