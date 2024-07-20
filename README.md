# IMO_prediction_algorithm
Algorithm for predicting various summary statistics (medal cuts, country performances) of an IMO, as well as some past data, using something approximating MCMC

The view of the algorithm currently tends to be that individual scores don't matter and country scores do for predicting performance, 
unsure if this is desirable behaviour 
usage: in a python shell
import prediction_alg_current
output=prediction_alg_current.predict(filename='IMO2024.csv')
output.printCuts()
output.CountryResults("AUS") etc.
