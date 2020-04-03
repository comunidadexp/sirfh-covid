
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from datetime import timedelta, datetime
import datetime as dt


class SIR(object):
    def __init__(self,
                 country='Brazil',
                 N = 200e6,
                 infectedAssumption=1,
                 recoveredAssumption=1,
                 nth=1,
                 daysPredict=150,
                 estimateBeta2 = False,
                 estimateBeta3 = False,
                 quarantineDate = None,
                 backDate = dt.datetime(2020, 4, 22),
                 alpha=0.5,
                 forcedBeta = None,
                 forcedGamma = None,
                 log = False,
                 betaBounds = (0.00000001, 2.0),
                 gammaBounds=(0.00000001, 2.0),
                 ):

        self.country = country
        self.N = N
        self.infectedAssumption = infectedAssumption  # Multiplier to account for non reported
        self.recoveredAssumption = recoveredAssumption
        self.nth = nth  # minimum number of cases to start modelling
        self.daysPredict = daysPredict
        self.quarantineDate = quarantineDate
        self.backDate = backDate
        self.estimateBeta2 = estimateBeta2
        self.estimateBeta3 = estimateBeta3
        self.alpha = alpha
        self.forcedBeta = forcedBeta
        self.forcedGamma = forcedGamma
        self.log = log
        self.betaBounds = betaBounds
        self.gammaBounds = gammaBounds

        self.load_data()

        self.end_data = self.confirmed.index.max()

    def load_CSSE(self,
                       dir=".\\COVID-19\\csse_covid_19_data\\csse_covid_19_time_series\\"):

        confirmed = pd.read_csv(dir+"time_series_covid19_confirmed_global.csv")
        confirmed = confirmed.drop(confirmed.columns[[0, 2, 3]], axis=1).set_index('Country/Region').T
        confirmed.index = pd.to_datetime(confirmed.index)
        self.confirmed = confirmed[self.country]


        deaths = pd.read_csv(dir + "time_series_covid19_deaths_global.csv")
        deaths = deaths.drop(deaths.columns[[0, 2, 3]], axis=1).set_index('Country/Region').T
        deaths.index = pd.to_datetime(deaths.index)
        self.fatal = deaths[self.country]

        recovered = pd.read_csv(dir + "time_series_covid19_recovered_global.csv")
        recovered = recovered.drop(recovered.columns[[0, 2, 3]], axis=1).set_index('Country/Region').T
        recovered.index = pd.to_datetime(recovered.index)
        self.recovered = recovered[self.country]

        if self.log:
            self.confirmed = self.confirmed.transform(np.log)
            self.fatal = self.confirmed.transform(np.log)
            self.recovered = self.confirmed.transform(np.log)

    def load_data(self):
        """
        New function to use our prop data
        """
        self.load_CSSE()

        # Using unreported estimate
        self.confirmed = self.confirmed * self.infectedAssumption
        self.recovered = self.recovered * self.infectedAssumption * self.recoveredAssumption

        # find date in which nth case is reached
        nth_index = self.confirmed[self.confirmed >= self.nth].index[0]

        if not self.quarantineDate:
            self.quarantineDate=self.confirmed.index[-1]
        quarantine_index = pd.Series(False, index=self.confirmed.index)
        quarantine_index[quarantine_index.index >= self.quarantineDate] = True

        self.quarantine_index = quarantine_index.loc[nth_index:]
        self.confirmed = self.confirmed.loc[nth_index:]
        self.fatal = self.fatal.loc[nth_index:]
        self.recovered = self.recovered.loc[nth_index:]

        self.initialize_parameters()

        #True data series
        self.R_actual = self.fatal + self.recovered
        self.I_actual = self.confirmed - self.R_actual

    def initialize_parameters(self):
        self.R_0 = self.recovered[0] + self.fatal[0]
        self.I_0 = (self.confirmed.iloc[0] - self.R_0)
        self.S_0 = self.N - self.R_0 - self.I_0

    def extend_index(self, index, new_size):

        new_values = pd.date_range(start=index[-1], periods=150)
        new_index = index.join(new_values, how='outer')

        return new_index

    def estimate(self):

        self.quarantine_loc = float(self.confirmed.index.get_loc(self.quarantineDate))

        if not self.forcedBeta:
            betaBounds = self.betaBounds
        else:
            betaBounds = (self.forcedBeta, self.forcedBeta)

        if not self.forcedGamma:
            gammaBounds = self.gammaBounds
        else:
            gammaBounds = (self.forcedGamma, self.forcedGamma)

        if self.estimateBeta2:
            optimal = minimize(
                self.loss,
                [0.2, 0.07, 0.2],
                args=(),
                method='L-BFGS-B',
                # options={'maxiter' : 5},
                # method='TNC',
                bounds=[betaBounds, gammaBounds, (0.00000001, 2.0), ]
            )
            self.optimizer = optimal
            beta, gamma, beta2 = optimal.x
            print("Beta:{beta} Beta2: {beta2} Gamma:{gamma}".format(beta=beta, gamma=gamma, beta2=beta2))
            self.beta = beta
            self.gamma = gamma
            self.beta2 = beta2
            self.params =[self.beta, self.gamma, self.beta2]
        else:
            optimal = minimize(
                self.loss,
                [0.2, 0.07,],
                args=(),
                method='L-BFGS-B',
                # options={'maxiter' : 5},
                # method='TNC',
                bounds=[betaBounds, gammaBounds, ]
            )
            self.optimizer = optimal
            beta, gamma, = optimal.x
            print("Beta:{beta} Gamma:{gamma}".format(beta=beta, gamma=gamma, ))
            self.beta = beta
            self.gamma = gamma
            self.params = [self.beta, self.gamma,]

        self.R0 = self.beta / self.gamma
        print('R0:{R0}'.format(R0=self.R0))

    def loss(self, point):
        """
        RMSE between actual confirmed cases and the estimated infectious people with given beta and gamma.
        """
        size = self.confirmed.shape[0]
        if self.estimateBeta2:
            beta, gamma, beta2, = point
        else:
            beta, gamma, = point
            beta2=beta

        def SIR(t, y):
            S = y[0]
            I = y[1]
            R = y[2]

            if t < self.quarantine_loc:
                ret = [-beta * S * I / self.N, beta * S * I / self.N - gamma * I, gamma * I]
            else:
                ret = [-beta2 * S * I / self.N, beta2 * S * I / self.N - gamma * I, gamma * I]
            return ret

        solution = solve_ivp(SIR, [0, size], [self.S_0, self.I_0, self.R_0], t_eval=np.arange(0, size, 1), vectorized=True)

        # Put more emphasis on recovered people
        alpha = self.alpha

        l1 = np.sqrt(np.mean((solution.y[1] - self.I_actual) ** 2))
        l2 = np.sqrt(np.mean((solution.y[2] - self.R_actual) ** 2))

        return alpha * l1 + (1 - alpha) * l2

    def predict(self, beta=None, gamma=None):
        """
        Predict how the number of people in each compartment can be changed through time toward the future.
        The model is formulated with the given beta and gamma.
        """

        #In case predict function is called with custom parameters
        if not beta:
            beta = self.beta
        if not gamma:
            gamma = self.gamma


        if self.estimateBeta2:
            beta2 = self.beta2
        else:
            beta2 = beta

        if self.estimateBeta3:
            # beta3 = self.beta2
            beta3 = (self.beta2 + self.beta) / 2
        else:
            beta3 = beta

        predict_range = self.daysPredict

        # print(self.confirmed.index)
        new_index = self.extend_index(self.confirmed.index, predict_range)
        # print(new_index) #AQUI JA ESTA ERRADO

        size = len(new_index)

        def SIR(t, y):
            S = y[0]
            I = y[1]
            R = y[2]

            if t < self.quarantine_loc:
                ret = [-beta * S * I / self.N, beta * S * I / self.N - gamma * I, gamma * I]
            elif t < self.back_loc:
                ret = [-beta3 * S * I / self.N, beta3 * S * I / self.N - gamma * I, gamma * I]
            else:
                ret = [-beta2 * S * I / self.N, beta2 * S * I / self.N - gamma * I, gamma * I]
            return ret

        # print(self.backDate)
        # print(new_index.get_loc(self.backDate))
        self.quarantine_loc = float(self.confirmed.index.get_loc(self.quarantineDate))
        self.back_loc = float(new_index.get_loc(self.backDate))

        prediction = solve_ivp(SIR, [0, size], [self.S_0, self.I_0, self.R_0],
                                                     t_eval=np.arange(0, size, 1))

        df = pd.DataFrame({
            'I_Actual': self.I_actual.reindex(new_index),
            'R_Actual': self.R_actual.reindex(new_index),
            'S': prediction.y[0],
            'I': prediction.y[1],
            'R': prediction.y[2]
        }, index=new_index)

        # if self.log:
        #     df = df.transform(np.exp)

        self.df = df
        print("Predicting with Beta:{beta} Beta2: {beta2} Gamma:{gamma}".format(beta=beta, gamma=gamma, beta2=beta2))

    def predict_linear(self, beta=None, gamma=None):
        """
        Predict how the number of people in each compartment can be changed through time toward the future.
        The model is formulated with the given beta and gamma.
        """

        #In case predict function is called with custom parameters
        if not beta:
            beta = self.beta
        if not gamma:
            gamma = self.gamma


        if self.estimateBeta2:
            beta2 = self.beta2
        else:
            beta2 = beta

        if self.estimateBeta3:
            # beta3 = self.beta2
            beta3 = (self.beta2 + self.beta) / 2
        else:
            beta3 = beta

        predict_range = self.daysPredict

        # print(self.confirmed.index)
        new_index = self.extend_index(self.confirmed.index, predict_range)
        # print(new_index) #AQUI JA ESTA ERRADO

        size = len(new_index)

        def SIR(t, y):
            S = y[0]
            I = y[1]
            R = y[2]

            if t < self.quarantine_loc:
                ret = [(-beta * S * I / self.N) + S, (beta * S * I / self.N - gamma * I) + I, (gamma * I) + R]
            elif t < self.back_loc:
                ret = [(-beta3 * S * I / self.N) + S, (beta3 * S * I / self.N - gamma * I) + I, (gamma * I) + R]
            else:
                ret = [(-beta2 * S * I / self.N) + S, (beta2 * S * I / self.N - gamma * I) + I, (gamma * I) + R]
            return ret

        # print(self.backDate)
        # print(new_index.get_loc(self.backDate))
        self.quarantine_loc = float(self.confirmed.index.get_loc(self.quarantineDate))
        self.back_loc = float(new_index.get_loc(self.backDate))

        #prediction = solve_ivp(SIR, [0, size], [self.S_0, self.I_0, self.R_0],t_eval=np.arange(0, size, 1))
        prediction = np.empty((size, 3))
        prediction[0, :] = [self.S_0, self.I_0, self.R_0]

        for t in range(1, size):
            prediction[t, :] = SIR(t, prediction[t-1, :])


        df = pd.DataFrame({
            'I_Actual': self.I_actual.reindex(new_index),
            'R_Actual': self.R_actual.reindex(new_index),
            'S': prediction[:, 0],
            'I': prediction[:, 1],
            'R': prediction[:, 2]
        }, index=new_index)
        self.df = df
        print("Predicting with Beta:{beta} Beta2: {beta2} Gamma:{gamma}".format(beta=beta, gamma=gamma, beta2=beta2))

    def train(self):
        """
        Run the optimization to estimate the beta and gamma fitting the given confirmed cases.
        """

        self.estimate()

        self.predict()

        # fig, ax = plt.subplots(figsize=(15, 10))
        # ax.set_title(self.country)
        # self.df.plot(ax=ax)
        # fig.savefig(f"{self.country}.png")

############## VISUALIZATION METHODS ################
    def I_fit_plot(self):
        self.df[['I_Actual', 'I']].loc[:self.end_data].plot()

    def R_fit_plot(self):
        self.df[['R_Actual', 'R']].loc[:self.end_data].plot()

    def main_plot(self):
        fig, ax = plt.subplots(figsize=(15, 10))
        ax.set_title(self.country)
        self.df.plot(ax=ax)


class SEIR(SIR):
    def __init__(self,
                 incubationPeriod = 7,
                 **kwargs):

        self.incubationPeriod = incubationPeriod
        self.sigma = 1 / incubationPeriod
        super().__init__(**kwargs)


    def initialize_parameters(self):
        ## incubated E_0 = 1/incPeriod * each of the following incPeriod days
        #TODO check
        self.E_0 = ((self.confirmed - self.recovered - self.fatal).iloc[0:(self.incubationPeriod-1)]).sum() / self.incubationPeriod
        self.R_0 = self.recovered[0] + self.fatal[0]
        self.I_0 = (self.confirmed.iloc[0] - self.R_0)
        self.S_0 = self.N - self.R_0 - self.I_0

    def loss(self, point):
        """
        RMSE between actual confirmed cases and the estimated infectious people with given beta and gamma.
        """
        size = self.confirmed.shape[0]
        sigma = self.sigma
        if self.estimateBeta2:
            beta, gamma, beta2 = point
        else:
            beta, gamma = point
            beta2 = beta

        def SIR(t, y):
            S = y[0]
            E = y[1]
            I = y[2]
            R = y[3]

            if t < self.quarantine_loc:
                ret = [-beta * S * I / self.N, beta * S * I / self.N - sigma * E, sigma * E - gamma * I, gamma * I]
            else:
                ret = [-beta2 * S * I / self.N, beta2 * S * I / self.N - sigma * E, sigma * E - gamma * I, gamma * I]
            return ret

        solution = solve_ivp(SIR, [0, size], [self.S_0, self.E_0, self.I_0, self.R_0], t_eval=np.arange(0, size, 1), vectorized=True)

        # Put more emphasis on recovered people
        alpha = self.alpha

        l1 = np.sqrt(np.mean((solution.y[2] - self.I_actual) ** 2))
        l2 = np.sqrt(np.mean((solution.y[3] - self.R_actual) ** 2))

        return alpha * l1 + (1 - alpha) * l2

class LearnerSEIR(object):
    def __init__(self,
                 country='Brazil',
                 N = 200e6,
                 infectedAssumption=1,
                 recoveredAssumption=1,
                 nth=1,
                 daysPredict=150,
                 estimateBeta2 = False,
                 estimateBeta3 = False,
                 quarantineDate = None,
                 backDate = dt.datetime(2020, 4, 22),
                 alpha=0.5,
                 forcedBeta = None,
                 forcedGamma = None,
                 elag=15,
                 ):

        self.country = country
        self.N = N
        self.infectedAssumption = infectedAssumption  # Multiplier to account for non reported
        self.recoveredAssumption = recoveredAssumption
        self.nth = nth  # minimum number of cases to start modelling
        self.daysPredict = daysPredict
        self.quarantineDate = quarantineDate
        self.backDate = backDate
        self.estimateBeta2 = estimateBeta2
        self.estimateBeta3 = estimateBeta3
        self.alpha = alpha
        self.forcedBeta = forcedBeta
        self.forcedGamma = forcedGamma
        self.elag = elag

        self.load_data()

        self.end_data = self.confirmed.index.max()

    def load_CSSE(self,
                       dir=".\\COVID-19\\csse_covid_19_data\\csse_covid_19_time_series\\"):

        confirmed = pd.read_csv(dir+"time_series_covid19_confirmed_global.csv")
        confirmed = confirmed.drop(confirmed.columns[[0, 2, 3]], axis=1).set_index('Country/Region').T
        confirmed.index = pd.to_datetime(confirmed.index)
        self.confirmed = confirmed[self.country]

        deaths = pd.read_csv(dir + "time_series_covid19_deaths_global.csv")
        deaths = deaths.drop(deaths.columns[[0, 2, 3]], axis=1).set_index('Country/Region').T
        deaths.index = pd.to_datetime(deaths.index)
        self.fatal = deaths[self.country]

        recovered = pd.read_csv(dir + "time_series_covid19_recovered_global.csv")
        recovered = recovered.drop(recovered.columns[[0, 2, 3]], axis=1).set_index('Country/Region').T
        recovered.index = pd.to_datetime(recovered.index)
        self.recovered = recovered[self.country]

    def load_data(self):
        """
        New function to use our prop data
        """
        self.load_CSSE()

        # Using unreported estimate
        self.confirmed = self.confirmed * self.infectedAssumption
        self.recovered = self.recovered * self.infectedAssumption * self.recoveredAssumption

        # find date in which nth case is reached
        nth_index = self.confirmed[self.confirmed >= self.nth].index[0]

        if not self.quarantineDate:
            self.quarantineDate=self.confirmed.index[-1]
        quarantine_index = pd.Series(False, index=self.confirmed.index)
        quarantine_index[quarantine_index.index >= self.quarantineDate] = True

        self.quarantine_index = quarantine_index.loc[nth_index:]
        self.confirmed = self.confirmed.loc[nth_index:]
        self.fatal = self.fatal.loc[nth_index:]
        self.recovered = self.recovered.loc[nth_index:]

        #Initial parameters
        self.E_0 = self.confirmed.iloc[self.elag]
        self.R_0 = self.recovered[0] + self.fatal[0]
        self.I_0 = (self.confirmed.iloc[0] - self.R_0)
        self.S_0 = self.N - self.R_0 - self.I_0

        #True data series
        self.R_actual = self.fatal + self.recovered
        self.I_actual = self.confirmed - self.R_actual

    def extend_index(self, index, new_size):

        new_values = pd.date_range(start=index[-1], periods=150)
        new_index = index.join(new_values, how='outer')

        return new_index

    def estimate(self):

        self.quarantine_loc = float(self.confirmed.index.get_loc(self.quarantineDate))

        if not self.forcedBeta:
            betaBounds = (0.00000001, 2.0)
        else:
            betaBounds = (self.forcedBeta, self.forcedBeta)

        if not self.forcedGamma:
            gammaBounds = (0.00000001, 2.0)
        else:
            gammaBounds = (self.forcedGamma, self.forcedGamma)

        sigmaBounds = (0.00001, 2.0)

        if self.estimateBeta2:
            optimal = minimize(
                self.loss,
                [0.001, 0.001, 0.001, 0.001],
                args=(),
                method='L-BFGS-B',
                # options={'maxiter' : 5},
                # method='TNC',
                bounds=[betaBounds, gammaBounds, (0.00000001, 2.0), sigmaBounds]
            )
            self.optimizer = optimal
            beta, gamma, beta2, sigma = optimal.x
            print("Beta:{beta} Beta2: {beta2} Gamma:{gamma} Sigma:{sigma}".format(beta=beta, gamma=gamma, beta2=beta2, sigma=sigma))
            self.beta = beta
            self.gamma = gamma
            self.beta2 = beta2
            self.params =[self.beta, self.gamma, self.beta2, self.sigma]
        else:
            optimal = minimize(
                self.loss,
                [0.1, 0.001, 0.1],
                args=(),
                method='L-BFGS-B',
                # options={'maxiter' : 5},
                # method='TNC',
                bounds=[betaBounds, gammaBounds, sigmaBounds]
            )
            self.optimizer = optimal
            beta, gamma, sigma = optimal.x
            print("Beta:{beta} Gamma:{gamma} Sigma:{sigma}".format(beta=beta, gamma=gamma, sigma=sigma))
            self.beta = beta
            self.gamma = gamma
            self.sigma = sigma
            self.params = [self.beta, self.gamma, self.sigma]

        self.R0 = self.beta / self.gamma
        print('R0:{R0}'.format(R0=self.R0))

    def loss(self, point):
        """
        RMSE between actual confirmed cases and the estimated infectious people with given beta and gamma.
        """
        size = self.confirmed.shape[0]
        if self.estimateBeta2:
            beta, gamma, beta2, sigma = point
        else:
            beta, gamma, sigma = point
            beta2 = beta

        def SIR(t, y):
            S = y[0]
            E = y[1]
            I = y[2]
            R = y[3]

            if t < self.quarantine_loc:
                ret = [-beta * S * I / self.N, beta * S * I / self.N - sigma * E, sigma * E - gamma * I, gamma * I]
            else:
                ret = [-beta2 * S * I / self.N, beta2 * S * I / self.N - sigma * E, sigma * E - gamma * I, gamma * I]
            return ret

        solution = solve_ivp(SIR, [0, size], [self.S_0, self.E_0, self.I_0, self.R_0], t_eval=np.arange(0, size, 1), vectorized=True)

        # Put more emphasis on recovered people
        alpha = self.alpha

        l1 = np.sqrt(np.mean((solution.y[2] - self.I_actual) ** 2))
        l2 = np.sqrt(np.mean((solution.y[3] - self.R_actual) ** 2))

        return alpha * l1 + (1 - alpha) * l2

    def predict(self, beta=None, gamma=None):
        """
        Predict how the number of people in each compartment can be changed through time toward the future.
        The model is formulated with the given beta and gamma.
        """

        #In case predict function is called with custom parameters
        if not beta:
            beta = self.beta
        if not gamma:
            gamma = self.gamma


        if self.estimateBeta2:
            beta2 = self.beta2
        else:
            beta2 = beta

        if self.estimateBeta3:
            # beta3 = self.beta2
            beta3 = (self.beta2 + self.beta) / 2
        else:
            beta3 = beta

        sigma = self.sigma

        predict_range = self.daysPredict

        # print(self.confirmed.index)
        new_index = self.extend_index(self.confirmed.index, predict_range)
        # print(new_index) #AQUI JA ESTA ERRADO

        size = len(new_index)

        def SIR(t, y):
            S = y[0]
            E = y[1]
            I = y[2]
            R = y[3]

            if t < self.quarantine_loc:
                ret = [-beta * S * I / self.N, beta * S * I / self.N - sigma * E, sigma * E - gamma * I, gamma * I]
            else:
                ret = [-beta2 * S * I / self.N, beta2 * S * I / self.N - sigma * E, sigma * E - gamma * I, gamma * I]
            return ret

        # print(self.backDate)
        # print(new_index.get_loc(self.backDate))
        self.quarantine_loc = float(self.confirmed.index.get_loc(self.quarantineDate))
        self.back_loc = float(new_index.get_loc(self.backDate))

        prediction = solve_ivp(SIR, [0, size], [self.S_0, self.E_0, self.I_0, self.R_0],
                                                     t_eval=np.arange(0, size, 1))

        df = pd.DataFrame({
            'I_Actual': self.I_actual.reindex(new_index),
            'R_Actual': self.R_actual.reindex(new_index),
            'S': prediction.y[0],
            'E': prediction.y[1],
            'I': prediction.y[2],
            'R': prediction.y[3]
        }, index=new_index)
        self.df = df
        print("Predicting with Beta:{beta} Beta2: {beta2} Gamma:{gamma} Sigma:{sigma}".format(beta=beta, gamma=gamma,
                                                                                              beta2=beta2,sigma=sigma))

    def predict_linear(self, beta=None, gamma=None):
        """
        Predict how the number of people in each compartment can be changed through time toward the future.
        The model is formulated with the given beta and gamma.
        """

        #In case predict function is called with custom parameters
        if not beta:
            beta = self.beta
        if not gamma:
            gamma = self.gamma


        if self.estimateBeta2:
            beta2 = self.beta2
        else:
            beta2 = beta

        if self.estimateBeta3:
            # beta3 = self.beta2
            beta3 = (self.beta2 + self.beta) / 2
        else:
            beta3 = beta

        predict_range = self.daysPredict

        # print(self.confirmed.index)
        new_index = self.extend_index(self.confirmed.index, predict_range)
        # print(new_index) #AQUI JA ESTA ERRADO

        size = len(new_index)

        def SIR(t, y):
            S = y[0]
            I = y[1]
            R = y[2]

            if t < self.quarantine_loc:
                ret = [(-beta * S * I / self.N) + S, (beta * S * I / self.N - gamma * I) + I, (gamma * I) + R]
            elif t < self.back_loc:
                ret = [(-beta3 * S * I / self.N) + S, (beta3 * S * I / self.N - gamma * I) + I, (gamma * I) + R]
            else:
                ret = [(-beta2 * S * I / self.N) + S, (beta2 * S * I / self.N - gamma * I) + I, (gamma * I) + R]
            return ret

        # print(self.backDate)
        # print(new_index.get_loc(self.backDate))
        self.quarantine_loc = float(self.confirmed.index.get_loc(self.quarantineDate))
        self.back_loc = float(new_index.get_loc(self.backDate))

        #prediction = solve_ivp(SIR, [0, size], [self.S_0, self.I_0, self.R_0],t_eval=np.arange(0, size, 1))
        prediction = np.empty((size, 3))
        prediction[0, :] = [self.S_0, self.I_0, self.R_0]

        for t in range(1, size):
            prediction[t, :] = SIR(t, prediction[t-1, :])


        df = pd.DataFrame({
            'I_Actual': self.I_actual.reindex(new_index),
            'R_Actual': self.R_actual.reindex(new_index),
            'S': prediction[:, 0],
            'I': prediction[:, 1],
            'R': prediction[:, 2]
        }, index=new_index)
        self.df = df
        print("Predicting with Beta:{beta} Beta2: {beta2} Gamma:{gamma}".format(beta=beta, gamma=gamma, beta2=beta2))

    def train(self):
        """
        Run the optimization to estimate the beta and gamma fitting the given confirmed cases.
        """

        self.estimate()

        self.predict()

        # fig, ax = plt.subplots(figsize=(15, 10))
        # ax.set_title(self.country)
        # self.df.plot(ax=ax)
        # fig.savefig(f"{self.country}.png")

############## VISUALIZATION METHODS ################
    def I_fit_plot(self):
        self.df[['I_Actual', 'I']].loc[:self.end_data].plot()

    def R_fit_plot(self):
        self.df[['R_Actual', 'R']].loc[:self.end_data].plot()

    def main_plot(self):
        fig, ax = plt.subplots(figsize=(15, 10))
        ax.set_title(self.country)
        self.df.plot(ax=ax)



if __name__ == '__main__':
    jap = SIR('Brazil',
                  N=200e6,
                  alpha=0.7,
                  betaBounds=(0.1, 0.3),
                  gammaBounds=(0.05, 0.1),
                  )
    # out = jap.train()
    jap.train()
    # jap.estimate()
    # jap.predict_linear(beta=0.2, gamma=0.07)