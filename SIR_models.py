
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
                 N=200e6,
                 infectedAssumption=1,
                 recoveredAssumption=1,
                 nth=1,
                 daysPredict=150,
                 quarantineDate=None,
                 alpha=0.5,
                 betaBounds=(0.00000001, 2.0),
                 gammaBounds=(0.00000001, 2.0),
                 S0pbounds=(10000, 10e6),
                 R0bounds=None,
                 hospRate=0.15,
                 daysToHosp=7,
                 daysToLeave=7,
                 opt='L-BFGS-B',
                 adjust_recovered=False
                 ):

        self.country = country
        self.N = N
        self.infectedAssumption = infectedAssumption  # Multiplier to account for non reported
        self.recoveredAssumption = recoveredAssumption
        self.R_0th = nth  # minimum number of cases to start modelling
        self.daysPredict = daysPredict
        self.quarantineDate = quarantineDate
        self.alpha = alpha
        self.betaBounds = betaBounds
        self.gammaBounds = gammaBounds
        self.S0pbounds = S0pbounds
        self.hospRate = hospRate
        self.daysToHosp = daysToHosp
        self.hospitalDuration = daysToLeave
        self.opt = opt
        self.R0bounds = R0bounds
        self.adjust_recovered = adjust_recovered

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

        # Adjust recovered curve
        if self.adjust_recovered:
            self.recovered = self.smoothCurve(self.recovered)

        # Using unreported estimate
        self.confirmed = self.confirmed * self.infectedAssumption
        self.recovered = self.recovered * self.infectedAssumption * self.recoveredAssumption

        # find date in which nth case is reached
        nth_index = self.confirmed[self.confirmed >= self.R_0th].index[0]

        if not self.quarantineDate:
            self.quarantineDate = self.confirmed.index[-1]
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

    def smoothCurve(self, df):
        df[df.diff() <= 0] = np.nan
        # df.loc[dt.datetime(2020, 4, 9)] = np.nan
        df.interpolate('linear', inplace=True)
        return df

    def initialize_parameters(self):
        self.R_0 = self.recovered[0] + self.fatal[0]
        self.I_0 = (self.confirmed.iloc[0] - self.R_0)

    def extend_index(self, index, new_size):

        new_values = pd.date_range(start=index[-1], periods=new_size)
        new_index = index.join(new_values, how='outer')

        return new_index

    def estimate(self, verbose=True, options=None):

        self.quarantine_loc = float(self.confirmed.index.get_loc(self.quarantineDate))

        betaBounds = self.betaBounds
        gammaBounds = self.gammaBounds
        S0pbounds = self.S0pbounds

        constraints = [
            {'type': 'ineq', 'fun': self.const_lowerBoundR0},
            {'type': 'ineq', 'fun': self.const_upperBoundR0},
        ]


        optimal = minimize(
            self.loss,
            # [0.2, 0.07, 1e5],
            [0.2, 0.07, 0.01],
            args=(),
            method='SLSQP',
            # options={'maxiter' : 5},
            # method='TNC',
            bounds=[betaBounds, gammaBounds, S0pbounds],
            constraints=constraints,
            options=options,
        )
        self.optimizer = optimal
        beta, gamma, S_0p = optimal.x
        S_0 = S_0p * self.N

        if verbose:
            print("Beta:{beta} Gamma:{gamma} S_0:{S_0}".format(beta=beta, gamma=gamma, S_0=S_0))
        self.beta = beta
        self.gamma = gamma
        self.S_0 = S_0

        self.R0 = self.beta / self.gamma
        if verbose:
            print('R0:{R0}'.format(R0=self.R0))

    def model(self, t, y):
        S = y[0]
        I = y[1]
        R = y[2]

        ret = [-self.beta_model * S * I / self.S_0_model,   # S
               self.beta_model * S * I / self.S_0_model - self.gamma_model * I,  # I
               self.gamma_model * I]  # R
        return ret

    def loss(self, point):
        """
        RMSE between actual confirmed cases and the estimated infectious people with given beta and gamma.
        """
        size = self.I_actual.shape[0]
        beta, gamma, S_0p = point

        self.beta_model = beta
        self.gamma_model = gamma
        self.S_0_model = self.N * S_0p

        # solution = solve_ivp(SIR, [0, size], [S_0, self.I_0, self.R_0], t_eval=np.arange(0, size, 1), vectorized=True)
        solution = solve_ivp(self.model, [0, size], [self.S_0_model, self.I_0, self.R_0], t_eval=np.arange(0, size, 1), vectorized=True)

        # Put more emphasis on recovered people
        alpha = self.alpha

        l1 = np.sqrt(np.mean((solution.y[1] - self.I_actual) ** 2))
        l2 = np.sqrt(np.mean((solution.y[2] - self.R_actual) ** 2))

        return alpha * l1 + (1 - alpha) * l2

    def predict(self,):
        """
        Predict how the number of people in each compartment can be changed through time toward the future.
        The model is formulated with the given beta and gamma.
        """

        predict_range = self.daysPredict

        # print(self.confirmed.index)
        new_index = self.extend_index(self.confirmed.index, predict_range)

        size = len(new_index)


        self.beta_model = self.beta
        self.gamma_model = self.gamma
        self.S_0_model = self.S_0

        self.quarantine_loc = float(self.confirmed.index.get_loc(self.quarantineDate))

        prediction = solve_ivp(self.model, [0, size], [self.S_0, self.I_0, self.R_0],
                               t_eval=np.arange(0, size, 1))

        df = pd.DataFrame({
            'I_Actual': self.I_actual.reindex(new_index),
            'R_Actual': self.R_actual.reindex(new_index),
            'S': prediction.y[0],
            'I': prediction.y[1],
            'R': prediction.y[2]
        }, index=new_index)

        self.df = df
        self.calculateNB()

    def train(self, options=None):
        """
        Run the optimization to estimate the beta and gamma fitting the given confirmed cases.
        """

        self.estimate(options=options)

        self.predict()

    def calculateNB(self):
        # Need of Beds
        #X% of new cases need hospitalization after n days
        #First tryout is right after new cases bluntly after n days
        self.df['hospDemand'] = ((self.df['S'].shift(1) - self.df['S']) * self.hospRate).shift(self.daysToHosp).copy()
        self.df['hospExpire'] = self.df['hospDemand'].shift(self.hospitalDuration).copy()
        self.df['H'] = (self.df['hospDemand'].cumsum() - self.df['hospExpire'].cumsum()).copy()
        self.df.drop(['hospDemand', 'hospExpire'], axis=1, inplace=True)

    def rollingBetas(self):

        betasList = []
        gammasList = []
        I_actual = self.I_actual.copy()
        R_actual = self.R_actual.copy()
        F_actual = self.F_actual.copy()
        for date in self.confirmed.index:
            self.I_actual = I_actual.loc[:date]
            self.R_actual = R_actual.loc[:date]
            self.F_actual = F_actual.loc[:date]
            self.estimate(verbose=False)
            betasList.append(self.beta)
            gammasList.append(self.gamma)

        self.rollingList = pd.DataFrame({'beta': betasList, 'gamma': gammasList})
        self.rollingList.index = self.I_actual.index
        return self.rollingList



############## CONSTRAINT METHODS ################

    def const_lowerBoundR0(self, point):
        "constraint has to be R0 > bounds(0) value, thus (R0 - bound) > 0"
        # self.const_lowerBoundR0_S0opt.__code__.co_varnames
        # print(**kwargs)
        # print(locals())
        beta, gamma, S_0 = point
        lowerBound = self.R0bounds[0]
        return (beta/gamma) - lowerBound

    def const_upperBoundR0(self, point):
        # print(locals())
        # print(**kwargs)
        # self.const_upperBoundR0_S0opt.__code__.co_varnames
        beta, gamma, S_0 = point
        upperBound = self.R0bounds[1]
        return upperBound - (beta/gamma)


############## VISUALIZATION METHODS ################
    def I_fit_plot(self):
        line_styles = {
            'I_Actual': '--',
            'R_Actual': '--',
        }

        self.df[['I_Actual', 'I']].loc[:self.end_data].plot(style=line_styles)

    def R_fit_plot(self):
        line_styles = {
            'I_Actual': '--',
            'R_Actual': '--',
        }
        self.df[['R_Actual', 'R']].loc[:self.end_data].plot(style=line_styles)

    def main_plot(self):
        fig, ax = plt.subplots(figsize=(15, 10))
        ax.set_title(self.country)
        line_styles ={
            'I_Actual': '--',
            'R_Actual': '--',
        }

        # color = {
        #     'I_Actual': '#FF0000',
        #     # 'R_Actual': '--',
        # }

        self.df.plot(ax=ax, style=line_styles, )

    def rollingPlot(self, export=False):
        axes = self.rollingList.plot()
        fig = axes.get_figure()

        axes.axvline(x=self.quarantineDate, color='red', linestyle='--', label='Quarentine')

        if export:
            self.rollingList.to_excel('export_RollingBetas.xlsx')


class SIHRF(SIR):
    """
    This SIR extension split the infected compartiment into non-hospital and hospital cases and the recovered group
    into recovered and fatal

    $$\frac{dS}{dt} = - \frac{\beta IS}{N}$$

    $$\frac{dIN}{dt} = (1 - \rho) \times \frac{\beta IS}{N} - \gamma_{IN} IN$$

    $$\frac{dIH}{dt} = \rho \times \frac{\beta IS}{N} - (1-\delta) \times \gamma_{IH} IH - \delta \times \omega_{IH} IH$$


    $$\frac{dR}{dt} = \gamma_{IN} IN + (1-\delta) \times \gamma_{IH} IH $$

    $$\frac{dF}{dt} = \delta \times \omega_{IH} IH$$
    """
    def __init__(self,
                 gamma_i_bounds=(1/(3*7), 1/(2*7)),
                 gamma_h_bounds=(1/(6*7), 1/(2*7)),
                 omega_bounds=(0.001, 0.1),
                 delta_bounds=(0, 1),
                 alphas=(1/3, 1/3, 1/3),

                 **kwargs):
        self.gamma_i_bounds = gamma_i_bounds
        self.gamma_h_bounds = gamma_h_bounds
        self.omega_bounds = omega_bounds
        self.delta_bounds = delta_bounds
        self.alphas = alphas

        self.rho = kwargs['hospRate']
        super().__init__(**kwargs)


    def model(self, t, y):
        S = y[0]
        I_n = y[1]
        H_r = y[2]
        H_f = y[3]
        R = y[4]
        F = y[5]

        I = I_n + H_r + H_f

        ret = [
            # S - Susceptible
            -self.beta_model * I * S / self.S_0_model,

            # I_n
            (1 - self.rho) * self.beta_model * I * S / self.S_0_model  # (1-rho) BIS/N
            - self.gamma_I_model * I_n,  # Gamma_I x I_n

            # H_r
            self.rho * (1 - self.delta_model) * self.beta_model * I * S / self.S_0_model  # rho * (1-delta) BIS/N
            - self.gamma_H_model * H_r,

            # H_f
            self.rho * self.delta_model * self.beta_model * I * S / self.S_0_model  # rho * (delta) BIS/N
            - self.omega_model * H_f,

            # R
            self.gamma_I_model * I_n  # gamma_I * In
            + self.gamma_H_model * H_r,  # gamma_H * Hr

            # F
            self.omega_model * H_f,
        ]

        return ret

    def load_data(self):
        """
        New function to use our prop data
        """
        super().load_data()

        #True data series
        self.R_actual = self.recovered
        self.F_actual = self.fatal
        self.I_actual = self.confirmed - self.R_actual - self.F_actual  # obs this is total I

    def estimate(self, verbose=True, options=None):
        """
        List of parameters to estimate:
        * beta
        * gamma I
        * gamma H
        * omega
        * S0
        * delta
        * maybe I0 has to be initialized or estimated

        Note: gamma bounds are applied to total gamma (gamma_I + (1-delta) gamma_H + delta omega)
        """
        self.quarantine_loc = float(self.confirmed.index.get_loc(self.quarantineDate))
        betaBounds = self.betaBounds
        S0pbounds = self.S0pbounds
        gamma_i_bounds = self.gamma_i_bounds
        gamma_h_bounds = self.gamma_h_bounds
        omega_bounds = self.omega_bounds
        deltaBounds = self.delta_bounds

        constraints = [
            {'type': 'ineq', 'fun': self.const_lowerBound_R0},
            {'type': 'ineq', 'fun': self.const_upperBound_R0},
            {'type': 'ineq', 'fun': self.const_lowerBound_gamma},
            {'type': 'ineq', 'fun': self.const_upperBound_gamma},
        ]


        optimal = minimize(
            self.loss,
            np.array([
                0.2,  # beta
                .07,  # gamma I
                0.07,  # gamma H
                0.07,  # omega
                .2,  # S0p
                0.05,  # delta
            ]),
            args=(),
            method='SLSQP',
            bounds=[
                betaBounds,
                # genericGammaBounds,
                # genericGammaBounds,
                # genericGammaBounds,
                gamma_i_bounds,
                gamma_h_bounds,
                omega_bounds,
                S0pbounds,
                deltaBounds,

            ],
            constraints=constraints,
            options=options,
        )
        self.optimizer = optimal
        beta, gamma_I, gamma_H, omega, S_0p, delta = optimal.x

        # NOTE this gamma formula only works because beta_IN = beta_H
        gamma = (1-self.rho) * gamma_I + self.rho * ((1 - delta) * gamma_H + delta * omega)

        S_0 = self.N * S_0p - self.I_0 - self.H_0 - self.R_0 - self.F_0

        if verbose:
            print("Beta:{beta} Gamma:{gamma} S_0:{S_0}".format(beta=beta, gamma=gamma, S_0=S_0))
            print("Beta:{value} ".format(value=beta,))
            print("Gamma:{value} ".format(value=gamma, ))
            print("Gamma I:{value} | {values2} days".format(value=gamma_I, values2=1 / gamma_I, ))
            print("Gamma H:{value} | {values2} days".format(value=gamma_H, values2=1 / gamma_H))
            print("Omega:{value} | {values2} days".format(value=omega, values2=1 / omega))
            print("S0:{value} ".format(value=S_0, ))
            print("Delta:{value} ".format(value=delta, ))

        self.beta = beta
        self.gamma = gamma
        self.S_0 = S_0
        self.gamma = gamma
        self.gamma_I = gamma_I
        self.gamma_H = gamma_H
        self.omega = omega
        self.delta = delta
        self.R0 = self.beta / self.gamma
        if verbose:
            print('R0:{R0}'.format(R0=self.R0))

    def initialize_parameters(self):
        self.R_0 = self.recovered[0]
        self.F_0 = self.fatal[0]
        self.I_0 = self.confirmed.iloc[0] - self.R_0 - self.F_0
        self.I_n_0 = self.I_0 * (1 - self.rho)
        self.H_r_0 = self.rho * (1 - 79/165) * self.I_0 #TODO Might be a strong assumption, check sensitivity
        self.H_f_0 = self.rho * (79/165) * self.I_0
        self.H_0 = self.H_r_0 + self.H_f_0

    def loss(self, point):
        """
        RMSE between actual confirmed cases and the estimated infectious people with given beta and gamma.
        """
        size = self.I_actual.shape[0]
        beta, gamma_I, gamma_H, omega, S_0p, delta = point

        gamma = (1-self.rho) * gamma_I + self.rho * ((1 - delta) * gamma_H + delta * omega) #TODO check gamma calc

        self.gamma_model = gamma

        self.beta_model = beta
        self.gamma_I_model = gamma_I
        self.gamma_H_model = gamma_H
        self.omega_model = omega
        self.delta_model = delta
        self.S_0_model = self.N * S_0p - self.I_0 - self.H_0 - self.R_0 - self.F_0



        # solution = solve_ivp(SIR, [0, size], [S_0, self.I_0, self.R_0], t_eval=np.arange(0, size, 1), vectorized=True)
        solution = solve_ivp(self.model, [0, size], [self.S_0_model, self.I_n_0, self.H_r_0, self.H_f_0, self.R_0, self.F_0],
                             t_eval=np.arange(0, size, 1), vectorized=True)

        y = solution.y
        S = y[0]
        I_n = y[1]
        H_r = y[2]
        H_f = y[3]
        R = y[4]
        F = y[5]

        I = I_n + H_r + H_f

        # Put more emphasis on recovered people
        alphas = self.alphas

        l1 = np.sqrt(np.mean(((I) - self.I_actual) ** 2))
        l2 = np.sqrt(np.mean((R - self.R_actual) ** 2))
        l3 = np.sqrt(np.mean((F - self.F_actual) ** 2))

        loss = alphas[0] * l1 + alphas[1] * l2 + alphas[2] * l3

        #print(S_0p, loss)

        return loss

    def predict(self,):
        """
        Predict how the number of people in each compartment can be changed through time toward the future.
        The model is formulated with the given beta and gamma.
        """

        predict_range = self.daysPredict

        # print(self.confirmed.index)
        new_index = self.extend_index(self.confirmed.index, predict_range)

        size = len(new_index)

        self.quarantine_loc = float(self.confirmed.index.get_loc(self.quarantineDate))

        prediction = solve_ivp(self.model, [0, size], [self.S_0_model, self.I_n_0, self.H_r_0, self.H_f_0, self.R_0, self.F_0],
                             t_eval=np.arange(0, size, 1), vectorized=True)

        y = prediction.y
        S = y[0]
        I_n = y[1]
        H_r = y[2]
        H_f = y[3]
        R = y[4]
        F = y[5]
        H = H_r + H_f
        I = I_n + H

        df = pd.DataFrame({
            'I_Actual': self.I_actual.reindex(new_index),
            'R_Actual': self.R_actual.reindex(new_index),
            'F_Actual': self.F_actual.reindex(new_index),
            'S': S,
            'I': I,
            'I_n': I_n,
            'H_r': H_r,
            'H_f': H_f,
            'H': H,
            'R': R,
            'F': F,
        }, index=new_index)

        self.df = df

    def rollingHosp(self):

        hospList = []
        gammasList = []
        I_actual = self.I_actual.copy()
        R_actual = self.R_actual.copy()
        F_actual = self.F_actual.copy()

        for date in self.confirmed.index:
            date1 = date + dt.timedelta(days=1)
            self.I_actual = I_actual.loc[:date1]
            self.R_actual = R_actual.loc[:date1]
            self.F_actual = F_actual.loc[:date1]
            self.estimate(verbose=False)
            self.predict()  #TODO CHECK IF THIS IS OK AND NO INDENXING IS NECESSARY
            hospList.append(self.df['H'].max())

        self.rollingHospList = pd.DataFrame({'H_max': hospList,})
        self.rollingHospList.index = self.I_actual.index
        return self.rollingHospList

############## VISUALIZATION METHODS ################
    def main_plot(self):
        fig, ax = plt.subplots(figsize=(15, 10))
        ax.set_title(self.country)
        line_styles ={
            'I_Actual': '--',
            'R_Actual': '--',
            'F_Actual': '--',
        }

        # color = {
        #     'I_Actual': '#FF0000',
        #     # 'R_Actual': '--',
        # }

        self.df.plot(ax=ax, style=line_styles, )

    def I_plot(self):
        line_styles = {
            'I_Actual': '--',
            'R_Actual': '--',
            'F_Actual': '--',
        }

        self.df[['I_Actual', 'I', 'I_n', 'H']].loc[:self.end_data].plot(style=line_styles)

    def H_F_plot(self):
        line_styles = {
            'I_Actual': '--',
            'R_Actual': '--',
            'F_Actual': '--',
        }

        self.df[['H', 'F', 'F_Actual',]].loc[:self.end_data].plot(style=line_styles)

    def F_fit_plot(self):
        line_styles = {
            'I_Actual': '--',
            'R_Actual': '--',
            'F_Actual': '--',
        }

        self.df[['F_Actual', 'F']].loc[:self.end_data].plot(style=line_styles)

    def actuals_plot(self):
        line_styles = {
            'I_Actual': '--',
            'R_Actual': '--',
            'F_Actual': '--',
        }

        self.df[['I_Actual', 'R_Actual', 'F_Actual']].loc[:self.end_data].plot(style=line_styles)

    def rollingHospPlot(self, export=False):
        axes = self.rollingHospList.plot()
        fig = axes.get_figure()
        if export:
            self.rollingHospList.to_excel('export_HospPlot.xlsx')
        axes.axvline(x=self.quarantineDate, color='red', linestyle='--', label='Quarentine')
        axes.axhline(y=50000, color='black', linestyle='--', label='Hospital Capacity')

############## CONSTRAINT METHODS ################

    def const_lowerBound_gamma(self, point):
        "constraint has to be R0 > bounds(0) value, thus (R0 - bound) > 0"

        lowerBound = self.gammaBounds[0]

        beta, gamma_I, gamma_H, omega, S0, delta = point

        gamma = gamma_I + (1-delta) * gamma_H + delta * omega

        return gamma - lowerBound

    def const_upperBound_gamma(self, point):

        upperBound = self.gammaBounds[1]

        beta, gamma_I, gamma_H, omega, S0, delta = point

        gamma = gamma_I + (1 - delta) * gamma_H + delta * omega

        return upperBound - gamma

    def const_lowerBound_R0(self, point):
        "constraint has to be R0 > bounds(0) value, thus (R0 - bound) > 0"

        lowerBound = self.R0bounds[0]

        beta, gamma_I, gamma_H, omega, S0, delta = point

        gamma = gamma_I + (1 - delta) * gamma_H + delta * omega
        R0 = beta / gamma
        return R0 - lowerBound

    def const_upperBound_R0(self, point):

        upperBound = self.R0bounds[1]

        beta, gamma_I, gamma_H, omega, S0, delta = point

        gamma = gamma_I + (1 - delta) * gamma_H + delta * omega
        R0 = beta / gamma
        return upperBound - R0

class SIHRF_Sigmoid(SIHRF):
    """
    This SIR extension split the infected compartiment into non-hospital and hospital cases and the recovered group
    into recovered and fatal

    $$\frac{dS}{dt} = - \frac{\beta IS}{N}$$

    $$\frac{dIN}{dt} = (1 - \rho) \times \frac{\beta IS}{N} - \gamma_{IN} IN$$

    $$\frac{dIH}{dt} = \rho \times \frac{\beta IS}{N} - (1-\delta) \times \gamma_{IH} IH - \delta \times \omega_{IH} IH$$


    $$\frac{dR}{dt} = \gamma_{IN} IN + (1-\delta) \times \gamma_{IH} IH $$

    $$\frac{dF}{dt} = \delta \times \omega_{IH} IH$$
    """
    def __init__(self,
                 lambda_bounds=(0.25, 4),
                 **kwargs):
        self.lambda_bounds = lambda_bounds
        super().__init__(**kwargs)

    def sigmoid(self, t):
        # Normalize t
        t = t - self.sig_normal_t
        return (self.beta1_model - self.beta2_model) / (1 + np.exp(t / self.lambda_model)) + self.beta2_model

    def model(self, t, y):
        S = y[0]
        I_n = y[1]
        H_r = y[2]
        H_f = y[3]
        R = y[4]
        F = y[5]

        I = I_n + H_r + H_f
        self.beta_model = self.sigmoid(t)

        ret = [
            # S - Susceptible
            -self.beta_model * I * S / self.S_0_model,

            # I_n
            (1 - self.rho) * self.beta_model * I * S / self.S_0_model  # (1-rho) BIS/N
            - self.gamma_I_model * I_n,  # Gamma_I x I_n

            # H_r
            self.rho * (1 - self.delta_model) * self.beta_model * I * S / self.S_0_model  # rho * (1-delta) BIS/N
            - self.gamma_H_model * H_r,

            # H_f
            self.rho * self.delta_model * self.beta_model * I * S / self.S_0_model  # rho * (delta) BIS/N
            - self.omega_model * H_f,

            # R
            self.gamma_I_model * I_n  # gamma_I * In
            + self.gamma_H_model * H_r,  # gamma_H * Hr

            # F
            self.omega_model * H_f,
        ]

        return ret

    def estimate(self, verbose=True, options=None):
        """
        List of parameters to estimate:
        * beta
        * gamma I
        * gamma H
        * omega
        * S0
        * delta
        * maybe I0 has to be initialized or estimated

        Note: gamma bounds are applied to total gamma (gamma_I + (1-delta) gamma_H + delta omega)
        """
        self.quarantine_loc = float(self.confirmed.index.get_loc(self.quarantineDate))
        self.sig_normal_t = self.quarantine_loc + 7
        betaBounds = self.betaBounds
        S0pbounds = self.S0pbounds
        gamma_i_bounds = self.gamma_i_bounds
        gamma_h_bounds = self.gamma_h_bounds
        omega_bounds = self.omega_bounds
        deltaBounds = self.delta_bounds
        lambdaBounds = self.lambda_bounds

        constraints = [
            {'type': 'ineq', 'fun': self.const_lowerBound_R0},
            {'type': 'ineq', 'fun': self.const_upperBound_R0},
            {'type': 'ineq', 'fun': self.const_lowerBound_gamma},
            {'type': 'ineq', 'fun': self.const_upperBound_gamma},
            {'type': 'ineq', 'fun': self.const_betas},
        ]


        optimal = minimize(
            self.loss,
            np.array([
                0.2,  # beta1
                0.2,  # beta2
                .07,  # gamma I
                0.07,  # gamma H
                0.07,  # omega
                .2,  # S0p
                # None,  # S0p
                0.05,  # delta
                1,  # lambda
            ]),
            args=(),
            method='SLSQP',
            bounds=[
                betaBounds,
                betaBounds,
                gamma_i_bounds,
                gamma_h_bounds,
                omega_bounds,
                S0pbounds,
                deltaBounds,
                lambdaBounds,
            ],
            constraints=constraints,
            options=options,
        )
        self.optimizer = optimal
        beta1, beta2, gamma_I, gamma_H, omega, S_0p, delta, lamb = optimal.x

        # NOTE this gamma formula only works because beta_IN = beta_H
        gamma = (1-self.rho) * gamma_I + self.rho * ((1 - delta) * gamma_H + delta * omega)

        S_0 = self.N * S_0p - self.I_0 - self.H_0 - self.R_0 - self.F_0

        if verbose:
            print("Beta1:{beta1} Beta2:{beta2} Gamma:{gamma} S_0:{S_0}".format(beta1=beta1, beta2=beta2, gamma=gamma, S_0=S_0))
            print("Beta1:{value} ".format(value=beta1,))
            print("Beta2:{value} ".format(value=beta2, ))
            print("Gamma:{value} | {values2} days ".format(value=gamma, values2=1/gamma))
            print("Gamma I:{value} | {values2} days".format(value=gamma_I, values2=1/gamma_I,))
            print("Gamma H:{value} | {values2} days".format(value=gamma_H, values2=1/gamma_H))
            print("Omega:{value} | {values2} days".format(value=omega, values2=1/omega))
            print("S0:{value} ".format(value=S_0, ))
            print("Delta:{value} ".format(value=delta, ))
            print("Lambda:{value} ".format(value=lamb, ))

        self.beta1 = beta1
        self.beta2 = beta2
        self.gamma = gamma
        self.S_0 = S_0
        self.gamma = gamma
        self.gamma_I = gamma_I
        self.gamma_H = gamma_H
        self.omega = omega
        self.delta = delta
        self.lamb = lamb
        self.R01 = self.beta1 / self.gamma
        self.R02 = self.beta2 / self.gamma
        if verbose:
            print('R0_initial:{R0}'.format(R0=self.R01))
            print('R0_quarantine:{R0}'.format(R0=self.R02))

    def loss(self, point):
        """
        RMSE between actual confirmed cases and the estimated infectious people with given beta and gamma.
        """
        size = self.I_actual.shape[0]
        beta1, beta2, gamma_I, gamma_H, omega, S_0p, delta, lamb = point

        gamma = (1-self.rho) * gamma_I + self.rho * ((1 - delta) * gamma_H + delta * omega) #TODO check gamma calc

        self.gamma_model = gamma
        self.beta1_model = beta1
        self.beta2_model = beta2
        self.gamma_I_model = gamma_I
        self.gamma_H_model = gamma_H
        self.omega_model = omega
        self.delta_model = delta
        self.lambda_model = lamb

        self.S_0_model = self.N * S_0p - self.I_0 - self.H_0 - self.R_0 - self.F_0



        # solution = solve_ivp(SIR, [0, size], [S_0, self.I_0, self.R_0], t_eval=np.arange(0, size, 1), vectorized=True)
        solution = solve_ivp(self.model, [0, size], [self.S_0_model, self.I_n_0, self.H_r_0, self.H_f_0, self.R_0, self.F_0],
                             t_eval=np.arange(0, size, 1), vectorized=True)

        y = solution.y
        S = y[0]
        I_n = y[1]
        H_r = y[2]
        H_f = y[3]
        R = y[4]
        F = y[5]

        I = I_n + H_r + H_f

        # Put more emphasis on recovered people
        alphas = self.alphas

        l1 = np.sqrt(np.mean((I - self.I_actual) ** 2))
        l2 = np.sqrt(np.mean((R - self.R_actual) ** 2))
        l3 = np.sqrt(np.mean((F - self.F_actual) ** 2))

        loss = alphas[0] * l1 + alphas[1] * l2 + alphas[2] * l3

        # print(S_0p, loss)

        return loss


############## VISUALIZATION METHODS ################

############## CONSTRAINT METHODS ################

    def const_betas(self, point):
        # print(locals())
        # print(**kwargs)
        # self.const_upperBoundR0_S0opt.__code__.co_varnames
        beta1, beta2, gamma_I, gamma_H, omega, S_0p, delta, lamb = point
        return beta1 - beta2

    def const_lowerBound_gamma(self, point):
        "constraint has to be R0 > bounds(0) value, thus (R0 - bound) > 0"

        lowerBound = self.gammaBounds[0]

        beta1, beta2, gamma_I, gamma_H, omega, S_0p, delta, lamb = point

        gamma = gamma_I + (1-delta) * gamma_H + delta * omega

        return gamma - lowerBound

    def const_upperBound_gamma(self, point):

        upperBound = self.gammaBounds[1]

        beta1, beta2, gamma_I, gamma_H, omega, S_0p, delta, lamb = point

        gamma = gamma_I + (1 - delta) * gamma_H + delta * omega

        return upperBound - gamma

    def const_lowerBound_R0(self, point):
        "constraint has to be R0 > bounds(0) value, thus (R0 - bound) > 0"

        lowerBound = self.R0bounds[0]

        beta1, beta2, gamma_I, gamma_H, omega, S_0p, delta, lamb = point

        gamma = gamma_I + (1 - delta) * gamma_H + delta * omega
        R0_1 = beta1 / gamma
        R0_2 = beta2 / gamma
        return min(R0_1, R0_2) - lowerBound

    def const_upperBound_R0(self, point):

        upperBound = self.R0bounds[1]

        beta1, beta2, gamma_I, gamma_H, omega, S_0p, delta, lamb = point

        gamma = gamma_I + (1 - delta) * gamma_H + delta * omega
        R0_1 = beta1 / gamma
        R0_2 = beta2 / gamma
        return upperBound - max(R0_1, R0_2)

class SIR_sigmoid(SIR):
    """
    Implements a SIR with a time varying beta according to a sigmoid function
    """
    def __init__(self,
                 lambda_bounds=(0.1, 4),
                 **kwargs):
        self.lambda_bounds = lambda_bounds
        super().__init__(**kwargs)


    def sigmoid(self, t):
        # Normalize t
        t = t - self.sig_normal_t
        return (self.beta1_model - self.beta2_model) / (1 + np.exp(t / self.lambda_model)) + self.beta2_model

    def model(self, t, y):
        S = y[0]
        I = y[1]
        R = y[2]

        self.beta_model = self.sigmoid(t)

        ret = [-self.beta_model * S * I / self.S_0_model, self.beta_model * S * I / self.S_0_model - self.gamma_model * I,
               self.gamma_model * I]
        return ret

    def estimate(self, verbose=True, options=None):

        self.quarantine_loc = float(self.confirmed.index.get_loc(self.quarantineDate))
        self.sig_normal_t = self.quarantine_loc + 7


        betaBounds = self.betaBounds
        gammaBounds = self.gammaBounds
        S0pbounds = self.S0pbounds
        lambdaBounds = self.lambda_bounds


        constraints = [
                {'type': 'ineq', 'fun': self.const_lowerBoundR0},
                {'type': 'ineq', 'fun': self.const_upperBoundR0},
                {'type': 'ineq', 'fun': self.const_betas},
            ]

        optimal = minimize(
            self.loss,
            [0.2, 0.2, 1, 0.07, 0.01],
            args=(),
            method='SLSQP',
            # options={'maxiter' : 5},
            # method='TNC',
            bounds=[betaBounds, betaBounds, lambdaBounds, gammaBounds, S0pbounds],
            constraints=constraints
        )
        self.optimizer = optimal
        beta1, beta2, lamb, gamma, S_0p = optimal.x
        S_0 = self.N * S_0p
        if verbose:
            print("Beta1:{beta1} Beta2:{beta2} Lambda:{lamb} Gamma:{gamma} S_0:{S_0}".format(beta1=beta1, beta2=beta2,
                                                                                             lamb=lamb, gamma=gamma, S_0=S_0))
        self.beta1 = beta1
        self.beta2 = beta2
        self.lamb = lamb
        self.gamma = gamma
        self.S_0 = S_0

        self.R01 = self.beta1 / self.gamma
        self.R02 = self.beta2 / self.gamma
        if verbose:
            print('R0_initial:{R0}'.format(R0=self.R01))
            print('R0_quarentine:{R0}'.format(R0=self.R02))

    def loss(self, point):
        """
        RMSE between actual confirmed cases and the estimated infectious people with given beta and gamma.
        """
        size = self.I_actual.shape[0]
        beta1, beta2, lamb, gamma, S_0p = point

        self.beta1_model = beta1
        self.beta2_model = beta2
        self.lambda_model = lamb
        self.gamma_model = gamma
        self.S_0_model = self.N * S_0p

        # solution = solve_ivp(SIR, [0, size], [S_0, self.I_0, self.R_0], t_eval=np.arange(0, size, 1), vectorized=True)
        solution = solve_ivp(self.model, [0, size], [self.S_0_model, self.I_0, self.R_0], t_eval=np.arange(0, size, 1), vectorized=True)

        # Put more emphasis on recovered people
        alpha = self.alpha

        l1 = np.sqrt(np.mean((solution.y[1] - self.I_actual) ** 2))
        l2 = np.sqrt(np.mean((solution.y[2] - self.R_actual) ** 2))

        return alpha * l1 + (1 - alpha) * l2

    def predict(self,):
        """
        Predict how the number of people in each compartment can be changed through time toward the future.
        The model is formulated with the given beta and gamma.
        """

        predict_range = self.daysPredict

        # print(self.confirmed.index)
        new_index = self.extend_index(self.confirmed.index, predict_range)

        size = len(new_index)

        self.quarantine_loc = float(self.confirmed.index.get_loc(self.quarantineDate))

        prediction = solve_ivp(self.model, [0, size], [self.S_0, self.I_0, self.R_0],
                               t_eval=np.arange(0, size, 1))

        df = pd.DataFrame({
            'I_Actual': self.I_actual.reindex(new_index),
            'R_Actual': self.R_actual.reindex(new_index),
            'S': prediction.y[0],
            'I': prediction.y[1],
            'R': prediction.y[2],
        }, index=new_index)

        self.df = df
        self.calculateNB()


############## CONSTRAINT METHODS ################

    def const_lowerBoundR0(self, point):
        "constraint has to be R0 > bounds(0) value, thus (R0 - bound) > 0"
        # self.const_lowerBoundR0_S0opt.__code__.co_varnames
        # print(**kwargs)
        # print(locals())
        beta1, beta2, lamb, gamma, S_0 = point
        lowerBound = self.R0bounds[0]
        beta = (beta1 + beta2) / 2
        return beta/gamma - lowerBound

    def const_upperBoundR0(self, point):
        # print(locals())
        # print(**kwargs)
        # self.const_upperBoundR0_S0opt.__code__.co_varnames
        beta1, beta2, lamb, gamma, S_0 = point
        upperBound = self.R0bounds[1]
        beta = (beta1 + beta2) / 2
        return upperBound - (beta/gamma)

    def const_betas(self, point):
        # print(locals())
        # print(**kwargs)
        # self.const_upperBoundR0_S0opt.__code__.co_varnames
        beta1, beta2, lamb, gamma, S_0 = point
        return beta1 - beta2





class SEIR(SIR):
    def __init__(self,
                 incubationPeriod = 7,
                 forceE0 = None,
                 **kwargs):
        self.forceE0 = forceE0
        self.incubationPeriod = incubationPeriod
        self.sigma = 1 / incubationPeriod
        super().__init__(**kwargs)


    def initialize_parameters(self):
        ## incubated E_0 = 1/incPeriod * each of the following incPeriod days
        #TODO check
        self.E_0 = int(((self.confirmed - self.recovered - self.fatal).iloc[0:(self.incubationPeriod-1)]).sum() / self.incubationPeriod)
        if self.forceE0:
            self.E_0 = self.forceE0
        self.R_0 = self.recovered[0] + self.fatal[0]
        self.I_0 = (self.confirmed.iloc[0] - self.R_0)

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
                ret = [-beta * S * I / self.S_0, beta * S * I / self.S_0 - sigma * E, sigma * E - gamma * I, gamma * I]
            else:
                ret = [-beta2 * S * I / self.S_0, beta2 * S * I / self.S_0 - sigma * E, sigma * E - gamma * I, gamma * I]
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

        sigma = self.sigma

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
            E = y[1]
            I = y[2]
            R = y[3]

            if t < self.quarantine_loc:
                ret = [-beta * S * I / self.S_0, beta * S * I / self.S_0 - sigma * E, sigma * E - gamma * I, gamma * I]
            else:
                ret = [-beta2 * S * I / self.S_0, beta2 * S * I / self.S_0 - sigma * E, sigma * E - gamma * I, gamma * I]
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

        # if self.log:
        #     df = df.transform(np.exp)

        self.df = df
        print("Predicting with Beta:{beta} Beta2: {beta2} Gamma:{gamma} Sigma:{sigma}".format(sigma=sigma,beta=beta, gamma=gamma, beta2=beta2))


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
        self.R_0 = N
        self.infectedAssumption = infectedAssumption  # Multiplier to account for non reported
        self.recoveredAssumption = recoveredAssumption
        self.R_0th = nth  # minimum number of cases to start modelling
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
        nth_index = self.confirmed[self.confirmed >= self.R_0th].index[0]

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
        self.S_0 = self.R_0 - self.R_0 - self.I_0

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
                ret = [-beta * S * I / self.R_0, beta * S * I / self.R_0 - sigma * E, sigma * E - gamma * I, gamma * I]
            else:
                ret = [-beta2 * S * I / self.R_0, beta2 * S * I / self.R_0 - sigma * E, sigma * E - gamma * I, gamma * I]
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
                ret = [-beta * S * I / self.R_0, beta * S * I / self.R_0 - sigma * E, sigma * E - gamma * I, gamma * I]
            else:
                ret = [-beta2 * S * I / self.R_0, beta2 * S * I / self.R_0 - sigma * E, sigma * E - gamma * I, gamma * I]
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
                ret = [(-beta * S * I / self.R_0) + S, (beta * S * I / self.R_0 - gamma * I) + I, (gamma * I) + R]
            elif t < self.back_loc:
                ret = [(-beta3 * S * I / self.R_0) + S, (beta3 * S * I / self.R_0 - gamma * I) + I, (gamma * I) + R]
            else:
                ret = [(-beta2 * S * I / self.R_0) + S, (beta2 * S * I / self.R_0 - gamma * I) + I, (gamma * I) + R]
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
    hospRate = 0.05
    # deltaUpperBound = 0.035 / hospRate
    deltaUpperBound = 79 / 165
    gi = 0.07
    gh = 0.07
    omega = 0.07

    # Mudar omega bouds /
    NB = 16e6

    t1 = SIHRF(country='Brazil',
               N=200e6,
               # N=1e6,
               alpha=.7,
               nth=100,
               daysToHosp=4,  # big for detction
               daysToLeave=12,
               daysPredict=150,
               infectedAssumption=1,
               # forcedBeta = 3,
               quarantineDate=dt.datetime(2020, 3, 24),  # italy lockdown was on the 9th
               # estimateBeta2 = True
               # opt='SLSQP',
               R0bounds=(0, 20),
               hospRate=hospRate,

               # Usual restrictions
               S0pbounds=(15e6 / 200e6, 15e6 / 200e6),
               delta_bounds=(0, deltaUpperBound),
               betaBounds=(0.1, 1.5),
               gammaBounds=(0.01, .2),
               gamma_i_bounds=(1 / (5 * 7), 1 / (1 * 7)),
               gamma_h_bounds=(1 / (8 * 7), 1 / (1 * 7)),
               omega_bounds=(1 / (6 * 7), 1 / (3)),

               # restricted
               # S0pbounds=(.5e6 / 200e6, 50e6 / 200e6),
               # delta_bounds=(deltaUpperBound, deltaUpperBound),
               # betaBounds=(0.3, 0.3),
               # gammaBounds=(0, 1),
               # gamma_i_bounds=(gi, gi),
               # gamma_h_bounds=(gh, gh),
               # omega_bounds=(omega, omega),

               # omega_bounds=(1/12, 1/12),
               alphas=(.015, .0, .985),
               adjust_recovered=True,
               )

    t1.train()
    # options={'eps': 5e-3, }
    # options={'eps': 1e-3, 'ftol': 1e-7}

    # t1.main_plot()

    print(t1.gamma_i_bounds)
    print(t1.gamma_h_bounds)
    print(t1.omega_bounds)
    print('\nI Max:')
    print(t1.df.I.max())
    print('Est:')
    print(t1.df.I.max() * .15)
    print('H Max:')
    print(t1.df['H'].max())
    print('R Max:')
    print(t1.df['R'].max())
    print('F Max:')
    print(t1.df['F'].max())
    print('F+R Max:')
    print(t1.df['F'].max() + t1.df['R'].max())
    # (t1.df.S + t1.df.I + t1.df.R + t1.df.F)

    # t1.optimizer
    t1.rollingBetas()
    # t1.rollingHospPlot()