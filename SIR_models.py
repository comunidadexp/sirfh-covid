
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from datetime import timedelta, datetime
import datetime as dt

yellow = (240/255, 203/255, 105/255)
grey = (153/255, 153/255, 153/255)
faded_grey = (153/255, 153/255, 153/255, .25)
red = (220/255, 83/255, 86/255)

class SIR(object):
    def __init__(self,
                 country='Brazil',
                 nth=100,
                 daysPredict=150,
                 alpha=[.5, .5],
                 parameter_bounds={},
                 constraints_bounds={},
                 force_parameters={},
                 # betaBounds=(0.00000001, 2.0),
                 # gammaBounds=(0.00000001, 2.0),
                 # S0pbounds=(10000, 10e6),
                 # R0bounds=None,
                 hospitalization_rate=0.05,
                 adjust_recovered=True,
                 cut_sample_date=None,
                 ):

        self.all_attributes = locals()
        del self.all_attributes['self']
        self.constraints_bounds = constraints_bounds
        self.country = country
        self.nth = nth  # minimum number of cases to start modelling
        self.daysPredict = daysPredict
        self.alpha = alpha
        self.parameter_bounds = parameter_bounds
        self.force_parameters = force_parameters
        self.cut_sample_date = cut_sample_date

        initial_guesses = {
            'beta': .2,
            'gamma': .07,
            'S0p': .05
        }

        if hasattr(self, 'initial_guesses'):
            self.initial_guesses = {**initial_guesses, **self.initial_guesses}
        else:
            self.initial_guesses = {
                'beta': .2,
                'gamma': .07,
                'S0p': .05
            }

        self.hospitalization_rate = hospitalization_rate
        self.adjust_recovered = adjust_recovered

        self.load_data()

        self.end_data = self.confirmed.index.max()
        self.quarantine_loc = float(self.confirmed.index.get_loc(self.quarantine_date))
        self.model_type = 'SIR'

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

    def load_population(self, dir="Population.xlsx"):
        """
        This function loads the country's population from an excel spreadsheet that should have a list of countries
        on the first column and the population on the second. The sheet headers should be ´Country´ and ´Population´.

        The function saves the population to ´self.country_population´

        :param dir: path do file
        :return: None
        """

        df = pd.read_excel(dir).set_index('Country')
        self.country_population = df.loc[self.country][0]

    def load_quarantine_date(self, dir="Quarantine_dates.xlsx"):
        """
        This function loads the country's quarantine date from an excel spreadsheet that should have a list of countries
        on the first column and the population on the second. The sheet headers should be ´Country´ and ´Quarantine´.

        The function saves the population to ´self.quarantine_date´

        :param dir: path do file
        :return: None
        """

        df = pd.read_excel(dir).set_index('Country')

        if self.country in df.index:
            self.quarantine_date = df.loc[self.country][0]
            self.quarantine_loc = float(self.confirmed.index.get_loc(self.quarantine_date))
        else:
            self.quarantine_date = self.confirmed.index[-1]

    def load_data(self):
        """
        New function to use our prop data
        """
        self.load_CSSE()
        self.load_population()

        # Adjust recovered curve
        if self.adjust_recovered:
            self.recovered = self.smoothCurve(self.recovered)

        # find date in which nth case is reached
        nth_index = self.confirmed[self.confirmed >= self.nth].index[0]

        self.load_quarantine_date()
        quarantine_index = pd.Series(False, index=self.confirmed.index)
        quarantine_index[quarantine_index.index >= self.quarantine_date] = True

        self.quarantine_index = quarantine_index.loc[nth_index:]
        self.confirmed = self.confirmed.loc[nth_index:]
        self.fatal = self.fatal.loc[nth_index:]
        self.recovered = self.recovered.loc[nth_index:]

        self.initialize_parameters()

        #True data series
        self.R_actual = self.fatal + self.recovered
        self.I_actual = self.confirmed - self.R_actual

        self.build_optimization_inputs()

    def cut_sample(self):

        cutDate = self.cut_sample_date
        if cutDate:
            if not isinstance(cutDate, dt.datetime):
                cutDate = self.I_actual.index[-1] + dt.timedelta(days=-cutDate)
            # cutDate = self.F_actual.index[-1] + dt.timedelta(days=-days)
            self.I_actual = self.I_actual.loc[:cutDate].copy()
            self.R_actual = self.R_actual.loc[:cutDate].copy()
            # self.F_actual = self.F_actual.loc[:self.cut_sample_date]

    def set_default_bounds(self):
        """
        Sets the default values for unprovided bounds
        :return:
        """
        if 'R0' not in self.constraints_bounds.keys():
            self.constraints_bounds['R0'] = (0, 20)

        if 'beta' not in self.parameter_bounds.keys():
            self.parameter_bounds['beta'] = (.01, .5)

        if 'gamma' not in self.parameter_bounds.keys():
            self.parameter_bounds['gamma'] = (.01, .2)

        if 'S0p' not in self.parameter_bounds.keys():
            self.parameter_bounds['S0p'] = (.01, .12)

    def build_optimization_inputs(self):
        """
        Since we allow parameters to be forced, we need a function to create the optimization inputs.

        Also, checks bounds that weren't provided and set them to default values.

        :return:
        """

        self.set_default_bounds()

        # optimization takes two arrays, one of initial values and one of bounds (same order)
        self.variable_parameters_list = []
        self.optimization_initial_values = []
        self.optimization_bounds = []
        for param in self.parameter_bounds.keys():
            if param not in self.force_parameters.keys():
                self.variable_parameters_list.append(param)

                if 'beta' in param:
                    self.optimization_initial_values.append(self.initial_guesses['beta'])
                elif 'gamma' in param:
                    self.optimization_initial_values.append(self.initial_guesses['gamma'])
                elif 'omega' in param:
                    self.optimization_initial_values.append(self.initial_guesses['gamma'])
                elif 'S0p' in param:
                    self.optimization_initial_values.append(self.initial_guesses['S0p'])
                else:
                    self.optimization_initial_values.append(self.initial_guesses[param])

                self.optimization_bounds.append(self.parameter_bounds[param])

        self.model_params = self.wrap_parameters(self.optimization_initial_values)

        # constraints
        self.constraints = [
            {'type': 'ineq', 'fun': self.const_lowerBoundR0},
            {'type': 'ineq', 'fun': self.const_upperBoundR0},
        ]

        self.add_constraints()

    def add_constraints(self):
        """
        This function is intended to be overridden by subclasses
        """

        pass

    def wrap_parameters(self, point):
        """
        Gets a list-like array and transform it to a parameter dictionary

        :param point: list-like array
        :return: dictionary containing the parameters names as keys
        """

        dic = {}
        for i in range(0, len(self.variable_parameters_list)):
            param = self.variable_parameters_list[i]
            dic[param] = point[i]



        return {**dic, **self.force_parameters}

    def calculateS0(self, S0p):

        # return self.country_population * S0p - self.I_0 - self.H_0 - self.R_0 - self.F_0

        return self.country_population * S0p - self.I_0 - self.R_0

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

    def calculate_r0(self):
        """
        Using the ´self.params´ dictionary, calculates R0

        :return: R0
        """

        gamma = self.calculate_gamma()

        return self.params['beta'] / gamma

    def calculate_gamma(self):
        """
        Using the ´self.params´ dictionary, calculates gamma.

        :return: R0
        """

        if hasattr(self, 'model_params'):
            gamma = self.model_params['gamma']
        else:
            gamma = .07

        return gamma

    def calculate_rmse(self, actual, forecast, cutDate, verbose=False):
        # Separate only the according values on the Dfs
        mse_F_actual = actual.loc[cutDate:].copy().diff()
        mse_F_forecast = forecast.loc[cutDate:].copy().diff()
        # get the size of it
        T = mse_F_actual.shape[0]

        mse = (((mse_F_forecast - mse_F_actual) ** 2).sum() / T) ** .5

        if verbose:
            print("MSE: {mse}".format(mse=mse))

        return mse

    def estimate(self, verbose=True, options=None, loss_func=None):

        if not loss_func:
            loss_func = self.loss

        optimal = minimize(
            loss_func,
            self.optimization_initial_values,
            args=(),
            method='SLSQP',
            bounds=self.optimization_bounds,
            constraints=self.constraints,
            options=options,
        )
        self.optimizer = optimal

        params = self.wrap_parameters(optimal.x)

        self.params = {**self.force_parameters, **params}

        self.R0 = self.calculate_r0()

        if verbose:
            self.print_parameters()

    def model(self, t, y):
        S = y[0]
        I = y[1]
        R = y[2]

        S0_model = self.calculateS0(self.model_params['S0p'])

        ret = [-self.model_params['beta'] * S * I / S0_model,  # S
               self.model_params['beta']* S * I / S0_model - self.model_params['gamma'] * I,  # I
               self.model_params['gamma'] * I]  # R
        return ret

    def loss(self, point):
        """
        RMSE between actual confirmed cases and the estimated infectious people with given beta and gamma.
        """
        size = self.I_actual.shape[0]

        self.model_params = self.wrap_parameters(point)

        S0 = self.calculateS0(self.model_params['S0p'])

        # solution = solve_ivp(SIR, [0, size], [S_0, self.I_0, self.R_0], t_eval=np.arange(0, size, 1), vectorized=True)
        solution = solve_ivp(self.model, [0, size], [S0, self.I_0, self.R_0], t_eval=np.arange(0, size, 1), vectorized=True)

        # Put more emphasis on recovered people
        alpha = self.alpha

        l1 = np.sqrt(np.mean((solution.y[1] - self.I_actual) ** 2))
        l2 = np.sqrt(np.mean((solution.y[2] - self.R_actual) ** 2))

        return alpha[0] * l1 + alpha[1] * l2

    def loss_outOfSample(self, point):
        """
        Alternative loss function to be use with the out-of-sample RMSE estimation method.
        This takes *exclusively* the S0p parameter, predicts out of sample and returns the RMSE.

        :param point: parameters array
        :return: RMSE
        """
        params = self.wrap_parameters(point)
        self.model_params = params
        self.params = params

        cutDate = self.F_actual.index[-1] + dt.timedelta(days=-self.outOfSample_days)

        self.predict()

        forecast = self.df.copy()

        # Calculate MSE
        self.mse = self.calculate_rmse(self.F_actual, forecast.F, cutDate)

        return self.mse

    def predict(self,):
        """
        Predict how the number of people in each compartment can be changed through time toward the future.
        The model is formulated with the given beta and gamma.
        """

        predict_range = self.daysPredict

        # print(self.confirmed.index)
        new_index = self.extend_index(self.confirmed.index, predict_range)

        size = len(new_index)

        self.model_params = self.params

        self.quarantine_loc = float(self.confirmed.index.get_loc(self.quarantine_date))

        S0 = self.calculateS0(self.model_params['S0p'])

        prediction = solve_ivp(self.model, [0, size], [S0, self.I_0, self.R_0],
                               t_eval=np.arange(0, size, 1))

        df = pd.DataFrame({
            'I_Actual': self.I_actual.reindex(new_index),
            'R_Actual': self.R_actual.reindex(new_index),
            'S': prediction.y[0],
            'I': prediction.y[1],
            'R': prediction.y[2]
        }, index=new_index)

        self.df = df

    def train(self, options=None, loss_func=None, verbose=True):
        """
        Run the optimization to estimate parameters fitting real cases
        """

        if self.variable_parameters_list:
            self.estimate(options=options, loss_func=loss_func, verbose=verbose)
        else:
            self.params = self.force_parameters.copy()
            self.model_params = self.params.copy()


        self.predict()

    def rolling_estimation(self):
        """
        Re-estimates the model for an increasing-only window for every data point.
        :return: pandas dataframe containing the historical parameters
        """
        params_list = []
        I_actual = self.I_actual.copy()
        R_actual = self.R_actual.copy()
        F_actual = self.F_actual.copy()
        for date in self.confirmed.index:
            self.I_actual = I_actual.loc[:date]
            self.R_actual = R_actual.loc[:date]
            self.F_actual = F_actual.loc[:date]
            self.estimate(verbose=False)
            params_list.append(self.params)

        self.rolling_parameters = pd.DataFrame(params_list)
        self.rolling_parameters.index = self.I_actual.index
        return self.rolling_parameters

    def outOfSample_forecast(self, cutDate=None, plot=True, days=14, k=1):

        if not cutDate:
            cutDate = self.F_actual.index[-1] + dt.timedelta(days=-days)

        I_actual = self.I_actual.copy()
        R_actual = self.R_actual.copy()
        F_actual = self.F_actual.copy()

        self.I_actual = I_actual.loc[:cutDate]
        self.R_actual = R_actual.loc[:cutDate]
        self.F_actual = F_actual.loc[:cutDate]

        self.estimate(verbose=False)

        self.predict()

        self.forecast = self.df.copy()

        self.I_actual = I_actual.copy()
        self.R_actual = R_actual.copy()
        self.F_actual = F_actual.copy()

        # Calculate MSE
        self.mse = self.calculate_rmse(self.F_actual, self.forecast.F, cutDate, verbose=True)

        if plot:
            self.outOfSample_plot(cutDate, days=days, k=k)

        return self.mse

    def plot_forecast(self, ax, diff, window, scenarios, cutDate=None, verbose=False):

        if not cutDate:
            cutDate = self.F_actual.index[-1] + dt.timedelta(days=-window)

        scenarios_forecast = []

        # Estimate a new scenario for each `S0p` in `scenarios`
        # estimate optimal parameters
        new_args = self.all_attributes.copy()
        new_args['cut_sample_date'] = cutDate

        estimator = self.create_new_object(self.model_type, new_args)
        estimator.train(verbose=verbose)
        optimal_parameters = estimator.params.copy()

        for scenario in scenarios:
            new_args = self.all_attributes.copy()
            new_args['cut_sample_date'] = cutDate
            new_args['force_parameters'] = optimal_parameters.copy()

            # new_args['parameter_bounds']['delta'] = (optimal_parameters['delta'], optimal_parameters[
            #     'delta'])  # little trick because fixing all parameters crashes
            # del new_args['force_parameters']['delta']

            # new_args['parameter_bounds']['lambda'] = (optimal_parameters['lambda'], optimal_parameters[
            #     'lambda'])  # little trick because fixing all parameters crashes
            # del new_args['force_parameters']['lambda']

            new_args['force_parameters']['S0p'] = scenario

            estimator = self.create_new_object(self.model_type, new_args)
            estimator.train(verbose=verbose)

            scenario_forecast = estimator.df.copy().F

            # if we are looking for past out of sample forecasts, we don't need the entire projection window.
            # If we are looking for future forecasts, we can leave the entire window.
            if window > 0:
                scenario_forecast.reindex(self.F_actual.index)

            scenarios_forecast.append(scenario_forecast)

        # # Calculate MSE
        # self.mse = self.calculate_rmse(self.F_actual, self.forecast.F, cutDate, verbose=True)

        ########## PLOT ################

        actual = self.F_actual.copy()

        if diff:
            actual = actual.diff()
            for i in range(len(scenarios_forecast)):
                scenarios_forecast[i] = scenarios_forecast[i].diff()

        forecast_outOfSample = scenarios_forecast[1].loc[cutDate:(cutDate + dt.timedelta(days=window))].copy()
        forecast_inSample = scenarios_forecast[1].loc[:cutDate].copy()

        lowerBound = scenarios_forecast[0].loc[cutDate:(cutDate + dt.timedelta(days=window))].copy()
        upperBound = scenarios_forecast[2].loc[cutDate:(cutDate + dt.timedelta(days=window))].copy()

        # plot true data
        actual.plot(color=yellow, marker='o', ax=ax, label='True data')

        # plot forecast
        forecast_outOfSample.plot(color=grey, marker='o', ax=ax, label='Out-of-sample forecast')
        forecast_inSample.plot(color=grey, ax=ax, label='In-sample forecast')

        # plot forecast scenarios (margins
        ax.fill_between(forecast_outOfSample.index, lowerBound, upperBound,
                          facecolor=faded_grey)
        ax.legend()
        df = pd.concat([actual, forecast_outOfSample, forecast_inSample, lowerBound, upperBound], axis=1)
        df.columns = ['Actual', 'Forecast_outOfSample', 'Forecast_inSample', 'lowerBound', 'upperBound']
        df.to_excel(".\Exports\{country}_forecast_{days}days_diff_{diff}.xlsx".format(country=self.country,
                                                                            days=window, diff=diff,))

    def outOfSample_forecast_scenarios(self, cutDate=None, days=[7, 14], scenarios=[.005, .01, .015], verbose=False, figsize=(15,10)):

        method = 'Standard Scenarios'
        if scenarios == 'estimate':
            method = 'Estimated Scenarios'
            new_args = self.all_attributes.copy()
            new_args['cut_sample_date'] = cutDate

            estimator = self.create_new_object(self.model_type, new_args)
            estimator.train(verbose=verbose)
            S0p_estimate = estimator.params.copy()['S0p']
            if S0p_estimate >= 0.006:
                scenarios = [S0p_estimate-0.005, S0p_estimate, S0p_estimate + 0.005]
            else:
                scenarios = [S0p_estimate * .8, S0p_estimate, S0p_estimate * 1.2]

            print(scenarios)

        n_subplots = len(days)

        fig, axes = plt.subplots(nrows=n_subplots, ncols=2, figsize=figsize)

        for i in range(n_subplots):
            window = days[i]
            axes[i, 0].set_title('Fatalities forecast - {window} days ahead'.format(window=window))
            axes[i, 1].set_title('Daily fatalities forecast - {window} days ahead'.format(window=window))
            self.plot_forecast(ax=axes[i, 0], diff=False, window=window, scenarios=scenarios, cutDate=cutDate, verbose=verbose)
            self.plot_forecast(ax=axes[i, 1], diff=True, window=window, scenarios=scenarios, cutDate=cutDate, verbose=verbose)

        plt.tight_layout()
        fig.suptitle('{model} - {country} - Out-of-sample forecasts\nMethod: {method}'.format(model=self.model_type,
                                                                            country=self.country, method=method),
                     fontsize=16, y=1.05)

        plt.savefig(".\Exports\{country}_outOfSample_forecast.png".format(country=self.country), bbox_inches='tight')

        return True

    def outOfSample_forecast_S0(self, cutDate=None, plot=True, days=14, k=1):

        if not cutDate:
            cutDate = self.F_actual.index[-1] + dt.timedelta(days=-days)

        new_args = self.all_attributes.copy()
        new_args['cut_sample_date'] = cutDate

        estimator = self.create_new_object(self.model_type, new_args)
        estimator.train_S0(days=days, S0_initial_guess=self.S0_initial_guess,)

        self.forecast = estimator.df.copy()

        self.forecast = self.forecast.reindex(self.F_actual.index)

        # Calculate MSE
        self.mse = self.calculate_rmse(self.F_actual, self.forecast.F, cutDate, verbose=True)

        if plot:
            self.outOfSample_plot(cutDate, days=days, k=k)

        return self.mse

    def create_new_object(self, name='SIR', data= None):
        """
        Auxiliary method to create new model instances from within the code. Because of inheritance,
        we need to be able to know which model we are instantiating
        :param name: `string` the same as a class name
        :param data: `**kwargs`
        :return: `SIR`-like object
        """

        if name == 'SIR':
            return SIR(**data)

        if name == 'SIRFH':
            return SIRFH(**data)

        if name == 'SIRFH_Sigmoid':
            return SIRFH_Sigmoid(**data)

    def train_S0(self, options=None, days=7, S0_initial_guess=.01):

        self.S0_initial_guess = S0_initial_guess

        # Step #1 - Train with initial S0 guess (usually around 5%) - cut sample?
            # Problem: If initial guess is too bad, parameters may not well behave
            # Potential solution: joint RMSE optimization with more restrict bounds
        # Step #2 - Lock parameters and minimize out of sample RMSE with respect to S0
        # Step #3 - Train full model to make final projections

        # Step #1 - Train with initial S0 guess (usually around 5%)
        new_args = self.all_attributes.copy()
        new_args['force_parameters']['S0p'] = S0_initial_guess

        other_params_estimator = self.create_new_object(self.model_type, new_args)

        # Step #2 - Lock parameters and minimize out of sample RMSE with respect to S0
        other_params_estimator.train(verbose=False)
        new_params = other_params_estimator.params
        del new_params['S0p']

        new_args = self.all_attributes.copy()
        new_args['force_parameters'] = new_params

        s0_estimator = self.create_new_object(self.model_type, new_args)
        s0_estimator.outOfSample_days = days
        s0_estimator.train(loss_func=s0_estimator.loss_outOfSample, verbose=False, options=options)

        # Step #3 - estimate final model with final S0 estimate

        new_args = self.all_attributes.copy()
        new_args['force_parameters']['S0p'] = s0_estimator.params['S0p']

        final_estimator = self.create_new_object(self.model_type, new_args)

        final_estimator.train(options=options)

        self.df = final_estimator.df
        self.params = final_estimator.params
        self.model_params = final_estimator.params

    def train_S0_joint(self, options=None, days=7, ):

        # Step #1 - Train with initial S0 guess (usually around 5%) - cut sample?
            # Problem: If initial guess is too bad, parameters may not well behave
            # Potential solution: joint RMSE optimization with more restrict bounds
        # Step #2 - Lock parameters and minimize out of sample RMSE with respect to S0
        # Step #3 - Train full model to make final projections

        self.outOfSample_days = days
        self.train(loss_func=self.loss_outOfSample)

        # # Step #1 - Train with initial S0 guess (usually around 5%)
        # new_args = self.all_attributes
        # new_args['force_parameters']['S0p'] = S0_initial_guess
        #
        # other_params_estimator = self.create_new_object(self.model_type, new_args)
        #
        # # Step #2 - Lock parameters and minimize out of sample RMSE with respect to S0
        # other_params_estimator.train()
        # new_params = other_params_estimator.params
        # del new_params['S0p']
        #
        # new_args = self.all_attributes
        # new_args['force_parameters'] = new_params
        #
        # s0_estimator = self.create_new_object(self.model_type, new_args)
        # s0_estimator.outOfSample_days = days
        # s0_estimator.train(loss_func=s0_estimator.loss_outOfSample)

        # self.model_params = new_params


############## CONSTRAINT METHODS ################

    def const_lowerBoundR0(self, point):
        "constraint has to be R0 > bounds(0) value, thus (R0 - bound) > 0"
        # self.const_lowerBoundR0_S0opt.__code__.co_varnames
        # print(**kwargs)
        # print(locals())
        params = self.wrap_parameters(point)

        lowerBound = self.constraints_bounds['R0'][0]

        gamma = self.calculate_gamma()
        return (params['beta']/gamma) - lowerBound

    def const_upperBoundR0(self, point):
        # print(locals())
        # print(**kwargs)
        # self.const_upperBoundR0_S0opt.__code__.co_varnames

        params = self.wrap_parameters(point)

        upperBound = self.constraints_bounds['R0'][1]

        gamma = self.calculate_gamma()
        return upperBound - (params['beta']/gamma)


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

    def rollingPlot(self, export=False, parameters_list=None):

        if parameters_list:
            rolling_parameters = self.rolling_parameters[parameters_list]
        else:
            rolling_parameters = self.rolling_parameters

        axes = rolling_parameters.plot()
        fig = axes.get_figure()

        axes.axvline(x=self.quarantine_date, color='red', linestyle='--', label='Quarentine')

        if export:
            rolling_parameters.to_excel('export_rolling.xlsx')

    def outOfSample_plot(self, cutDate=None, days=14, diff=False, k=1):

        if not cutDate:
            cutDate = self.F_actual.index[-1] + dt.timedelta(days=-days)

        actual = self.F_actual.loc[:].copy()
        forecast_outOfSample = self.forecast.loc[cutDate:(cutDate + dt.timedelta(days=days))].F.copy()
        forecast_inSample = self.forecast.loc[:cutDate].F.copy()

        std = actual.diff().std()

        if diff:
            actual = actual.diff()
            forecast_outOfSample = forecast_outOfSample.diff()
            forecast_inSample = forecast_inSample.diff()

        # plot true data
        axes = actual.plot(color=yellow, marker='o')
        fig = axes.get_figure()

        # plot forecast
        forecast_outOfSample.plot(color=grey, marker='o')
        forecast_inSample.plot(color=grey)
        # pd.DataFrame([forecast_outOfSample, forecast_inSample])

        # plot forecast scenarios (margins
        axes.fill_between(forecast_outOfSample.index, forecast_outOfSample - k * std, forecast_outOfSample + k * std, facecolor=faded_grey)
        # return forecast_outOfSample

    def print_parameters(self):

        for param in self.params.keys():
            print("{param}: {value}".format(param=param, value=self.params[param]))

        print("S0: {value}".format(value=self.calculateS0(self.params['S0p'])))
        print('R0:{R0}'.format(R0=self.R0))

class SIRFH(SIR):
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
                 alpha=(0.025, 0.005, .97),
                 hospitalization_rate=.05,
                 **kwargs):

        self.rho = hospitalization_rate

        initial_guesses = {
            'delta': .05,
        }

        if hasattr(self, 'initial_guesses'):
            self.initial_guesses = {**initial_guesses, **self.initial_guesses}
        else:
            self.initial_guesses = {
                'delta': .05,
            }

        new_args = {**kwargs, **{'alpha': alpha, 'hospitalization_rate': hospitalization_rate}}
        super().__init__(**new_args)
        self.model_type = 'SIRFH'

    def set_default_bounds(self):
        """
        Sets the default values for unprovided bounds
        :return:
        """

        super().set_default_bounds()

        if 'gamma_i' not in self.parameter_bounds.keys():
            self.parameter_bounds['gamma_i'] = (1/(3*7), 1/(1*7))

        if 'gamma_h' not in self.parameter_bounds.keys():
            self.parameter_bounds['gamma_h'] = (1/(7*7), 1/(2*7))

        if 'omega' not in self.parameter_bounds.keys():
            self.parameter_bounds['omega'] = (1/(20), 1/(5))

        if 'delta' not in self.parameter_bounds.keys():
            self.parameter_bounds['delta'] = (0, 79/165)

        del self.parameter_bounds['gamma']

    def model(self, t, y):
        S = y[0]
        I_n = y[1]
        H_r = y[2]
        H_f = y[3]
        R = y[4]
        F = y[5]

        I = I_n + H_r + H_f

        S0_model = self.calculateS0(self.model_params['S0p'])

        beta = self.beta(t)

        ret = [
            # S - Susceptible
            -beta * I * S / S0_model,

            # I_n
            (1 - self.rho) * beta * I * S / S0_model  # (1-rho) BIS/N
            - self.model_params['gamma_i'] * I_n,  # Gamma_I x I_n

            # H_r
            self.rho * (1 - self.model_params['delta']) * beta * I * S / S0_model  # rho * (1-delta) BIS/N
            - self.model_params['gamma_h'] * H_r,

            # H_f
            self.rho * self.model_params['delta'] * beta * I * S / S0_model  # rho * (delta) BIS/N
            - self.model_params['omega'] * H_f,

            # R
            self.model_params['gamma_i'] * I_n  # gamma_I * In
            + self.model_params['gamma_h'] * H_r,  # gamma_H * Hr

            # F
            self.model_params['omega'] * H_f,
        ]

        return ret

    def cut_sample(self):

        cutDate = self.cut_sample_date
        if cutDate:
            if not isinstance(cutDate, dt.datetime):
                cutDate = self.I_actual.index[-1] + dt.timedelta(days=-cutDate)

            self.I_actual = self.I_actual.loc[:cutDate].copy()
            self.R_actual = self.R_actual.loc[:cutDate].copy()
            self.F_actual = self.F_actual.loc[:cutDate].copy()

    def load_data(self):
        """
        New function to use our prop data
        """
        super().load_data()

        #True data series
        self.R_actual = self.recovered
        self.F_actual = self.fatal
        self.I_actual = self.confirmed - self.R_actual - self.F_actual  # obs this is total I

        self.cut_sample()

    def beta(self, t):
        return self.model_params['beta']

    def calculate_gamma(self):
        """
        Using the ´self.params´ dictionary, calculates gamma with its adaptation to the SIRFH model.

        :return: gamma
        """

        if hasattr(self, 'model_params'):
            gamma = (1 - self.rho) * self.model_params['gamma_i'] + self.rho * ((1 - self.model_params['delta']) * self.model_params['gamma_h']
                                                                          + self.model_params['delta'] * self.model_params['omega'])
        else:
            gamma = .07

        return gamma

    def initialize_parameters(self):
        self.R_0 = self.recovered[0]
        self.F_0 = self.fatal[0]
        self.I_0 = self.confirmed.iloc[0] - self.R_0 - self.F_0
        self.I_n_0 = self.I_0 * (1 - self.rho)
        self.H_r_0 = self.rho * (1 - 79/165) * self.I_0 #TODO Might be a strong assumption, check sensitivity
        self.H_f_0 = self.rho * (79/165) * self.I_0
        self.H_0 = self.H_r_0 + self.H_f_0

    def calculateS0(self, S0p):

        return self.country_population * S0p - self.I_0 - self.H_0 - self.R_0 - self.F_0

    def loss(self, point):
        """
        RMSE between actual confirmed cases and the estimated infectious people with given beta and gamma.
        """
        size = self.I_actual.shape[0]
        self.model_params = self.wrap_parameters(point)

        S0 = self.calculateS0(self.model_params['S0p'])

        # solution = solve_ivp(SIR, [0, size], [S_0, self.I_0, self.R_0], t_eval=np.arange(0, size, 1), vectorized=True)
        solution = solve_ivp(self.model, [0, size], [S0, self.I_n_0, self.H_r_0, self.H_f_0, self.R_0, self.F_0],
                             t_eval=np.arange(0, size, 1), vectorized=True)

        y = solution.y
        S = y[0]
        I_n = y[1]
        H_r = y[2]
        H_f = y[3]
        R = y[4]
        F = y[5]

        I = I_n + H_r + H_f

        alphas = self.alpha

        # l1 = ((I - self.I_actual) / self.I_actual) ** 2
        l1 = (np.diff(I, prepend=np.nan) - self.I_actual.diff()) ** 2
        l1.replace([np.inf, -np.inf, np.nan], 0, inplace=True)
        l1 = np.sqrt(np.mean(l1))

        # l2 = ((R - self.R_actual) / self.R_actual) ** 2
        l2 = (np.diff(R, prepend=np.nan) - self.R_actual.diff()) ** 2
        l2.replace([np.inf, -np.inf, np.nan], 0, inplace=True)
        l2 = np.sqrt(np.mean(l2))

        # l3 = ((F - self.F_actual) / self.F_actual) ** 2
        l3 = (np.diff(F, prepend=np.nan) - self.F_actual.diff()) ** 2
        l3.replace([np.inf, -np.inf, np.nan], 0, inplace=True)
        l3 = np.sqrt(np.mean(l3))

        loss = alphas[0] * l1 + alphas[1] * l2 + alphas[2] * l3
        # loss = l3

        return loss

    def loss_level(self, point):
        """
        RMSE between actual confirmed cases and the estimated infectious people with given beta and gamma.
        """
        size = self.I_actual.shape[0]
        self.model_params = self.wrap_parameters(point)

        S0 = self.calculateS0(self.model_params['S0p'])

        # solution = solve_ivp(SIR, [0, size], [S_0, self.I_0, self.R_0], t_eval=np.arange(0, size, 1), vectorized=True)
        solution = solve_ivp(self.model, [0, size], [S0, self.I_n_0, self.H_r_0, self.H_f_0, self.R_0, self.F_0],
                             t_eval=np.arange(0, size, 1), vectorized=True)

        y = solution.y
        S = y[0]
        I_n = y[1]
        H_r = y[2]
        H_f = y[3]
        R = y[4]
        F = y[5]

        I = I_n + H_r + H_f

        alphas = self.alpha

        l1 = ((I - self.I_actual) / self.I_actual) ** 2
        l1.replace([np.inf, -np.inf, np.nan], 0, inplace=True)
        l1 = np.sqrt(np.mean(l1))

        l2 = ((R - self.R_actual) / self.R_actual) ** 2
        l2.replace([np.inf, -np.inf, np.nan], 0, inplace=True)
        l2 = np.sqrt(np.mean(l2))

        l3 = ((F - self.F_actual) / self.F_actual) ** 2
        l3.replace([np.inf, -np.inf, np.nan], 0, inplace=True)
        l3 = np.sqrt(np.mean(l3))

        loss = alphas[0] * l1 + alphas[1] * l2 + alphas[2] * l3

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

        self.quarantine_loc = float(self.confirmed.index.get_loc(self.quarantine_date))

        S0 = self.calculateS0(self.params['S0p'])

        prediction = solve_ivp(self.model, [0, size], [S0, self.I_n_0, self.H_r_0, self.H_f_0, self.R_0, self.F_0],
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

    def rolling_peak(self, figsize=(15, 8)):
        """
        This function estimates the fatalities peak with an ever-increasing data window.
        :return:
        """

        params_list = []

        # for cutDate in self.confirmed.index:
        for cutDate in self.I_actual.index:

            new_args = self.all_attributes.copy()
            new_args['cut_sample_date'] = cutDate

            estimator = self.create_new_object(self.model_type, new_args)
            estimator.train(verbose=False)
            # optimal_parameters = estimator.params.copy()

            current_peak = estimator.df.index[estimator.df.diff().F_Actual == estimator.df.F_Actual.diff().max()]
            if current_peak.shape[0] > 0:
                current_peak = current_peak[0]
            else:
                current_peak = np.nan

            estimated_peak = estimator.df.index[estimator.df.diff().F == estimator.df.diff().F.max()]
            if estimated_peak.shape[0] > 0:
                estimated_peak = estimated_peak[0]
            else:
                estimated_peak = np.nan

            params_list.append({
                'Current peak': current_peak,
                'Estimated peak': estimated_peak,
            })

        self.rolling_peak_df = pd.DataFrame(params_list, index=self.I_actual.index)
        self.rolling_peak_df.fillna(method='bfill', inplace=True)
        self.rolling_peak_df.fillna(method='ffill', inplace=True)

        self.rolling_peak_df = self.rolling_peak_df.iloc[1:]

        fig, ax = plt.subplots(figsize=figsize)

        # Cut data on the peak date
        peak_date = pd.Series([self.rolling_peak_df['Current peak'].iloc[-1],
                               self.rolling_peak_df['Estimated peak'].iloc[-1]])

        peak_date = peak_date.max()

        self.rolling_peak_df = self.rolling_peak_df.loc[:peak_date]

        self.rolling_peak_df['Estimated peak'].plot(axes=ax, color=yellow, marker='o', label='Estimated')
        self.rolling_peak_df['Current peak'].plot(axes=ax, color=grey, marker='o', label='Current')

        self.rolling_peak_df['Peak max'] = self.rolling_peak_df['Current peak'].max()

        ax.axhline(y=self.rolling_peak_df['Current peak'].max(), color='black', linestyle='--', label='True peak')
        ax.legend()

        self.rolling_peak_df['UB'] = self.rolling_peak_df['Current peak'].max() + dt.timedelta(days=5)
        self.rolling_peak_df['LB'] = self.rolling_peak_df['Current peak'].max() + dt.timedelta(days=-5)

        ax.fill_between(self.rolling_peak_df['Current peak'].index,
                          self.rolling_peak_df['LB'],
                          self.rolling_peak_df['UB'],
                          facecolor=faded_grey)

        plt.tight_layout()
        fig.suptitle('Fatality peak forecast - {country}'.format(model=self.model_type, country=self.country,),
                     fontsize=16, y=1.05)

        plt.savefig(".\Exports\{country}_rolling_peak.png".format(country=self.country), bbox_inches='tight')

        export_df = self.rolling_peak_df[['Current peak', 'Estimated peak', 'LB', 'UB', 'Peak max']].copy()
        export_df.to_excel(".\Exports\{country}_rolling_peak_dates.xlsx".format(
            country=self.country))

        export_df = export_df - self.F_actual.index[0]
        export_df = export_df / np.timedelta64(1, 'D')

        export_df.index = (export_df.index - self.F_actual.index[0]) / np.timedelta64(1, 'D')

        export_df.to_excel(".\Exports\{country}_rolling_peak.xlsx".format(
            country=self.country))

        return self.rolling_peak_df

    def rolling_n_fatal(self, nfatal=[50, 100], figsize=(15, 8)):
        """
        This function creates the rolling estimate of the date in which the model indicates less than `n` daily fatalities
        :return:
        """

        params_list = []

        for cutDate in self.I_actual.index:
        # for cutDate in self.confirmed.index:

            new_args = self.all_attributes.copy()
            new_args['cut_sample_date'] = cutDate

            estimator = self.create_new_object(self.model_type, new_args)
            estimator.train(verbose=False)

            # get the date with less than n daily fatalities after the peak
            # find peak
            peak = estimator.df.index[estimator.df.diff().F == estimator.df.F.diff().max()]

            dic = {}

            if peak.shape[0] > 0:
                peak_i = peak[0]
                analysis_period = estimator.df.diff().F.loc[peak_i:]
                current_estimate = analysis_period.index[analysis_period <= nfatal[0]]
                current_estimate = current_estimate[0]
            else:
                current_estimate = np.nan

            dic["n={n}".format(n=nfatal[0])] = current_estimate

            if peak.shape[0] > 0:
                peak_i = peak[0]
                analysis_period = estimator.df.diff().F.loc[peak_i:]
                current_estimate = analysis_period.index[analysis_period <= nfatal[1]]
                current_estimate = current_estimate[0]
            else:
                current_estimate = np.nan

            dic["n={n}".format(n=nfatal[1])] = current_estimate


            params_list.append(dic)

        self.rolling_peak_df = pd.DataFrame(params_list, index=self.I_actual.index)
        self.rolling_peak_df.fillna(method='bfill', inplace=True)
        self.rolling_peak_df.fillna(method='ffill', inplace=True)

        self.rolling_peak_df = self.rolling_peak_df.iloc[1:]

        fig, ax = plt.subplots(figsize=figsize)

        self.rolling_peak_df["n={n}".format(n=nfatal[0])].plot(axes=ax, color=yellow, marker='o', label="n={n}".format(n=nfatal[0]))
        self.rolling_peak_df["n={n}".format(n=nfatal[1])].plot(axes=ax, color=grey, marker='o', label="n={n}".format(n=nfatal[1]))

        # ax.axhline(y=self.rolling_peak_df['Current peak'].max(), color='black', linestyle='--', label='True peak')
        ax.legend()

        plt.tight_layout()
        fig.suptitle('Daily Deaths Forecast - {country}'.format(model=self.model_type, country=self.country,),
                     fontsize=16, y=1.05)

        plt.savefig(".\Exports\{country}_rolling_n_fatal.png".format(country=self.country), bbox_inches='tight')

        return self.rolling_peak_df

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
        axes.axvline(x=self.quarantine_date, color='red', linestyle='--', label='Quarentine')
        axes.axhline(y=50000, color='black', linestyle='--', label='Hospital Capacity')

    def print_parameters(self):

        super().print_parameters()

        print("gamma_i: {value} days".format(value= 1 / self.params['gamma_i']))
        print("gamma_h: {value} days".format(value=1 / self.params['gamma_h']))
        print("omega: {value} days".format(value=1 / self.params['omega']))

    def plot_main_forecasts(self,  figsize=(15, 5),):

        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=figsize)

        axes[0].set_title('Daily Fatalities fit')
        axes[1].set_title('Total Fatalities fit')

        ax = axes[1]
        self.df['F_Actual'].plot(ax=ax, color=grey, label='Fatalities')
        self.df['F'].plot(ax=ax, color=yellow, label='Forecast fatalities')

        self.df[['F_Actual', 'F']].to_excel(".\Exports\{country}_HF_level.xlsx".format(country=self.country))

        ax.legend()

        ax = axes[0]
        self.df['F'].diff().plot(ax=ax, color=grey, label='Forecast fatalities')
        self.df['F_Actual'].diff().plot(ax=ax, color=yellow, label='Fatalities')
        df = self.df.copy()
        df['F'] = df['F'].diff()
        df['F_Actual'] = df['F_Actual'].diff()
        df[['F', 'F_Actual']].to_excel(".\Exports\{country}_HF_diff.xlsx".format(country=self.country))

        ax.legend()

        plt.tight_layout()
        fig.suptitle('{model} - {country} - Forecasts'.format(model=self.model_type, country=self.country),
                     fontsize=16, y=1.05)

        plt.savefig(".\Exports\{country}_main_forecast.png".format(country=self.country), bbox_inches='tight')

    def plot_main_forecasts_hospital(self,  figsize=(15, 5), hospital_line=False):

        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=figsize)

        axes[0].set_title('Fatalities fit')
        axes[1].set_title('Hospital Demand')

        ax = axes[1]
        self.df['H'].plot(ax=ax, color=red, label='Hospital Demand')
        self.df['F'].plot(ax=ax, color=grey, label='Fatalities')
        self.df['F_Actual'].plot(ax=ax, color=yellow, label='True Fatalities')

        self.df[['H', 'F', 'F_Actual']].to_excel(".\Exports\{country}_HF_level.xlsx".format(country=self.country))

        if hospital_line:
            ax.axhline(y=61800, color='black', linestyle='--', label='Hospital Capacity')
        ax.legend()

        ax = axes[0]
        self.df['F'].diff().plot(ax=ax, color=grey, label='Forecast fatalities')
        self.df['F_Actual'].diff().plot(ax=ax, color=yellow, label='Fatalities')
        df = self.df.copy()
        df['F'] = df['F'].diff()
        df['F_Actual'] = df['F_Actual'].diff()
        df[['F', 'F_Actual']].to_excel(".\Exports\{country}_HF_diff.xlsx".format(country=self.country))

        ax.legend()

        plt.tight_layout()
        fig.suptitle('{model} - {country} - Forecasts'.format(model=self.model_type, country=self.country),
                     fontsize=16, y=1.05)

        plt.savefig(".\Exports\{country}_main_forecast.png".format(country=self.country), bbox_inches='tight')

class SIRFH_Sigmoid(SIRFH):
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
                 **kwargs):

        self.initial_guesses = {
            'lambda': 1.0,
        }

        super().__init__(**kwargs)

        self.model_type = 'SIRFH_Sigmoid'
        self.sig_normal_t = self.quarantine_loc + 7

    def set_default_bounds(self):
        """
        Sets the default values for unprovided bounds. All model parameters should be on this function.
        Pay attention to the fact that it is inheriting bounds from its parent classes
        :return:
        """

        super().set_default_bounds()

        if 'lambda' not in self.parameter_bounds.keys():
            self.parameter_bounds['lambda'] = (1/4, 4)

        if 'beta1' not in self.parameter_bounds.keys():
            self.parameter_bounds['beta1'] = (.05, .5)

        if 'beta2' not in self.parameter_bounds.keys():
            self.parameter_bounds['beta2'] = (.05, .5)

        del self.parameter_bounds['beta']

    def add_constraints(self):
        self.constraints.append({'type': 'ineq', 'fun': self.const_betas},)

    def beta(self, t):
        # Normalize t
        t -= self.sig_normal_t

        return (self.model_params['beta1'] -
                self.model_params['beta2']) / (1 + np.exp(t / self.model_params['lambda'])) + self.model_params['beta2']

    def calculate_r0(self):
        """
        Using the ´self.params´ dictionary, calculates R0

        :return: R0
        """

        gamma = self.calculate_gamma()



        return {"R0_initial": self.params['beta1'] / gamma, "R0_final": self.params['beta2'] / gamma, }

    def outOfSample_loss(self, S0p):

        self.S0pbounds = (S0p, S0p)

        return self.outOfSample_forecast()

    def outOfSample_train(self, cutDate=None, days=7):

        if not cutDate:
            cutDate = self.F_actual.index[-1] + dt.timedelta(days=-days)

        # Lock and backup S0
        S0_initital_guess = 0.05
        bkp_S0_bounds = self.S0pbounds
        self.S0pbounds = (S0_initital_guess, S0_initital_guess)

        self.estimate(verbose=False)
        #release lock
        self.S0pbounds = bkp_S0_bounds

        # Create new object with the locked parameters
        newObj = SIRFH_Sigmoid(
            country=self.country,
            N=self.country_population,
            nth=self.nth,
            quarantineDate=self.quarantine_date,
            hospRate=self.hospitalization_rate,
            alphas=self.alpha,
            adjust_recovered=self.adjust_recovered,

            beta1_bounds=(self.beta1, self.beta1),
            beta2_bounds=(self.beta2, self.beta2),
            delta_bounds=(self.delta, self.delta),
            gamma_i_bounds=(self.gamma_I, self.gamma_I),
            gamma_h_bounds=(self.gamma_H, self.gamma_H),
            omega_bounds=(self.omega, self.omega),
            lambda_bounds=(self.lambda_bounds, self.lambda_bounds),

            S0pbounds=self.S0pbounds,

        )

        # minimize out of sample forecast RMSE

        optimal = minimize(
            self.outOfSample_loss,
            np.array([
                0.05
            ]),
            args=(),
            method='SLSQP',
            bounds=[
                self.S0pbounds
            ],
        )
        self.optimizer = optimal
        S_0p, = optimal.x

        S_0 = self.country_population * S_0p - self.I_0 - self.H_0 - self.R_0 - self.F_0

        print("S0p new estimate")
        print("S0: {S0}".format(S0=self.S_0))

        # Release locks
        self.delta_bounds = bkp_delta_bounds
        self.beta1_bounds = bkp_betaBounds
        self.beta2_bounds = bkp_gamma_i_bounds
        self.gamma_i_bounds = bkp_gamma_i_bounds
        self.gamma_h_bounds = bkp_gamma_h_bounds
        self.omega_bounds = bkp_omega_bounds
        self.lambda_bounds = bkp_lambda_bounds

############## CONSTRAINT METHODS ################

    def const_betas(self, point):
        """
        Initial `beta` should be higher than final `beta`
        :param point:
        :return:
        """
        params = self.wrap_parameters(point)
        return params['beta1'] - params['beta2']

    def const_lowerBoundR0(self, point):
        "constraint has to be R0 > bounds(0) value, thus (R0 - bound) > 0"

        params = self.wrap_parameters(point)

        lowerBound = self.constraints_bounds['R0'][0]

        gamma = self.calculate_gamma()

        R0_1 = params['beta1'] / gamma
        R0_2 = params['beta2'] / gamma

        return min(R0_1, R0_2) - lowerBound

    def const_upperBoundR0(self, point):

        params = self.wrap_parameters(point)

        upperBound = self.constraints_bounds['R0'][1]

        gamma = self.calculate_gamma()

        R0_1 = params['beta1'] / gamma
        R0_2 = params['beta2'] / gamma

        return upperBound - max(R0_1, R0_2)


if __name__ == '__main__':
    hospRate = 0.05
    deltaUpperBound = 79 / 165
    cut_sample_date = dt.datetime(2020, 5, 14)

    t1 = SIRFH(country='Korea, South',
               # quarantineDate = dt.datetime(2020,3,24), #italy lockdown was on the 9th
               hospitalization_rate=hospRate,
               alpha=[.00, 0.00, .998],

               # Loose restrictions
               # S0pbounds=(10e6 / 200e6, 10e6 / 200e6),
               # delta_bounds=(0, deltaUpperBound),
               # betaBounds=(0.20, 1.5),
               # gammaBounds=(0.01, .2),
               # gamma_i_bounds=(1/(20), 1/(1)),
               # gamma_h_bounds=(1/(8*7), 1/(2*7)),
               # omega_bounds=(1/(4*7), 1/(3)),

               # Tight restrictions
               # S0pbounds=(10e6 / N, 10e6 / N),
               force_parameters={
                   # 'S0p': .05,
                   # 'delta': 79/165,
                   # 'beta1': 0.31118164052008357,
                   # 'beta2': .2,
                   # 'gamma_i': 0.19999999999999982,
                   # 'gamma_h': 0.023809523809525043,
                   # 'omega': 0.14199161301361687,
                   # 'lambda': 0.5,

               },

               parameter_bounds={
                   'S0p': (.0001, .02),
                   #    'delta': (0, deltaUpperBound),
                   #    'beta1': (0.20, 1.5),
                   #    'beta2': (0.20, 1.5),
                   'gamma_i': (1 / (14), 1 / (4)),
                   'gamma_h': (1 / (6.5 * 7), 1 / (2.5 * 7)),
                   'omega': (1 / (21), 1 / (5)),
                   #    'lambda': (.5,2)

               },

               constraints_bounds={
                   'R0': (1, 6),
               },

               cut_sample_date = cut_sample_date,

               )

    # t1.train_S0()
    t1.train()
    roll = t1.rolling_peak()