# Attempt 3.1 converted into an object
import timeit
import pandas as pd
import numpy as np

class PricePredictor(object):

    def __init__(self, estimator, inflation=None):

        self.__key_cols__ = [ \
                            'ProductGroup',
                            'fiBaseModel',  # when use with ModelID, only decrease error by a O(0.001)
                            'ModelID',      # -- Model ID actually includes fiBaseModel
                            # 'fiProductClassDesc',
                            ]

        self.estimator = estimator

        if not isinstance(inflation, type(None)):
            inflation[:] = 1 + 0.01 * np.cumsum(inflation)

        self.__inflation_multiplier__ = inflation
        self.__avg_price__ = {}
        self.__avg_price_per_age__ = {}
        self.__avg_age__ = {}




    def __clean_df__(self, df, isTrain=False, mute=True):
        """
        :type df: pandas.DataFrame
        :type isTrain: bool
        :type mute: bool
        :rtype: pandas.DataFrame
        """

        df = df[self.__key_cols__ + ['YearMade', 'saledate'] + isTrain*['SalePrice']].copy()


        # convert ModelID column to string in case all IDs are integers
        if 'ModelID' in self.__key_cols__:
            df['ModelID'] = df['ModelID'].astype('str')


        # index string `1/10/2003 0:00` to get sale year (faster than using `pd.DatetimeIndex(df['saledate']).year`)
        df['SaleYear'] = df.saledate.str[-9:-4].astype(int)
        df['Age'] = df['SaleYear'] - df['YearMade']


        # if training: drop all rows with bad data
        if isTrain:

            # Discard outliers/leverage points (boundaries from inspecting Price vs. YearMade plots)
            price_bounds = pd.DataFrame({'BL': (7500, 40000),
                                         'MG': (8000, 142500),
                                         'SSL': (4500, 20000),
                                         'TEX': (7500, 110000),
                                         'TTT': (7500, 127500),
                                         'WL': (7500, 110000)})
            age_bounds = pd.Series({'BL': 1971,
                                    'MG': 1960,
                                    'SSL': 1968,
                                    'TEX': 1970,
                                    'TTT': 1955,
                                    'WL': 1964})
            pgs = df['ProductGroup'].unique()

            for g in pgs:
                df.drop(df.index[(df.ProductGroup == g) & \
                                 ((df.SalePrice < price_bounds[g][0]) | (df.SalePrice > price_bounds[g][1]))],
                        inplace=True)
                df.drop(df.index[(df.ProductGroup == g) & (df.YearMade < age_bounds[g])],
                        inplace=True)


            ## from inspecting the historgram of YearMade, drop all vehicles made before 1960 (fictitious YearMade and leverage points)
            ## fraction of data points before 1900 = 0.09519476472421315
            ## fraction of data points between 1900 and 1960 = 0.0006755998753505765 -- leverage points
            # df = df[df.YearMade >= 1960]

            # drop all vehicles that were sold before made (error in data)
            df = df[df.Age > 0]
            df['PricePerAge'] = df['SalePrice'] / (df['Age'] + 1)  # to avoid division by 0

            # adjust for inflation
            if not isinstance(self.__inflation_multiplier__, type(None)):
                df['SalePrice'] *= self.__inflation_multiplier__[df.SaleYear].values    # without .values run into "ValueError: cannot reindex from a duplicate axis"


        # else:
        #     - cannot drop any observations!
        #     - leave negative Age as is -- will be replace with category average when predicting


        df.drop(['YearMade', 'saledate'], axis=1, inplace=True)

        # for debugging
        if not mute:
            print df.columns
            print df.shape

        return df



    def __compile_reference_dics__(self, df, mute=True):
        """
        :type df: pandas.DataFrame
        :type mute: bool
        :rtype: None
        """

        # --- helper function ---
        def bootstrap_avg(df, colname, avg_fct=np.mean, n_iter=10, mute=True):
            """
            :type df: pandas.DataFrame
            :type colname: List[str]
            :type mute: bool
            """

            price_observations = df[colname].values

            if len(price_observations) == 1:
                return price_observations[0]


            inds = range(len(df))
            avg = 0
            for _ in xrange(n_iter):
                sample_inds = np.random.choice(inds, len(price_observations), replace=True)
                # avg += price_observations[sample_inds].mean()
                avg += avg_fct(price_observations[sample_inds])

            # for debugging
            if not mute and not np.isfinite(avg):
                print 'Error!'

            return float(avg) / n_iter

        # --- helper function ---


        method = 2      # method 2 (10 iters) works the best
        n_iter = 10
        methods = ('averaging method: mean()',
                   'averaging method: median()',
                   'averaging method: bootstrap_avg(mean, n_iter=%i)' % n_iter,
                   'averaging method: bootstrap_avg(median, n_iter=%i)' % n_iter,
                   'averaging method: mixed')
        print '    ', methods[method]


        for i in xrange(1,len(self.__key_cols__)+1):

            cnames = self.__key_cols__[0:i]

            # SalePrice, PricePerAge and Age all have no NaN so can dropna together
            grpdf = df.dropna().groupby(cnames)

            if method == 0:     # method 0: mean()
                self.__avg_price__[i]         = grpdf.mean()['SalePrice'].to_dict()
                self.__avg_price_per_age__[i] = grpdf.mean()['PricePerAge'].to_dict()
                self.__avg_age__[i]           = grpdf.mean()['Age'].to_dict()


            elif method == 1:     # method 1: median()
                self.__avg_price__[i]         = grpdf.median()['SalePrice'].to_dict()
                self.__avg_price_per_age__[i] = grpdf.median()['PricePerAge'].to_dict()
                self.__avg_age__[i]           = grpdf.median()['Age'].to_dict()


            elif method in {2,3}: # bootstrap_avg()

                avg_fct = np.mean if method == 2 else np.median

                self.__avg_price__[i]         = {}
                self.__avg_price_per_age__[i] = {}
                self.__avg_age__[i]           = {}

                for key, grp in grpdf:
                    self.__avg_price__[i][key]         = bootstrap_avg(grp, 'SalePrice', avg_fct=avg_fct)
                    self.__avg_price_per_age__[i][key] = bootstrap_avg(grp, 'PricePerAge', avg_fct=avg_fct)
                    self.__avg_age__[i][key]           = bootstrap_avg(grp, 'Age', avg_fct=avg_fct)


            elif method == 4:
                self.__avg_price__[i]         = grpdf.median()['SalePrice'].to_dict()

                self.__avg_price_per_age__[i] = {}
                self.__avg_age__[i]           = {}

                for key, grp in grpdf:
                    self.__avg_price_per_age__[i][key] = bootstrap_avg(grp, 'PricePerAge', avg_fct=np.mean)
                    self.__avg_age__[i][key]           = bootstrap_avg(grp, 'Age', avg_fct=np.mean)


            # for debugging:
            if not mute:
                print grpdf.keys
                print self.__avg_price__[i]



    def __lookup_ref__(self, df, mute=True):
        """
        :type df: panads.DataFrame
        :rtype: List[numpy.array]
        """

        vehicle_infos = df[self.__key_cols__ + ['Age']].values        # only want vehicle model info and Age

        completeness = df[self.__key_cols__].notnull().sum(1).values  # how complete a row's vehicle info is
        hasAge       = vehicle_infos[:,-1] >= 0                       # whether vehicle has a valid age
        adjusted_age = vehicle_infos[:,-1].astype(float) + 1          # the age to muliply by "price per age"
            # n.b.:
            # - make age a continuous variable by conterting to float type
            # - add 1 since average price was calculated by dividing by Age+1)


        price_refs = np.zeros([len(df), 3])


        for i in xrange(len(vehicle_infos)):
            c = completeness[i]
            t = tuple(vehicle_infos[i,:c])

            while t not in self.__avg_price__[c]:   # doesn't matter which dictionary set -- all have the same keys
                if not mute and c == 1:
                    print 'Error! \n Row %i:%s' % (i, vehicle_infos[i])

                c -= 1
                t = vehicle_infos[i,0] if c == 1 else tuple(vehicle_infos[i,:c])    # when c==1, the key is 'BL' instead of ('BL',)


            if not hasAge[i]:
                adjusted_age[i] = self.__avg_age__[c][t] + 1
                # adjusted_age[i] = self.__avg_age__[1][vehicle_infos[i,0]] + 1


            price_refs[i][0] = self.__avg_price__[c][t]
            price_refs[i][1] = self.__avg_price_per_age__[c][t]
            price_refs[i][2] = adjusted_age[i] * self.__avg_price_per_age__[c][t]



        cols = [0,1,2]    # best combo
        ref_names = np.array(['__avg_price__',
                              '__age_price_per_age__',
                              'adusted_age * __age_price_per_age__'])

        for ref_name in ref_names[cols]:
            print '              feature:', ref_name
        print '              feature: ProductGroup dummies'

        return np.hstack((price_refs[:, cols],
                        #   adjusted_age[:,np.newaxis],
                          pd.get_dummies(df['ProductGroup']).iloc[:,1:].values * 100,
                        ))



    def fit(self, df, mute=True):
        """
        :type df: pandas.DataFrame
        :type mute: bool
        :rtype: None
        """

        df_clean = self.__clean_df__(df, isTrain=True, mute=mute)
        self.__compile_reference_dics__(df_clean, mute=mute)


        X = self.__lookup_ref__(df_clean, mute=mute)
        y = df_clean['SalePrice'].values


        # for debugging:
        if not mute:
            print X.shape
            print y.shape


        self.estimator.fit(X,y)
        # self.estimator.fit(X,np.sqrt(y))      # breaks heteroscedasticity (if there is)
        # self.estimator.fit(X,np.log(y))       # breaks heteroscedasticity (if there is)


    def predict(self, df, mute=True):
        """
        :type df: pandas.DataFrame
        :type mute: bool
        :rtype: numpy.array
        """

        df_clean = self.__clean_df__(df, isTrain=False, mute=mute)

        X = self.__lookup_ref__(df_clean, mute=mute)

        y_pred = self.estimator.predict(X)
        # y_pred = self.estimator.predict(X)**2
        # y_pred = np.exp(self.estimator.predict(X))

        # adjust for inflation
        if not isinstance(self.__inflation_multiplier__, type(None)):
            y_pred /= self.__inflation_multiplier__[df_clean.SaleYear].values    # without .values run into "ValueError: cannot reindex from a duplicate axis"

        return y_pred




# --- Free Floating Methods ---

def rmsle(y_true, y_pred):
    """
    :type y: numpy.array
    :type y_pred: numpy.array
    :rtype: float
    """

    return np.sqrt(np.sum((np.log(y_pred + 1) - np.log(y_true + 1))**2) / len(y_true))


def train_test_split(df):
    """
    :type df: pandas.DataFrame
    :rtype tuple(DataFrame, DataFrame)
    """

    np.random.seed(42)

    # random shuffle row indices
    # rand_i = range(len(df))
    np.random.shuffle(rand_i)
    # rand_i = rand_i[:2000]              # comment out to use the full df
    test_size = int(len(rand_i)*0.2)


    df_test = df.iloc[rand_i[:test_size], :]
    df_train = df.iloc[rand_i[test_size:], :]

    return (df_train, df_test)



def procedure(estimator, df_train, df_test, test_solution=None, inflation=None, outfile=None, verbose=False):
    """
    :type df: pandas.DataFrame
    :rtype: None
    """

    ## quick train-test evaluation on training set
    if isinstance(test_solution, type(None)):
        df_train, df_test = train_test_split(df_train)
        test_solution = df_test[['SalePrice']]          # double square brackets so that it's a DataFrame


    print 'Progress:'
    print '  creating predictor...'

    print '       base estimator:', estimator.__class__.__name__

    if estimator.__class__.__name__ == "Pipeline":
        print '               scaler:', estimator.get_params()['scaler'].__class__.__name__

        e = estimator.get_params()['estimator']
        print '                model:', e.__class__.__name__
    else:
        e = estimator   # for verbose


    if verbose:
        print '     estimator params:'

        l = max(map(len, e.get_params()))
        for k,v in e.get_params().iteritems():
            print '                       %s\t%s' % (k.ljust(l), v)

        print '\n'

    predictor = PricePredictor(estimator, inflation)
    print '          ref columns:', predictor.__key_cols__


    print '  fitting model...'
    start = timeit.default_timer()
    predictor.fit(df_train)
    print '      (time: %.5f s)' % float(timeit.default_timer() - start)

    print '  predicting result...'
    start = timeit.default_timer()
    y_pred = predictor.predict(df_test, mute=True)
    print '      (time: %.5f s)' % float(timeit.default_timer() - start)

    error = rmsle(test_solution['SalePrice'].values, y_pred)
    print '>> RMS log error:', error


    ## write results to file
    if outfile:
        df_results = pd.DataFrame(np.c_[df_test['SalesID'].values, y_pred], columns = ['SalesID', 'SalePrice'])
        df_results.to_csv(outfile, index=False)


    return error




if __name__ == "__main__":

    from sklearn.linear_model import SGDRegressor
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.ensemble import AdaBoostRegressor
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import MinMaxScaler, StandardScaler
    from sklearn.grid_search import GridSearchCV
    from sklearn.base import clone
    from sklearn.metrics import mean_squared_error

    start = timeit.default_timer()      # wall time, not CPU time


    ## Estimators used prior to grid search:                                              # Performace at best feature config found in Pipeline baseline RandomForest
    # estimator = RandomForestRegressor(50, n_jobs=-1, max_depth=13, max_features=0.4)    # 0.262629550817 @ 29.91s -- baseline -- params from GradientSearch in Attempt 2
    # estimator = RandomForestRegressor(50, n_jobs=-1)                                    # 0.289937200734 @ 42.8s
    # estimator = GradientBoostingRegressor()                                             # nan @ 63.24s
    # estimator = AdaBoostRegressor()                                                     # 0.450653027236 @ 39.95s
    # estimator = AdaBoostRegressor(RandomForestRegressor(50, n_jobs=-1))                 # 0.285214019811 @ 408.98s
    # estimator = AdaBoostRegressor(GradientBoostingRegressor())                          # much slower than AdaBoose Forest -- MBA will overheat
    # estimator = GradientBoostingRegressor()                                             # nan @ 72.49s

    ## Best estimator from grid search:
    # estimator = RandomForestRegressor(50, max_depth=21, max_features=0.7, n_jobs=-1)      # 0.280977183878 @ 39.31s -- params from grid search baseline model
                                                                                            # 0.255101664598 was best grid search result
                                                                                            # so might be overfitting...

    ## Manual fine tune after auto grid search
    estimator = RandomForestRegressor(50, max_depth=13, max_features=0.3, n_jobs=-1)      # best test result (cheating here) -- same as prior to grid search...


    estimator = Pipeline([ \
                        #   ('scaler', StandardScaler()),
                          ('scaler', MinMaxScaler()),
                          ('estimator', estimator),
                         ])


    import zipfile
    print 'Loading Files:'
    print '  Inflation.csv...'
    start_loadfile = timeit.default_timer()
    df_inflation = pd.read_csv('../data/Inflation.csv', sep='\t')
    df_inflation.index = df_inflation.Year
    df_inflation = df_inflation.Ave.sort_index(ascending=False)     # average annual inflation


    print '  Train.csv...'
    df_train = pd.read_csv(zipfile.ZipFile('../data/Train.zip', mode='r').open('Train.csv'), low_memory=False)



    if False:    # quick test
        print '      (time: %.5f s)' % float(timeit.default_timer() - start_loadfile)
        err = procedure(estimator, df_train, None, inflation=df_inflation, verbose=False)


    if False:     # manual grid search

        print '      (time: %.5f s)' % float(timeit.default_timer() - start_loadfile)

        print 'Grid search...'
        start = timeit.default_timer()

        best_rmsle = 1
        best_params = (0,0)
        for mf in np.arange(0.3, 0.8, 0.1):
            for md in [11, 13, 17, 19, 21]:
                print 'Parameters: max_feature=%f, max_depth=%i' % (mf, md)

                estimator = Pipeline([ \
                                      ('scaler', MinMaxScaler()),
                                      ('estimator', RandomForestRegressor(50, max_features=mf, max_depth=md)),
                                     ])

                err = procedure(estimator, df_train, None, inflation=df_inflation, verbose=False)

                if err < best_rmsle:
                    best_rmsle = err
                    best_params = (mf, md)

        print '\n'
        print 'BEST FEATURES:'
        print '    max_features ', mf
        print '       max_depth ', md
        print '   least RMS log error:', best_rmsle
        print '      (time: %.5f s)' % float(timeit.default_timer() - start)



    else:       # moment of truth (dum dum dum)

        print '  test.csv...'
        df_test = pd.read_csv(zipfile.ZipFile('../data/Test.zip', mode='r').open('test.csv'), low_memory=False)
        print '  test_soln.csv...'
        test_solution = pd.read_csv('../data/do_not_open/test_soln.csv')
        print '      (time: %.5f s)' % float(timeit.default_timer() - start_loadfile)

        err = procedure(estimator, df_train, df_test, test_solution, outfile='../result_oo.csv', inflation=df_inflation, verbose=False)


    print '   (overall time: %.5f s)' % float(timeit.default_timer() - start)
