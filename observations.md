Observations on features:

**All observations below are for RandomForestRegressor(50, n_jobs=-1, max_depth=13, max_features=0.4)**

(the values max_depth=13 and max_feature=0.4 are from doing *really quick* GridSearch on each ProductGroup individually in Attempt 2 and take the majority vote-ish average)


- group models by ProductGroup, fiBaseModel and ModelID then compute average price and age in each group
    - averaging method: boostrap mean or median works better
    - bootstrap mean: larger boostrap sample size does not further improve model
    - fiBaseModel -- with it model improves by ~O(0.002)

- discard outliers/leverage points -- improves model by ~O(0.02)

- adjust for inflation (bring all prices to Present Value when modelling then discount prdictions back) -- improves model by ~0(0.02)

- observed in Price vs. Age plots (did not adjust for inflation)
    - min price for every ProductGroup stays constant over time
    - high price for every ProductGroup increases over timeit
    - maybe there is heteroscedasticity?
        - tried predicting log(y) and sqrt(y) instead of y -- but both are worse off by ~O(0.01)


- features:
    - average price based on observation's ProductGroup, fiBaseModel and ModelID
    - average price per age = sum(price / (Age+1) of each observation) -- Age + 1 to avoid division by zero
    - adjusted_age -- use to calculate adjusted_age*avg_price
        - better not to further include it as an individual feature -- model worse off by ~O(0.01)
        - two ways to compute adjusted_age:
            * adjusted_age = average age based observation's ProductGroup, fiBaseModel and ModelID  + 1  works less well
            * adjusted_age = average age of observation's ProductGroup + 1 works better ~O(0.01)
            - maybe because breaking collinearity between avg_price, adjusted_age*avg_price, adjusted_price??
    - ProductGroup dummies
        - include it improve model by ~0.1
        - multiplying it by a factor of 100 improves the model by ~0.03 (why??)
        0 WITHOUT IT CHANGES-- * 100 improves model by 0.03
    - states dummies -- adding this worsen the model by ~0.2

- Normalise X
    - model better off with min-max scaler ~O(0.2)
    - model worse off with StandardScaler (standard normalisation) and Normalizer (unit-length)
    - kinda make sense -- Price (face value) vs. Age plots do not look Gaussian


**With these calculations model takes ~30 seconds to run.**



**All observations below use the best configuration from above**

Observations on models:

    - linear models are no good -- RMS log error = nan
    - without tuning, RandomForest better (faster and less error) than AdaBoost
    - merge attempt 1 and this attempt (3.2) -- running 6 models, one for each group -- result is not better, only on par

Observations on grid search:

- RandomForestRegressor:
    - lower max_features from 0.7 to 0.5 speeds up ~6s
    - lower max_features from 21 to 15 only speeds up ~1s
    - doubling features form 50 to 100 @ (11, 0.5) takes ~6s longer, but doesn't improve model performance
    - best parameters (max_depth=21, max_features=0.7) gave lower training error but larger test error than not-tuned model from above -- overfitting?
    - after manual tune -- back to the same max_depth=13, max_features=0.4
