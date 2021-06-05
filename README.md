## Modelling Steps
1) feature_engineering
2) train_test_splitting
3) mutual_information_and_correlation_feature_selection & rrf_feature_selection
4) feature_selection
5) final_preprocessing
6) cross_validation_LM & cross_validation_RF & cross_validation_XGB
7) merge_cv_results
8) delay_forecasting
9) result_plotting

## feature_engineering
In this step new features are generated using the raw simulation data. <br>
**Followed Steps:**
1) Warm-up and cool-down periods are taken out of the data
2) Free-flow travel time is determined using the simulation runs with no accidents.
3) Experienced delay for each section in each simulation step is calculated by substracting the free-flow travel time from the actualized travel time.
4) Delay vs. Time plots are investigated for different detectors and it is seen that an accident in a road section affects the traffic metrics in both the upstream and downstream detectors.
5) Various features are generated using the simulation metrics. These features can be categorized as lag features, adjacent detector metrics & accident related metrics.
    
    ### Generated Features:
    **Lag Features**
    - Flow Lag (1-10)
    - Density Lag (1-10)
    - Avg. Speed Lag (1-10)
    - Section Travel Time Lag (1-10)
    - Section Travel Time Delay Lag (1-10)

    **Adjacent Detector Features**
    - Flow (next & previous detector)
    - Density (next & previous detector)
    - Avg. Speed (next & previous detector)
    - Section Travel Time (next & previous detector)
    - Section Travel Time Delay (next & previous detector)

    **Accident Related Features**
    - Is there an accident in this timestep?
    - Time passed since the accident started (0 if there is no accident)
    - Distance to the accident point (Infinite if there is no accident)

6) Target column is created as the delay occured in the next timestep in given detector.

### Notes
- Lagged features had NA values for the beginning timesteps. Uniformity assumption is applied and NA values are backfilled using the first non-NA value.
- Previous detector features for the first detector and next detector features for the last detector had NA values. Again, uniformity assumption is applied and NA values are backfilled and forwardfilled for the first and last detector, respectively.

## train_test_splitting
In this step enriched (feature engineered) data is divided into training and testing sets. Since the data consist of time series with different simulation parameters a simple random data splitting could not be used. Instead a simulation parameter aware data splitting approach is used.

**Followed Steps:**
1) Determine unique simulation parameter combinations
2) Randomly mark simulation runs for train and test sets for each simulation parameter combination
3) Build train and test sets using the data of the marked simulation runs

## mutual_information_and_correlation_feature_selection
Feature selection is an important step in machine learning model building. Relevant features are selected and others are dicarded. This step decreases the dimensionality of features, lowers the computation times and improves the model convergence. In this step [mutual information](#https://arxiv.org/pdf/1907.07384.pdf) between features and target is calculated for the training set. Mutual information is usefull since it can capture non-linear relationships between two random variables (i.e. feature and target) compared to correlation analysis which captures linear dependence. Mutual information between two random variables X and Y gives the reduction in entropy (uncertainty) of Y after observing the value of X.

$$ 
    I(X;Y) = H(Y) - H(Y|X) = \iint{p(x,y) log(\frac{p(x,y)}{p(x)(y)})dxdy}

$$
Correlation analysis is also applied to capture linear relationships between features and target.
## rrf_feature_selection
In this step feature importance values of tree based model [regularized random forest](#https://arxiv.org/pdf/1201.1587.pdf) is used. 

Mutual information is calculated between two variables X and Y however there may be cases where two variables $X_1$ and $X_2$ explain the variable Y but neither X1 nor X2 can individually explain the target Y. One such example is $Y = XOR(X_1,X_2)$. Conditional mutual information $I(Y;X_1,X_2)$ is needed in such cases. Random forest feature importance overcomes this by considering all the features (for the given tree) to make a split.

Using regularized random forest over standard random forest, like other regularization methods, gives sparse results. Trees in standard random forest greedily divide the feature space for decreasing the entropy resulting with redundant features in the end. 

Regularized random forest solves this issue by discounting the entropy reduction with a regularization coefficient $\lambda \in [0,1]$ if the given split introduces a new feature to the feature set. If the feature used by the split is already in the feature set no discounting is applied. This discounting discourages the algorithm from adding new features to the feature set if their entropy reduction is not high enough.

## feature_selection
In this step features are ranked by their relevance (importance) scores calculated by mutual information, correlation analysis & regularized random forest feature importance and merged together. Any feature that ranked higher than 30 in any of these feature selection methods is selected for the model.

## final_preprocessing
In this step selected features are extracted from the whole training data and categorical variable "accident lane" is one-hot-encoded.

## cross_validation_LM & cross_validation_RF & cross_validation_XGB
5-repeated 5-fold randomized cross validation is used for hyperparameter tuning and model selection. min-max normalization is used for scaling, mean squared error is used for the comparison metric.
### cross_validation_LM
Linear Regression, Ridge Regression, Lasso regression and Bayesian Ridge Regression are trained and evaluated in this step.
### cross_validation_RF
Random Forest Regressor with different hyperparameters are trained and evaluated in this step.
### cross_validation_XGB
XGBoost Regressor with different hyperparameters are trained and evaluated in this step.

## merge_cv_results
Cross validation results from cross_validation_LM, cross_validation_RF & cross_validation_XGB are merged together and best model-hyperparameter combination is found out to be Random Forest regressor with 100 trees, minimum 10 samples in leaf nodes, 0.3 * number_of_features for number of features to consider in each tree.

## delay_forecasting
Best model-hyperparameter combination chosen from cross validation is re-trained on all training set and tested in test set.

## result_plotting
Plot predicted delay, actual delay, prediction error & cumulative error for given simulation runs.