import math

from imblearn.metrics import classification_report_imbalanced
from matplotlib import pyplot as plt
from sklearn import metrics
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

from yellowcab.io.output import save_model
from yellowcab.io.utils import flatten_list, get_random_state, get_zone_information
from yellowcab.preprocessing import transform_columns


def _make_data_preparation(df, relevant_features, is_manhattan=False, drop_first=False, use_created_features=False):
    """
    This function reduces the dataframe to one containing only relevant features
    for prediction purposes.
    ----------------------------------------------
    :param
        df (pandas.DataFrame): The given pandas data frame with all initial features.
        relevant_features (list): The given features upon which the model should be created.
        is_manhattan(bool): If True only the manhattan data gets selected.
        use_created_features(bool): Denotes if columns created during the data preparation step are included
        in relevant features and should be used.
    :return:
        pandas.DataFrame: Data frame containing only those features which
             are relevant for prediction.
    """
    if use_created_features:
        created_features = relevant_features.copy()
        created_features.pop("categorical_features")
    if "created_features" in relevant_features:
        relevant_features.pop("created_features")
    df = get_zone_information(df, zone_file="taxi_zones.csv")
    mask = flatten_list(list(relevant_features.values()))
    if is_manhattan:
        df = df.loc[
             (df.Borough_pickup == "Manhattan") | (df.Borough_dropoff == "Manhattan"), :
             ]
    df = df[mask]
    df = transform_columns(df=df, col_dict=relevant_features, drop_first=drop_first)
    if use_created_features:
        final_feature_list = flatten_list(list(created_features.values()))
        df = df[final_feature_list]
    return df


def _make_pipeline(
        model, model_name, feature_selector=None, feature_selection=False, scaler_type=None
):
    """
    This function assembles several steps that can be cross-validated together
    while setting different parameters.
    ----------------------------------------------
    :param
        model: Used model for prediction.
        scaler_type (boolean): What scaler should be used to transform our data.
        model_name (String): Name of used model for prediction.
        feature_selection (boolean): If features should be selected to improve estimators’
                                        accuracy scores or to boost their performance.
        feature_selector: What feature selector should be applied.
    :return:
        sklearn.pipeline: Sequentially applies the list of transformers
        and a final estimator
    """
    steps = []
    if scaler_type is not None:
        if scaler_type == "Robust":
            scaler = ("robust_scaler", RobustScaler())  # with outlier detection
            steps.append(scaler)
        else:
            scaler = ("standard_scaler", StandardScaler())
            steps.append(scaler)
    if feature_selection and feature_selector is not None:
        feature_selector_step = ("feature_selector", feature_selector)
        steps.append(feature_selector_step)
    prediction_model = (model_name, model)
    steps.append(prediction_model)
    return Pipeline(steps=steps)


def _make_train_test_split(df, target, use_sampler, sampler):
    """
    This function splits the input data set into a train an a test set, each
    for the regressors X and the dependent variable y.
    ----------------------------------------------
    :param df (pandas.DataFrame): The given pandas data frame containing data which
                                  need to be split into train and test data sets.
           target (String): Dependent variable for prediction purposes.
           use_sampler (boolean): Denotes, whether a sampler is used to handle imbalanced data with over-/ under-sampling.
           sampler: What sampler should be used for over-/ under-sampling.
    :return:
        X_train (pandas.DataFrame): Regressors used for training a model
        X_test (pandas.DataFrame): Regressors used for testing a model
        y_train (pandas.Series): Target values for training a model
        y_test (pandas.Series): Target values for testing a model
    """
    y = df.pop(target)
    X = df

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=get_random_state()
    )
    if use_sampler:
        test = sampler
        X_train, y_train = test.fit_resample(X_train, y_train)
    return X_train, X_test, y_train, y_test


def _print_prediction_scores(prediction_type, y_test, X_test, pipeline):
    """
    This function prints the prediction scores for either a regression or
    a classification.
    ----------------------------------------------
    :param
        prediction_type (String): Denotes whether used for regression or classification.
        y_test (pandas.Series): Target values for testing a model
        X_test (pandas.DataFrame): Regressors used for testing a model
        pipeline (sklearn.pipeline): Sequentially applies a list of transforms and a final estimator.
    """
    if prediction_type == "classification":
        print("-------MODEL SCORES-------")
        print(classification_report_imbalanced(y_test, pipeline.predict(X_test)))
    else:
        y_pred = pipeline.predict(X_test)
        print("-------MODEL SCORES-------")
        print(f"MAE: {metrics.mean_absolute_error(y_test, y_pred): .3f}")
        print(f"MSE: {metrics.mean_squared_error(y_test, y_pred): .3f}")
        print(f"RMSE: {math.sqrt(metrics.mean_squared_error(y_test, y_pred)): .3f}")
        print(f"R2: {100 * metrics.r2_score(y_test, y_pred): .3f} %")


def _get_information_for_feature_selection(pipeline, X_train):
    """
    This function returns the result of the performed feature selection process.
    :param
        pipeline (sklearn.pipeline): Sequentially applies a list of transforms and a final estimator.
        X_train (pandas.DataFrame):  Regressors used for training a model
    """
    selected_feature_mask = pipeline.named_steps["feature_selector"].get_support()
    original_features = len(X_train.columns)
    selected_features = len(X_train.columns[selected_feature_mask])
    print(
        f"{selected_features} features were selected for prediction from the original {original_features} features."
    )


def _get_sample_weight(y_train):
    """
    This function calculates class weights for XGBClassifier in order to balance out the imbalanced class distribution.
    ----------------------------------------------
    :param
           df (pandas.DataFrame): The given pandas data frame containing data.
           target (String): Dependent variable for prediction purposes.

    """
    sample_weight = compute_sample_weight("balanced", y_train)
    return sample_weight


def make_predictions(
        df,
        prediction_type,
        target,
        model,
        model_name,
        scaler_type,
        relevant_features,
        feature_selector=None,
        feature_selection=False,
        use_sampler=False,
        sampler=None,
        show_feature_importance=False,
        drop_first_category=True,
        is_grid_search=False,
        grid_search_params=None,
        scoring=None,
        is_manhattan=False,
        use_created_features=False,
        weigh_classes=False,
):
    """
    This function predicts and prints the prediction scores of a prediction
    task dynamically defined by the input parameters.
    ----------------------------------------------
    :param
           df (pandas.DataFrame): the given pandas data frame containing data used for prediction.
           prediction_type (String: Denotes whether used for regression or classification.
           target (String): Dependent variable for prediction purposes.
           model: Used model for prediction.
           model_name (String): Name of used model for prediction.
           scaler_type (String): scaler_type: What scaler should be used to transform our data.
           relevant_features (list): Those features that should be used for prediction.
           feature_selection (boolean): If features should be selected to improve estimators’
                                        accuracy scores or to boost their performance.
           use_sampler (boolean): denotes, whether a sampler is used to handle imbalanced data with over-/ under-sampling.
           sampler: what sampler should be used for over-/ under-sampling.
           is_grid_search (boolean): It denotes if the function should be used for hyperparameter search instead of prediction. Defaults to True.
           grid_search_params (dict): The grid search parameter space that should be used.
           scoring(String): the scoring methode which will be used in the grid search cv.
           is_manhattan(bool): Builds a prediction model only for manhattan data
           use_created_features(bool): Denotes if columns created during the data preparation step are included in relevant features and should be used.
           weigh_classes(bool): denotes if the target classes should be weight to help with a imbalance dataset
    """
    if not is_grid_search:
        print(
            f"\nPredicted target: {target}, model name: {model_name}, prediction type: {prediction_type}"
        )
    else:
        print("-------GRID SEARCH-------")

    if is_manhattan:
        model_name = model_name + "_manhattan"

    df = _make_data_preparation(
        df,
        relevant_features=relevant_features,
        drop_first=drop_first_category,
        is_manhattan=is_manhattan,
        use_created_features=use_created_features,
    )
    X_train, X_test, y_train, y_test = _make_train_test_split(
        df=df, target=target, sampler=sampler, use_sampler=use_sampler
    )

    if feature_selection:
        pipeline = _make_pipeline(
            model=model,
            model_name=model_name,
            scaler_type=scaler_type,
            feature_selection=feature_selection,
            feature_selector=feature_selector,
        )
    else:
        pipeline = _make_pipeline(
            model=model, model_name=model_name, scaler_type=scaler_type
        )

    if weigh_classes:
        sample_weight = _get_sample_weight(y_train)
        pipeline_sample_weight = {pipeline.steps[-1][0] + '__sample_weight': sample_weight}

    if not is_grid_search:
        if not weigh_classes:
            pipeline = pipeline.fit(X_train, y_train)
        else:
            pipeline = pipeline.fit(X_train, y_train, **pipeline_sample_weight)
        if feature_selection:
            _get_information_for_feature_selection(pipeline=pipeline, X_train=X_train)
        if show_feature_importance and feature_selection:
            selected_feature_mask = pipeline.named_steps[
                "feature_selector"
            ].get_support()
            plt.barh(
                X_train.columns[selected_feature_mask],
                pipeline.named_steps[model_name].feature_importances_,
            )
            plt.show()
        save_model(pipeline.named_steps[model_name], model_name)
        _print_prediction_scores(
            prediction_type=prediction_type,
            y_test=y_test,
            X_test=X_test,
            pipeline=pipeline,
        )
    else:
        if not weigh_classes:
            return find_best_parameters_for_model(
                pipeline=pipeline,
                X_train=X_train,
                y_train=y_train,
                model_name=model_name,
                model_params=grid_search_params,
                scoring=scoring,
            )
        else:
            return find_best_parameters_for_model(
                pipeline=pipeline,
                X_train=X_train,
                y_train=y_train,
                model_name=model_name,
                model_params=grid_search_params,
                scoring=scoring,
                sample_weight=pipeline_sample_weight
            )

def find_best_parameters_for_model(
        pipeline, X_train, y_train, model_params, model_name, scoring, sample_weight=None
):
    """
    This function performs a grid search with three cv on the training set.
    ----------------------------------------------
    :param
           df (pandas.DataFrame): The given pandas data frame containing data which
                                  need to be split into train and test data sets.
           scoring(String): The scoring metric used for grid search
           model: Used model for prediction.
           model_name (String): Name of used model for prediction.
           X_train (pandas.DataFrame): Training features
           y_train (pandas.DataFrame): Target
           pipeline (sklearn.pipeline): The pipeline with the model and transformers which will be used for grid search.
           sample_weight (dict): Sample weight to be used during model fit.
    """
    print(f"Running grid search for {model_name} based on {scoring}")
    grid_pipeline = GridSearchCV(
        estimator=pipeline,
        param_grid=model_params,
        n_jobs=-1,
        cv=3,
        scoring=scoring,
        verbose=True,
    )
    if sample_weight is None:
        grid_pipeline.fit(X_train, y_train)
    else:
        grid_pipeline.fit(X_train, y_train, **sample_weight)
    print(f"Best {scoring} Score was: {grid_pipeline.best_score_}")
    print(f"The best hyper parameters for {model_name} are:")
    print(grid_pipeline.best_params_)
