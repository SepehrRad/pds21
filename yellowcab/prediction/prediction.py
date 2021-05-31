#%%
from imblearn.under_sampling import NearMiss
from sklearn.model_selection import train_test_split
from yellowcab.preprocessing import transform_columns
from yellowcab.io.output import save_model
from yellowcab.io.input import read_model
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import RobustScaler
from imblearn.metrics import classification_report_imbalanced
from sklearn.linear_model import LogisticRegression, LinearRegression
import numpy as np
import pandas as pd
from sklearn import metrics
import matplotlib.pyplot as plt
from yellowcab.io.utils import get_zone_information
from yellowcab.feature_engineering import add_relevant_features

#%%
def _get_random_state():
    """
    This function defines our random state, so that it's the same in
    every application
    ----------------------------------------------
    :return:
        int: the integer used for random state paramenter
    """
    RANDOM_STATE = 7
    return RANDOM_STATE

def _get_column_description_for_prediction():
    """
    This function defines the categories our features belong to.
    ----------------------------------------------
    :return:
        dictionary: Dictionary containing our feature categories with the associated attributes.
    """
    column_description = {'cyclical_features':['pickup_month','pickup_day','pickup_hour', 'dropoff_hour','dropoff_day','dropoff_month'],
                          'categorical_features':['RatecodeID', 'payment_type', 'Season'],
                          'temporal_features':['pickup_datetime','dropoff_datetime'],
                          'spatial_features':['LocationID_pickup', 'Borough_pickup' , 'Zone_pickup', 'service_zone_pickup', 'LocationID_dropoff', 'Borough_dropoff', 'Zone_dropoff', 'service_zone_dropoff']}
    return column_description

def _make_data_preparation(df, prediction_type, target):
    """
    TODO: As soon as Simon has corrected weekend in cleaning, line can be removed.
    This function reduces the dataframe to one containing only relevant features
    for prediction purposes.
    ----------------------------------------------
    :param df (pandas.DataFrame): The given pandas data frame with all initial features
           prediction_type (String): Denotes whether used for regression or classification.
           target: Dependent variable for prediction purposes.
    :return: pandas.DataFrame: Data frame containing only those features which
             are relevant for prediction.

              32  pickup_month_sine              10000 non-null  float64
 33  pickup_month_cosine            10000 non-null  float64
 34  pickup_day_sine                10000 non-null  float64
 35  pickup_day_cosine              10000 non-null  float64
 36  pickup_hour_sine               10000 non-null  float64
 37  pickup_hour_cosine             10000 non-null  float64
 38  dropoff_hour_sine              10000 non-null  float64
 39  dropoff_hour_cosine            10000 non-null  float64
 40  dropoff_day_sine               10000 non-null  float64
 41  dropoff_day_cosine             10000 non-null  float64
 42  dropoff_month_sine             10000 non-null  float64
 43  dropoff_month_cosine

    """
    df = get_zone_information(df, zone_file="taxi_zones.csv")
    if prediction_type == "regression":
        if target == "trip_distance":
            print("\nTRIP DISTANCE:")
            column_description = _get_column_description_for_prediction()
            new_df = transform_columns(df,column_description)
            new_df.drop(column_description.get('temporal_features'),inplace=True, axis=1)
            new_df.drop(column_description.get('spatial_features'), inplace=True, axis=1)
            new_df = new_df[new_df.columns.drop(list(new_df.filter(regex='dropoff')))]
            new_df = new_df[new_df.columns.drop(list(new_df.filter(regex='Ratecode')))]
            new_df = new_df[new_df.columns.drop(list(new_df.filter(regex='payment')))]
            new_df = new_df[new_df.columns.drop(['tolls_amount', 'tip_amount', 'trip_duration_minutes', 'DOLocationID'])]
            new_df.pop('weekend') # remove after Simons correction
        if target == "fare_amount":
            print("\nFARE AMOUNT:")
            column_description = _get_column_description_for_prediction()
            cols = [col for col in df.columns if 'Zone' in col]
            cols.append("fare_amount")
            cols.append('pickup_month')
            cols.append('pickup_day')
            cols.append('pickup_hour')
            cols.append('dropoff_hour')
            cols.append('dropoff_day')
            cols.append('dropoff_month')
            cols
            df = df[cols]
            column_description["categorical_features"] = df.loc[:, df.columns.str.contains('^Zone')].columns.tolist()
            column_description.pop("temporal_features")
            column_description.pop("spatial_features")
            new_df = transform_columns(df, column_description)
    else:
        print("\nCLASSIFICATION:")
        column_description = _get_column_description_for_prediction()
        column_description['categorical_features'] = ['RatecodeID', 'Season'] #payment type should not be transofrmed here
        new_df = transform_columns(df,column_description)
        new_df.drop(column_description.get('spatial_features'),inplace=True, axis=1)
        new_df.drop(column_description.get('temporal_features'),inplace=True, axis=1)
        new_df.pop('weekend') #remove after Simons correction
    return new_df

def _make_pipeline(model,scaler_type, model_name):
    """
    This function assembles several steps that can be cross-validated together
    while setting different parameters.
    ----------------------------------------------
    :param model: Used model for prediction.
           scaler_type: What scaler should be used to transform our data.
           model_name: Name of used model for prediction.
    :return: Pipeline: Sequentially applies the list of transforms
             and a final estimator.
    """
    steps = []
    if scaler_type == 'Robust':
        scaler = ("robust_scaler", RobustScaler()) #with outlier detection on top
        steps.append(scaler)
    else:
        scaler = ("standard_scaler", StandardScaler())
        steps.append(scaler)
    prediction_model=(model_name,model)
    steps.append(prediction_model)
    return Pipeline(steps=steps)

def _make_train_test_split(df,target, use_sampler, sampler):
    """
    This function splits the input data set into a train an a test set, each
    for the regressors X and the dependent variable y.
    ----------------------------------------------
    :param df (pandas.DataFrame): The given pandas data frame containing data which need to be split into train and test data sets.
           target: Dependent variable for prediction purposes.
           use_sampler: Denotes, whether a sampler is used to handle imbalanced data with over-/ under-sampling.
           sampler: What sampler should be used for over-/ under-sampling.
    :return:
        X_train (pandas.DataFrame): Regressors used for training a model
        X_test (pandas.DataFrame): Regressors used for testing a model
        y_train (pandas.Series): Target values for training a model
        y_test (pandas.Series): Target values for testing a model
    """
    y = df.pop(target)
    X = df

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=_get_random_state())
    if use_sampler:
        test = sampler
        X_train, y_train = test.fit_resample(X_train, y_train)
    return X_train, X_test, y_train, y_test

def _print_prediction_scores(prediction_type, y_test, X_test, pipeline):
    """
    This function prints the prediction scores for either a regression or
    a classification.
    ----------------------------------------------
    :param prediction_type (String): Denotes whether used for regression or classification.
           y_test (pandas.Series): Target values for testing a model
           X_test (pandas.DataFrame): Regressors used for testing a model
           pipeline: Sequentially applies a list of transforms and a final estimator.
    :return:
    """
    if prediction_type == "classification":
        print(classification_report_imbalanced(y_test, pipeline.predict(X_test)))
    else:
        print("RMSE: ", np.sqrt(metrics.mean_squared_error(y_test, pipeline.predict(X_test))))
        print("MAE: ", metrics.mean_absolute_error(y_test, pipeline.predict(X_test)))
        print("R2: ", metrics.r2_score(y_test, pipeline.predict(X_test)))


def _get_coefficients(X_train, model, prediction_type):
    """
    This function lists the coefficients of the passed model and plots those.
    :param X_train (pandas.DataFrame): Regressors used for training a model
           model: Used model for prediction.
           prediction_type (String): Denotes whether used for regression or classification.
    :return:
    """
    feature_list = X_train.columns
    #feature_list_with_intercept = feature_list.insert(0, "Intercept")
    mod = read_model("{}.pkl".format(model))
    if prediction_type == "classification":
        coefs = pd.DataFrame(mod.coef_[0], index=feature_list)
    if prediction_type == "regression":
        coefs = pd.DataFrame(mod.coef_.flatten(), index=feature_list)
    coefs.rename(columns={0: "Coef"}, inplace=True)
    coefs = coefs.sort_values(coefs.columns[0], ascending=False)
    print(coefs)

    coefs.plot(kind='barh', figsize=(9, 7))
    plt.title('Coefficients')
    plt.show()


def make_predictions(df, prediction_type, target, model, model_name, scaler_type, sampler_name=None, use_sampler=False,
                     sampler=None):
        """
        This function predicts and prints the prediction scores of a prediction
        task dynamically defined by the input parameters.
        ----------------------------------------------
        :param df (pandas.DataFrame): the given pandas data frame containing data used for prediction.
        :param prediction_type (String): Denotes whether used for regression or classification.
        :param target: Dependent variable for prediction purposes.
        :param model: Used model for prediction.
        :param model_name: Name of used model for prediction.
        :param scaler_type: scaler_type: What scaler should be used to transform our data.
        :param sampler_name: Name of the sampler that should be used for over-/ undersampling
        :param use_sampler: denotes, whether a sampler is used to handle imbalanced data with over-/ under-sampling.
        :param sampler: what sampler should be used for over-/ under-sampling.
        :return:
        """
        df = _make_data_preparation(df, prediction_type, target = target)
        numerical_features = df.select_dtypes(include=['float64']).columns
        X_train, X_test, y_train, y_test = _make_train_test_split(df=df, target=target, sampler=sampler, use_sampler=use_sampler)
        pipeline = _make_pipeline(model = model, model_name = model_name, scaler_type =scaler_type) #parameter overloading
        pipeline = pipeline.fit(X_train, y_train)
        save_model(pipeline.named_steps[model_name], model_name)
        _print_prediction_scores(prediction_type=prediction_type, y_test=y_test, X_test=X_test,
                                 pipeline=pipeline)
        _get_coefficients(X_train, model_name, prediction_type)

def make_baseline_predictions(df):
    """
    This function performs baseline predictions for both classification and
    regression. Those can be used to evaluate fall other machine learning algorithms,
    as it provides the required point of comparison.
    ----------------------------------------------
    :param df (pandas.DataFrame): the given pandas data frame containing data
                                  used for prediction.
    :return:
    """
    df = add_relevant_features(df, 'pickup_datetime')
    classification_model = LogisticRegression(multi_class="multinomial", class_weight="balanced",
                                              n_jobs=-1, random_state=_get_random_state(), max_iter=5000)
    regression_model = LinearRegression()

    nr = NearMiss()

    # classification for "payment_type"
    make_predictions(
        df=df, prediction_type="classification", target ="payment_type", model = classification_model,
        scaler_type= "standard_scaler", model_name="base_clas_payment_type"
    )

    # classification for "payment_type" using under-sampling (near-miss)
    make_predictions(
        df=df, prediction_type="classification", target="payment_type", model=classification_model,
       scaler_type="standard_scaler", model_name="base_clas_payment_type_nm", use_sampler=True, sampler= nr
    )

    # regression for "trip_distance"
    make_predictions(
        df=df, prediction_type="regression", target ="trip_distance", model = regression_model,
        scaler_type= "standard_scaler", model_name="base_reg_trip_distance"
    )

    # regression for "fare_amount"
    make_predictions(
        df=df, prediction_type="regression", target="fare_amount", model=regression_model,
        scaler_type="standard_scaler", model_name="base_reg_fareamount"
    )

#%%

