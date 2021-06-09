import xgboost as xgb
from imblearn.under_sampling import NearMiss
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import Lasso, LinearRegression, LogisticRegression

from ..feature_engineering import add_relevant_features
from ..io import get_random_state
from ..prediction import make_predictions


def make_baseline_predictions(df):
    """
    This function performs baseline predictions for both classification and
    regression. Those can be used to evaluate fall other machine learning algorithms,
    as it provides the required point of comparison.
    ----------------------------------------------
    :param
            df (pandas.DataFrame): the given pandas data frame containing data
                                  used for prediction.
    :return:
    """
    classification_model = LogisticRegression(
        multi_class="multinomial",
        class_weight="balanced",
        n_jobs=-1,
        random_state=get_random_state(),
        max_iter=5000,
    )
    regression_model = LinearRegression(n_jobs=-1)

    nr = NearMiss()

    # base_line classification for "payment_type"
    make_predictions(
        df=df,
        prediction_type="classification",
        target="payment_type",
        model=classification_model,
        relevant_features={
            "target": "payment_type",
            "cyclical_features": [
                "pickup_month",
                "pickup_day",
                "pickup_hour",
                "dropoff_hour",
                "dropoff_day",
                "dropoff_month",
            ],
            "categorical_features": ["Zone_pickup", "Zone_dropoff"],
            "numerical_features": [
                "passenger_count",
                "trip_distance",
                "total_amount",
                "trip_duration_minutes",
            ],
        },
        scaler_type="standard_scaler",
        model_name="base_clas_payment_type",
    )

    # base_line classification for "payment_type" using under-sampling (near-miss)
    make_predictions(
        df=df,
        prediction_type="classification",
        target="payment_type",
        model=classification_model,
        relevant_features={
            "target": "payment_type",
            "cyclical_features": [
                "pickup_month",
                "pickup_day",
                "pickup_hour",
                "dropoff_hour",
                "dropoff_day",
                "dropoff_month",
            ],
            "categorical_features": ["Zone_pickup", "Zone_dropoff"],
            "numerical_features": [
                "passenger_count",
                "trip_distance",
                "total_amount",
                "trip_duration_minutes",
            ],
        },
        scaler_type="standard_scaler",
        model_name="base_clas_payment_type_nm",
        use_sampler=True,
        sampler=nr,
    )

    # base_line regression for "trip_distance"
    make_predictions(
        df=df,
        prediction_type="regression",
        target="trip_distance",
        model=regression_model,
        relevant_features={
            "target": "trip_distance",
            "categorical_features": ["Zone_dropoff", "Zone_pickup"],
            "cyclical_features": ["pickup_month", "pickup_day", "pickup_hour"],
            "numerical_features": ["passenger_count"],
        },
        scaler_type=None,
        model_name="base_reg_trip_distance",
    )

    # base-line regression for "fare_amount"
    make_predictions(
        df=df,
        prediction_type="regression",
        target="fare_amount",
        model=regression_model,
        scaler_type=None,
        relevant_features={
            "target": "fare_amount",
            "categorical_features": ["Zone_dropoff", "Zone_pickup"],
            "cyclical_features": ["pickup_month", "pickup_day", "pickup_hour"],
            "numerical_features": ["passenger_count"],
        },
        model_name="base_reg_fare_amount",
        use_sampler=False,
        sampler=None,
    )


def trip_distance_regression_base(df):
    """
    This function predicts the trip distance with using xgboost. Information is used that was available at
    the start of the trip (including additional created features).
    :param
            df (pandas.DataFrame): the given pandas data frame containing data
                                  used for prediction.
           feature_selection (boolean): If features should be selected to improve estimators’
                                        accuracy scores or to boost their performance.
    """

    feature_selector = SelectFromModel(Lasso(alpha=0.01))
    df = add_relevant_features(df, "pickup_datetime")

    make_predictions(
        df=df,
        prediction_type="regression",
        target="trip_distance",
        relevant_features={
            "target": "trip_distance",
            "categorical_features": ["Zone_dropoff", "Zone_pickup"],
            "cyclical_features": [],
            "numerical_features": [
                "passenger_count",
                "Holiday",
                "covid_lockdown",
                "covid_school_restrictions",
                "covid_new_cases",
                "pickup_month",
                "pickup_day",
                "pickup_hour",
                "haversine_distance",
                "bearing_distance",
                "manhattan_distance",
                "weekend",
                "weekday",
            ],
        },
        feature_selector=feature_selector,
        feature_selection=True,
        model=xgb.XGBRegressor(n_estimators=100),
        model_name="xg_boost",
        scaler_type=None,
        use_sampler=False,
        show_feature_importance=True,
        sampler=None,
    )


def build_fare_amount_model_base(df):
    """
    This Function predicts a fare amount based on training data, using XGBoost Regressor.
    This Function should be called only with a sub sample of the available data.
    This is a basic model with no hyper parameter being tuned.
    -------------------------------------------------------------------------------
    :param
        df(pandas.DataFrame): the given pandas data frame containing data
                                  used for prediction.
    """
    # The pickup month/day/hour will not be transformed as
    # there is no need for cyclical transformation when using a decision tree
    relevant_features = {
        "target": "fare_amount",
        "categorical_features": ["Zone_dropoff", "Zone_pickup"],
        "numerical_features": [
            "trip_distance",
            "trip_duration_minutes",
            "pickup_month",
            "pickup_day",
            "pickup_hour",
            "dropoff_month",
            "dropoff_day",
            "dropoff_hour",
        ],
        "cyclical_features": [],
    }

    feature_selector = SelectFromModel(Lasso(alpha=0.1))
    model = xgb.XGBRegressor(n_jobs=-1, n_estimators=100)
    make_predictions(
        df=df,
        relevant_features=relevant_features,
        target="fare_amount",
        scaler_type=None,
        prediction_type="regression",
        model_name="xgb_model_fare_amount_base",
        model=model,
        feature_selection=True,
        feature_selector=feature_selector,
        show_feature_importance=True,
        drop_first_category=False,
    )


def fare_amount_hyper_parameter_optimization(df):
    """
    This Function run a grid search on the data and print out the best hyper parameters for a the fare amount model
    This Function should be called only with a sub sample of the available data.
    -------------------------------------------------------------------------------
    :param
        df(pandas.DataFrame): the given pandas data frame containing data
                                  used for grid search.
    """
    # The pickup month/day/hour will not be transformed as
    # there is no need for cyclical transformation when using a decision tree
    relevant_features = {
        "target": "fare_amount",
        "categorical_features": [],
        "numerical_features": [
            "trip_distance",
            "trip_duration_minutes",
            "pickup_month",
            "pickup_hour",
        ],
        "cyclical_features": [],
    }

    model = xgb.XGBRegressor(n_jobs=-1, subsample=0.7, colsample_bytree=0.8)
    model_params = {
        "xgb_fare_amount_model__learning_rate": [0.1, 0.05, 1],
        "xgb_fare_amount_model__max_depth": [3, 5, 7, 10, 20],
        "xgb_fare_amount_model__min_child_weight": [1, 4, 7],
        "xgb_fare_amount_model__reg_lambda": [5, 10, 50],
        "xgb_fare_amount_model__subsample": [0.5, 0.7, 1],
        "xgb_fare_amount_model__colsample_bytree": [0.5, 0.7, 0.9],
        "xgb_fare_amount_model__n_estimators": [60, 80, 100],
    }
    make_predictions(
        df=df,
        relevant_features=relevant_features,
        target="fare_amount",
        scaler_type=None,
        prediction_type="regression",
        model_name="xgb_fare_amount_model",
        model=model,
        feature_selection=False,
        show_feature_importance=True,
        drop_first_category=False,
        is_grid_search=True,
        grid_search_params=model_params,
        scoring="neg_mean_absolute_error",
    )


def build_fare_amount_model_optimized(df, manhattan=False):
    """
    This Function predicts a fare amount based training data, using XGBoost Regressor.
    This model uses the optimized hyper parameters and the selected features in the base model
    -------------------------------------------------------------------------------
    :param
        df(pandas.DataFrame): the given pandas data frame containing data
                                  used for prediction.
        manhattan(bool): If set to true the model will only considers the pickup/dropoff from/to manhattan. Default: False
    """
    # The pickup month/day/hour will not be transformed as
    # there is no need for cyclical transformation when using a decision tree
    relevant_features = {
        "target": "fare_amount",
        "categorical_features": [],
        "numerical_features": [
            "trip_distance",
            "trip_duration_minutes",
            "pickup_month",
            "pickup_hour",
        ],
        "cyclical_features": [],
    }

    model = xgb.XGBRegressor(
        n_jobs=-1,
        n_estimators=80,
        learning_rate=0.1,
        max_depth=10,
        min_child_weight=1,
        reg_lambda=10,
        subsample=0.7,
        colsample_bytree=0.9,
    )

    make_predictions(
        df=df,
        relevant_features=relevant_features,
        target="fare_amount",
        scaler_type=None,
        prediction_type="regression",
        model_name="xgb_model_fare_amount_optimized",
        model=model,
        feature_selection=False,
        show_feature_importance=True,
        drop_first_category=False,
        is_manhattan=manhattan,
    )
