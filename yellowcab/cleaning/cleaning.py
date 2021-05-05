import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest
from sklearn.covariance import EllipticEnvelope
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error


def __get_date_components(df):
    """
    This function splits the pickup- and dropoff-time columns into 3 columns each, devided by month, day and hour &
    returns the processed DataFrame.

    ----------------------------------------------

    :param
        df(pd.DataFrame): DataFrame to be processed.
    :returns
        df(pd.DataFrame): Processed DataFrame.
    """
    df["start_month"] = df["tpep_pickup_datetime"].dt.month
    df["start_day"] = df["tpep_pickup_datetime"].dt.day
    df["start_hour"] = df["tpep_pickup_datetime"].dt.hour
    df["end_month"] = df["tpep_dropoff_datetime"].dt.month
    df["end_day"] = df["tpep_dropoff_datetime"].dt.day
    df["end_hour"] = df["tpep_dropoff_datetime"].dt.hour
    return df


def __is_weekend(df):
    """
    This function adds a "weekend"-column to the given DataFrame. It indicates whether the the trip is on the
    weekend (TRUE) or not (FALSE) & returns the processed DataFrame.

    ----------------------------------------------

    :param
        df(pd.DataFrame): DataFrame to be processed.
    :returns
        df(pd.DataFrame): Processed DataFrame.
    """
    # get day of week from pickup
    df["weekend"] = df["tpep_pickup_datetime"].dt.dayofweek > 4
    return df


def __get_duration(df):
    """
    This function adds a "trip_duration_minutes"-column to the given DataFrame & returns the processed DataFrame

    ----------------------------------------------

    :param
        df(pd.DataFrame): DataFrame to be processed.
    :returns
        df(pd.DataFrame): Processed DataFrame.
    """
    df["trip_duration_minutes"] = (df["tpep_dropoff_datetime"] - df["tpep_pickup_datetime"]).dt.total_seconds() / 60
    return df


def __to_int(df):
    """
    This function converts the datatype of "passenger_count"-column from "float" to "int" & returns the
    processed DataFrame.

    ----------------------------------------------

    :param
        df(pd.DataFrame): DataFrame to be processed.
    :returns
        df(pd.DataFrame): Processed DataFrame.
    """
    df["passenger_count"] = df["passenger_count"].astype("int")
    return df


def __replace_ids(df):
    """
    This function replaces the IDs of "rate_id_dict" and "payment_type_dict" & returns the processed DataFrame.

    ----------------------------------------------

    :param
        df(pd.DataFrame): DataFrame to be processed.
    :returns
        df(pd.DataFrame): Processed DataFrame.
    """
    rate_id_dict = {1.0: 'Standard rate', 2.0: 'JFK', 3.0: 'Newark', 4.0: 'Nassau/Westchester', 5.0: 'Negotiated fare',
                    6.0: 'Group ride'}
    payment_type_dict = {1.0: 'Credit card', 2.0: 'Cash', 3.0: 'No charge', 4.0: 'Dispute', 5.0: 'Unknown',
                         6.0: 'Voided trip'}
    df['RatecodeID'].replace(rate_id_dict, inplace=True)
    df['payment_type'].replace(payment_type_dict, inplace=True)
    return df


def __categorical(df):
    """
     This function sets several columns as type 'category' & returns the processed DataFrame.

     ----------------------------------------------

     :param
         df(pd.DataFrame): DataFrame to be processed.
     :returns
         df(pd.DataFrame): Processed DataFrame.
     """
    df['RatecodeID'] = df['RatecodeID'].astype('category')
    df['payment_type'] = df['payment_type'].astype('category')
    df['PULocationID'] = df['PULocationID'].astype('category')
    df['DOLocationID'] = df['DOLocationID'].astype('category')
    return df


def remove_outliers(df):
    """
     This function combines several detections of outliers,drops them & returns the processed DataFrame.

     ----------------------------------------------

     :param
         df(pd.DataFrame): DataFrame to be processed.
     :returns
         df(pd.DataFrame): Processed DataFrame.
     """
    # remove voyages
    voyages = df[df["passenger_count"] <= 0]
    df = df[~df.isin(voyages).all(1)]
    # remove trips with trip_distance = 0
    trips = df[df["trip_distance"] <= 0]
    df = df[~df.isin(trips).all(1)]
    # remove invalid ratecodes
    ratecodes = df[(df["RatecodeID"] > 6) | (df["RatecodeID"] < 1)]
    df = df[~df.isin(ratecodes).all(1)]
    # remove invalid PULocations (no geojson data for zones > 263 and unknown boroughs)
    PU = df[(df["PULocationID"] > 263) | (df["PULocationID"] < 1)]
    df = df[~df.isin(PU).all(1)]
    # remove invalid DOLocations (no geojson data for zones > 263 and unknown boroughs)
    DO = df[(df["DOLocationID"] > 263) | (df["DOLocationID"] < 1)]
    df = df[~df.isin(DO).all(1)]
    # remove invalid payments types
    payment = df[(df["payment_type"] > 6) | (df["payment_type"] < 1)]
    df = df[~df.isin(payment).all(1)]
    # remove trips with fare amount < 0
    fare = df[df["fare_amount"] < 0]
    df = df[~df.isin(fare).all(1)]
    # remove trips with extra < 0
    extra = df[df["extra"] < 0]
    df = df[~df.isin(extra).all(1)]
    # remove trips with mta_tax != 0 or != 0.5
    mta = df[(df["mta_tax"] != 0) & (df["mta_tax"] != 0.5)]
    df = df[~df.isin(mta).all(1)]
    # remove trips with tips < 0
    tip = df[df["tip_amount"] < 0]
    df = df[~df.isin(tip).all(1)]
    # remove trips with tolls < 0
    tolls = df[df["tolls_amount"] < 0]
    df = df[~df.isin(tolls).all(1)]
    # remove trips with improvement_surcharge != 0 or != 0.3
    surcharge = df[(df["improvement_surcharge"] != 0) & (df["improvement_surcharge"] != 0.3)]
    df = df[~df.isin(surcharge).all(1)]
    # remove trips with congestion_surcharge < 0
    congestion = df[df["congestion_surcharge"] < 0]
    df = df[~df.isin(congestion).all(1)]
    # remove trips with driving time larger than 8 hours or smaller than 0
    time = df[(df["trip_duration_minutes"].dt.seconds > 28800) | (df["trip_duration_minutes"].dt.seconds < 0)]
    df = df[~df.isin(time).all(1)]
    return df


def clean_dataset(df, y_column, method='isolation', random_state=42):
    """
     This function calls several DataFrame optimization functions from this package,
     calls a chosen ('method') outlier detection algorithm & returns the processed DataFrame.

     ----------------------------------------------

     :param
         df(pd.DataFrame): DataFrame to be processed.
         y_column(String): Name of the target feature column.
         method(String): Name of the outlier detection method that is called within this function.
         random_state(int): Value of random_state variable to make the train and test splits deterministic and
         reproducible results.
     :returns
         pd.DataFrame: Processed DataFrame.
     """
    # df = __replace_ids(df)
    df = __categorical(df)
    df = __to_int(df)
    df = __get_date_components(df)
    df = __is_weekend(df)
    df = __get_duration(df)
    df = df.drop(columns=['tpep_pickup_datetime', 'tpep_dropoff_datetime'])
    # split data into train and test set
    X_train, X_test, y_train, y_test = __split_data(df, y_column, random_state)
    # Testing several outlier detection algorithms
    __baseline(X_train, X_test, y_train, y_test)
    if method == 'isolation':
        X_train, y_train = __isolation_forest(X_train, X_test, y_train, y_test, random_state)
    elif method == 'covariance':
        X_train, y_train = __minimum_covariance_determinant(X_train, X_test, y_train, y_test, random_state)
    elif method == 'local':
        X_train, y_train = __local_outlier_factor(X_train, X_test, y_train, y_test)
    elif method == 'svm':
        X_train, y_train = __one_class_svm(X_train, X_test, y_train, y_test)
    # concat the X and y data together using the old column names
    column_names_x = list(df.columns)
    column_names_x.remove(y_column)
    X_train = pd.DataFrame(X_train, columns=column_names_x)
    y_train = pd.DataFrame(y_train, columns=[y_column])
    return pd.concat([X_train, y_train], axis=1)


def __baseline(X_train, X_test, y_train, y_test):
    """
     This function provides a baseline in performance to which we can compare different outlier identification
     and removal procedures & prints the shape and MAE of the DataFrame with outliers included.

     ----------------------------------------------

     :param
         X_train(pd.DataFrame): Features of the training set.
         X_test(pd.DataFrame): Features of the test set.
         y_train(pd.DataFrame): Target features of the training set.
         y_test(pd.DataFrame): Target features of the test set.
     :returns
         print(): Shape of the training set with outliers included.
         print(): MAE with outliers.
     """
    # summarize the shape of the training dataset
    print('Shape with outliers:', X_train.shape, y_train.shape)
    # fit the model
    model = LinearRegression()
    model.fit(X_train, y_train)
    # evaluate the model
    yhat = model.predict(X_test)
    # evaluate predictions
    mae = mean_absolute_error(y_test, yhat)
    print('MAE with outliers: %.3f' % mae)


def __isolation_forest(X_train, X_test, y_train, y_test, random_state):
    """
     This function detects outliers by making use of Isolation Forest from the sklearn-package & calls the
     __lr_model_performance function.

     ----------------------------------------------

     :param
         X_train(pd.DataFrame): Features of the training set.
         X_test(pd.DataFrame): Features of the test set.
         y_train(pd.DataFrame): Target features of the training set.
         y_test(pd.DataFrame): Target features of the test set.
         random_state(int): Value of random_state variable for reproducible results.
     :returns
         calls(): __lr_model_performance with train and test split and detected outliers.
     """
    # identify outliers in the training dataset
    iso = IsolationForest(random_state=random_state)
    yhat = iso.fit_predict(X_train)
    return __lr_model_performance(X_train, X_test, y_train, y_test, yhat, 'isolation forest')


def __minimum_covariance_determinant(X_train, X_test, y_train, y_test, random_state):
    """
     This function detects outliers by making use of Minimum Covariance Determinant method from the sklearn-package &
      calls the __lr_model_performance function.

     ----------------------------------------------

     :param
         X_train(pd.DataFrame): Features of the training set.
         X_test(pd.DataFrame): Features of the test set.
         y_train(pd.DataFrame): Target features of the training set.
         y_test(pd.DataFrame): Target features of the test set.
         random_state(int): Value of random_state variable for reproducible results.
     :returns
         calls(): __lr_model_performance with train and test split and detected outliers.
     """
    # identify outliers in the training dataset
    ee = EllipticEnvelope(random_state=random_state)
    yhat = ee.fit_predict(X_train)
    return __lr_model_performance(X_train, X_test, y_train, y_test, yhat, 'minimum covariance determinant')


def __local_outlier_factor(X_train, X_test, y_train, y_test):
    """
      This function detects outliers by making use of Local Outlier Factor technique from the sklearn-package &
       calls the __lr_model_performance function.

      ----------------------------------------------

      :param
          X_train(pd.DataFrame): Features of the training set.
          X_test(pd.DataFrame): Features of the test set.
          y_train(pd.DataFrame): Target features of the training set.
          y_test(pd.DataFrame): Target features of the test set.
      :returns
          calls(): __lr_model_performance with train and test split and detected outliers.
      """
    # identify outliers in the training dataset
    lof = LocalOutlierFactor()
    yhat = lof.fit_predict(X_train)
    return __lr_model_performance(X_train, X_test, y_train, y_test, yhat, 'local outlier factor')


def __one_class_svm(X_train, X_test, y_train, y_test):
    """
      This function detects outliers by making use of One-Class SVM from the sklearn-package &
       calls the __lr_model_performance function.

      ----------------------------------------------

      :param
          X_train(pd.DataFrame): Features of the training set.
          X_test(pd.DataFrame): Features of the test set.
          y_train(pd.DataFrame): Target features of the training set.
          y_test(pd.DataFrame): Target features of the test set.
      :returns
          calls(): __lr_model_performance with train and test split and detected outliers.
      """
    # identify outliers in the training dataset
    ocsvm = OneClassSVM()
    yhat = ocsvm.fit_predict(X_train)
    return __lr_model_performance(X_train, X_test, y_train, y_test, yhat, 'one class SVM')


def __split_data(df, y_column, random_state):
    """
      This function splits the given DataFrame into training and test splits.

      ----------------------------------------------

      :param
          df(pd.DataFrame): DataFrame to be processed.
          y_column(String): Name of the target feature column.
          random_state(int): Value of random_state variable to make the train and test splits deterministic and
          reproducible results.
      :returns
          pd.DataFrame: X_train, X_test, y_train, y_test as DataFrames
      """
    X, y = df.drop(columns=y_column).values, df[y_column].values
    return train_test_split(X, y, test_size=0.3, random_state=random_state)


def __lr_model_performance(X_train, X_test, y_train, y_test, yhat, name):
    """
      This function generates Linear Regression models considering the dropped outliers ('yhat') &
       calls the __lr_model_performance function.

      ----------------------------------------------

      :param
          X_train(pd.DataFrame): Features of the training set.
          X_test(pd.DataFrame): Features of the test set.
          y_train(pd.DataFrame): Target features of the training set.
          y_test(pd.DataFrame): Target features of the test set.
          yhat(pd.DataFrame): Outliers to drop
          name(String): Name of the used method to determine outliers
      :returns
          print(): Shape of the Training DataFrames
          print(): MAE of y_test and yhat
          X_train(pd.DataFrame): Feature training split
          Y_train(pd.DataFrame): Target feature training split
      """
    # select all rows that are not outliers
    mask = yhat != -1
    X_train, y_train = X_train[mask, :], y_train[mask]
    # summarize the shape of the updated training dataset
    print(f'Shape {name}:', X_train.shape, y_train.shape)
    # fit the model
    model = LinearRegression()
    model.fit(X_train, y_train)
    # evaluate the model
    yhat = model.predict(X_test)
    # evaluate predictions
    mae = mean_absolute_error(y_test, yhat)
    print(f'MAE {name}: %.3f' % mae)
    return X_train, y_train
