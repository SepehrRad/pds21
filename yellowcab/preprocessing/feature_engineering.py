from datetime import date, datetime
from pandas.tseries.holiday import USFederalHolidayCalendar as calendar


def get_season_in_ny(date_time):
    """
                This function says what season the input belongs to.
                ----------------------------------------------
                :param
                    datr_time

                :returns
                    String
        """
    seasons = [('winter', (date(2020, 1, 1), date(2020, 3, 18))),
               ('spring', (date(2020, 3, 19), date(2020, 6, 19))),
               ('summer', (date(2020, 6, 20), date(2020, 9, 20))),
               ('autumn', (date(2020, 9, 21), date(2020, 12, 20))),
               ('winter', (date(2020, 12, 21), date(2020, 12, 31)))]

    if isinstance(date_time, datetime):
        date_time = date_time.date()
    date_time = date_time.replace(year=2020)
    return next(season for season, (start, end) in seasons
                if start <= date_time <= end)


def create_season_column(data_set, date_column):
    """
            This function adds a column saying if the date is a holiday day in NY.
            ----------------------------------------------
            :param
                data_set
                date_column

            :returns
                data_set
    """
    local = data_set.copy()
    local['Season'] = local[date_column].apply(get_season_in_ny)
    return local


def create_holiday_column(data_set, date_column):
    """
            This function adds a column saying if the date is a holiday day in NY.
            ----------------------------------------------
            :param
                data_set
                date_column

            :returns
                data_set
        """
    cal = calendar()
    holidays = cal.holidays(data_set[date_column].min(), data_set[date_column].max())
    data_set['Holiday'] = data_set[date_column].dt.date.astype('datetime64').isin(holidays)
    return data_set
