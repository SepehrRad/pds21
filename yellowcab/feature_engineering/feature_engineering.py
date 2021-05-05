from datetime import date, datetime
from pandas.tseries.holiday import USFederalHolidayCalendar as calendar

def add_relevant_features(data_set, date_colummn):
    """
        This function adds all in the sub-package "feature_engineering" created columns
        to the input dataset, based on the input date_column in case it is based on time.
        ----------------------------------------------
        :param
            data_set: Dataframe to what the columns should be added.
            date_colummn: The column we use for comparing the dates.
        :return:
            dataframe: The input dataframe with the new columns added.
    """
    data_set = create_holiday_column(data_set, date_colummn)
    data_set = create_season_column(data_set, date_colummn)
    data_set = create_covid_relevant_features(data_set, date_colummn)
    return data_set

def _get_season_in_ny(date_time):
    """
        This function returns the season which the input date belongs to.
        ----------------------------------------------
        :param
            date_time: The date we want to get the season for.
        :returns
            String: Name of the season.
    """
    seasons = [('winter', (date(2020, 1, 1), date(2020, 3, 18))),
               ('spring', (date(2020, 3, 19), date(2020, 6, 19))),
               ('summer', (date(2020, 6, 20), date(2020, 9, 20))),
               ('autumn', (date(2020, 9, 21), date(2020, 12, 20))),
               ('winter', (date(2020, 12, 21), date(2020, 12, 31)))]
    if isinstance(date_time, datetime):
        date_time = date_time.date()
    date_time = date_time.replace(year=2020)
    try:
        return next(season for season, (start, end) in seasons
                    if start <= date_time <= end)
    except StopIteration:
        pass

def create_season_column(data_set, date_column):
    """
        This function adds a column stating if the date handed over is a holiday day in the US.
        ----------------------------------------------
        :param
            data_set: Dataframe to what the column should be added.
            date_column: The column we use for comparing the dates.

        :returns
            dataframe: The input dataframe with a season column added.
    """
    data_set['Season'] = data_set[date_column].apply(_get_season_in_ny)
    return data_set


def create_holiday_column(data_set, date_column):
    """
        This function adds a column saying if the date is a holiday day in NY.
        ----------------------------------------------
        :param
            data_set: Dataframe to what the column should be added.
            date_column: The column we use for comparing the dates.

        :returns
            dataframe: The input dataframe with a holiday day column added.
        """
    cal = calendar()
    holidays = cal.holidays(data_set[date_column].min(), data_set[date_column].max())
    data_set['Holiday'] = data_set[date_column].dt.date.astype('datetime64').isin(holidays)
    return data_set

def get_covid_restrictions():
    """
        Information for first covid case:
        https://www.investopedia.com/historical-timeline-of-covid-19-in-new-york-city-5071986
        https://news.google.com/covid19/map?hl=de&mid=%2Fm%2F02_286&gl=DE&ceid=DE%3Ade
        This function returns relevant covid restriction time ranges.
        Information for lockdown time-range:
        https://ny.eater.com/2020/12/30/22203053/nyc-coronavirus-timeline-restaurants-bars-2020
        Information for school restrictions time-range:
        https://ballotpedia.org/School_responses_in_New_York_to_the_coronavirus_(COVID-19)_pandemic_during_the_2020-2021_school_year
        https://mommypoppins.com/new-york-city-kids/schools/heres-the-nyc-public-school-calendar-for-2020-2021
        ----------------------------------------------
        :return
            dictionary
    """
    covid_restrictions = {
        "covid_new_cases_start": datetime(day=1, month=3, year=2020),
        "covid_new_cases_end": datetime(day=31, month=12, year=2020),
        "covid_lockdown_start": datetime(day=16, month=3, year=2020),
        "covid_lockdown_end": datetime(day=8, month=6, year=2020),
        "covid_school_restrictions_start": datetime(day=16, month=3, year=2020),
        "covid_school_restrictions_end": datetime(day=20, month=3, year=2020)
    }
    return covid_restrictions

def create_covid_relevant_features(data_set, date_column):
    """
        This function adds a column adds column for relevant covid restrictions matching the dates.
        ----------------------------------------------
        :param
            data_set: Dataframe to what the columns should be added.
            date_column: The column we use for comparing the dates.
        :return
            dataframe: The input dataframe with covid restrictions columns added.
    """
    covid_restrictions = get_covid_restrictions()
    data_set.loc[data_set[date_column].between(covid_restrictions.get("covid_new_cases_start"), covid_restrictions.get(
        "covid_new_cases_end")), "covid_new_cases"] = 1
    data_set.loc[data_set[date_column].between(covid_restrictions.get("covid_lockdown_start"), covid_restrictions.get(
        "covid_lockdown_end")), "covid_lockdown"] = 1
    data_set.loc[data_set[date_column].between(covid_restrictions.get("covid_school_restrictions_start"), covid_restrictions.get(
        "covid_school_restrictions_end")), "covid_school_restrictions"] = 1

    data_set["covid_new_cases"].fillna(0, inplace=True)
    data_set["covid_lockdown"].fillna(0, inplace=True)
    data_set["covid_school_restrictions"].fillna(0, inplace=True)

    return data_set