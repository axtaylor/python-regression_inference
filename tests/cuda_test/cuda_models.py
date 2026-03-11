from regression_inference import MultinomialLogisticRegression, OrdinalLogisticRegression
import numpy as np
import pandas as pd
from functools import wraps
import time

'''

CUDA Model Training 
---------------------------------------------------------------------------

Data attribution:

    Government of Canada, 2020, General Social Survey - Social Identity

    Access at: https://www150.statcan.gc.ca/n1/pub/45-25-0001/index-eng.htm


---------------------------------------------------------------------------

Multinomial Logistic (Partially Optimized):

    shape: (570810, 31)

    tol: 1e-08

    iter: 150

    Average fit time: 11.8263 seconds on RTX3060 12GB

---------------------------------------------------------------------------

Ordinal Logistic (Optimized):

    shape: (570810, 30)

    tol: 1e-08

    iter: 150

    Average fit time: 7.0645 seconds on RTX3060 12GB

---------------------------------------------------------------------------

'''


def main():

    print("\nProcessing dataset")

    df = (
        pd.read_csv("gss20.csv")
    )

    print("\nCleaning dataset")

    data = (
        clean(df)
    )

    multinomial = (
        train_multinomial(data)
    )

    print(multinomial)

    ordinal = (
        train_ordinal(data)
    )

    print(ordinal)


def timer(func):

    @wraps(func)
    def wrapper(*args, **kwargs):

        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        print(f"\nOperation: {func.__name__} resolved in {end - start:.4f} seconds.\n")

        return result
    return wrapper


@timer
def train_multinomial(df):

    mn_regression_1 = df[[
        'const', 'mob1_cat', 'age', 'age^2', 'female', 'married_bin', 'minority_bin',
        'edu_HS', 'edu_>HS', 'edu_Bachelors', 'edu_>Bachelors',
        'inc_second_quartile',
        'inc_third_quartile',
        'inc_fourth_quartile',
        'ord_hhsize_2persons',
        'ord_hhsize_3persons',
        'ord_hhsize_4+persons',
        'prov_NewfoundlandandLabrador', 'prov_Quebec', 'prov_Alberta', 'prov_BritishColumbia', 'prov_NewBrunswick',
        'prov_Manitoba', 'prov_NorthernCanada', 'prov_NovaScotia', 'prov_Saskatchewan', 'prov_PrinceEdwardIsland',
        'kol_Frenchonly', 'kol_EnglishandFrench', 'kol_NeitherEnglishnorFrench',
        'immig_Immigrants', 'immig_Non-permanentresidents'
    ]]

    mn_regression_1 = mn_regression_1.copy()
    mn_regression_1 = mn_regression_1.dropna()

    X = mn_regression_1.drop(columns=['mob1_cat'])
    y = mn_regression_1['mob1_cat']

    return MultinomialLogisticRegression().fit(X=X, y=y, alpha=0.05, cuda=True, cov_type="HC0")


@timer
def train_ordinal(df):

    ord_regression_1 = df[[
        'mob1_cat', 'age', 'age^2', 'married_bin', 'minority_bin', 'female',
        'edu_HS', 'edu_>HS', 'edu_Bachelors', 'edu_>Bachelors',
        'inc_second_quartile', 'inc_third_quartile', 'inc_fourth_quartile',
        'prov_NewfoundlandandLabrador', 'prov_Quebec', 'prov_Alberta', 'prov_BritishColumbia', 'prov_NewBrunswick',
        'prov_Manitoba', 'prov_NorthernCanada', 'prov_NovaScotia', 'prov_Saskatchewan', 'prov_PrinceEdwardIsland',
        'kol_Frenchonly', 'kol_EnglishandFrench', 'kol_NeitherEnglishnorFrench',
        'ord_hhsize_2persons', 'ord_hhsize_3persons', 'ord_hhsize_4+persons',
        'immig_Immigrants', 'immig_Non-permanentresidents',
    ]]

    ord_regression_1 = ord_regression_1.copy()
    ord_regression_1 = ord_regression_1.dropna()

    X = ord_regression_1.drop(columns=['mob1_cat'])
    y = ord_regression_1['mob1_cat']

    return OrdinalLogisticRegression().fit(X=X, y=y, alpha=0.05, max_iter=150, cuda=True, adj_cutpoints=False)


@timer
def clean(df):

    df['const'] = np.ones(len(df))

    REGRESSION_SUBSETS = df[['const', 'agegrp', 'Mob1', 'Mob5', 'Gender', 'marsth', 'hdgree', 'hhsize',
                             'pr', 'PR5', 'fol', 'kol', 'immstat', 'aboid', 'vismin', 'NOC21', 'naics', 'dtype', 'MrkInc']]

    REGRESSION_SUBSETS = REGRESSION_SUBSETS.copy()

    REGRESSION_SUBSETS['income_dirty'] = REGRESSION_SUBSETS['MrkInc'].replace(
        {99999999: np.nan, 88888888: np.nan})
    REGRESSION_SUBSETS['income_clean'] = REGRESSION_SUBSETS['income_dirty'].where(
        REGRESSION_SUBSETS['income_dirty'] > 0)
    REGRESSION_SUBSETS['log(income)'] = np.log(
        REGRESSION_SUBSETS['income_clean'])

    REGRESSION_SUBSETS['income_quartile'] = (
        pd.qcut(REGRESSION_SUBSETS['income_clean'], q=4, labels=[
                'first_quartile', 'second_quartile', 'third_quartile', 'fourth_quartile'])
    )

    categories = ['first_quartile', 'second_quartile',
                  'third_quartile', 'fourth_quartile',]
    for category in categories:
        col_name = f"inc_{category.replace(' ', '')}"

        REGRESSION_SUBSETS[col_name] = (
            REGRESSION_SUBSETS['income_quartile'] == category
        ).astype(float).where(REGRESSION_SUBSETS['income_quartile'].notna())

    REGRESSION_SUBSETS['age'] = REGRESSION_SUBSETS['agegrp'].map({
        'Not available': np.nan,
        '0 to 4 years': np.nan,
        '5 to 6 years': np.nan,
        '7 to 9 years': np.nan,
        '10 to 11 years': np.nan,
        '12 to 14 years': np.nan,
        '15 to 17 years': np.nan,
        '18 to 19 years': np.nan,
        '20 to 24 years': np.nan,
        '25 to 29 years': 1,
        '30 to 34 years': 2,
        '35 to 39 years': 3,
        '40 to 44 years': 4,
        '45 to 49 years': 5,
        '50 to 54 years': 6,
        '55 to 59 years': 7,
        '60 to 64 years': 8,
        '65 to 69 years': 9,
        '70 to 74 years': 10,
        '75 to 79 years': 11,
        '80 to 84 years': 12,
        '85 years and over': np.nan
    })

    REGRESSION_SUBSETS['age^2'] = REGRESSION_SUBSETS['age']**2

    REGRESSION_SUBSETS['mob1_bin'] = REGRESSION_SUBSETS['Mob1'].map({
        "Non-movers": 0,
        "Non-migrants": 0,
        'Different CSD, same census division': 0,
        'Different CD, same province': 1,
        'Interprovincial migrants': 1,
        'External migrants': np.nan,
        "Not available": np.nan,
        "Not applicable": np.nan
    })

    REGRESSION_SUBSETS['mob5_bin'] = REGRESSION_SUBSETS['Mob5'].map({
        "Non-movers": 0,
        "Non-migrants": 0,
        'Different CSD, same census division': 0,
        'Different CD, same province': 1,
        'Interprovincial migrants': 1,
        'External migrants': np.nan,
        "Not applicable": np.nan
    })

    REGRESSION_SUBSETS['mob1_cat'] = REGRESSION_SUBSETS['Mob1'].map({
        "Non-movers": 0,
        "Non-migrants": 0,
        'Different CSD, same census division': 1,
        'Different CD, same province': 2,
        'Interprovincial migrants': 3,
        'External migrants': np.nan,
        "Not available": np.nan,
        "Not applicable": np.nan
    })

    REGRESSION_SUBSETS['mob5_cat'] = REGRESSION_SUBSETS['Mob5'].map({
        "Non-movers": 0,
        "Non-migrants": 0,
        'Different CSD, same census division': 1,
        'Different CD, same province': 2,
        'Interprovincial migrants': 3,
        'External migrants': np.nan,
        "Not applicable": np.nan
    })

    REGRESSION_SUBSETS['female'] = REGRESSION_SUBSETS['Gender'].map({
        "Man+": 0,
        "Woman+": 1,
    })

    REGRESSION_SUBSETS['married_bin'] = REGRESSION_SUBSETS['marsth'].map({
        'Not available': np.nan,
        'Married': 1,
        'Never married (not living common law)': 0,
        'Divorced (not living common law)': 0,
        'Widowed (not living common law)': 0,
        'Living common law': 0,
        'Separated (not living common law)': 0,
    })

    REGRESSION_SUBSETS['educ_cat'] = REGRESSION_SUBSETS['hdgree'].map({
        'Not available': np.nan,
        'Not applicable': np.nan,
        'No certificate, diploma or degree': "<HS",
        'High (secondary) school diploma or equivalency certificate': "HS",
        'Program of 3 months to less than 1 year (College, CEGEP and other non-university certificates or diplomas)': ">HS",
        'Program of 1 to 2 years (College, CEGEP and other non-university certificates or diplomas)': ">HS",
        'Program of more than 2 years (College, CEGEP and other non-university certificates or diplomas)': ">HS",
        'University certificate or diploma below bachelor level': ">HS",
        ' Apprenticeship certificate': ">HS",
        'Non-apprenticeship trades certificate or diploma': ">HS",
        "Bachelor's degree": "Bachelors",
        'University certificate or diploma above bachelor level': ">Bachelors",
        'Degree in medicine, dentistry, veterinary medicine or optometry': ">Bachelors",
        "Master's degree": ">Bachelors",
        'Earned doctorate': ">Bachelors",
    })

    categories = ["<HS", "HS", ">HS", "Bachelors", ">Bachelors"]
    for category in categories:
        col_name = f"edu_{category.replace(' ', '')}"

        REGRESSION_SUBSETS[col_name] = (
            REGRESSION_SUBSETS['educ_cat'] == category
        ).astype(float).where(REGRESSION_SUBSETS['educ_cat'].notna())

    REGRESSION_SUBSETS["hhsize_clean"] = REGRESSION_SUBSETS["hhsize"].replace(
        {"Not available": np.nan})

    categories = ['1 person', '2 persons', '3 persons',
                  '4 persons', '5 persons', '6 persons', '7 persons or more']
    for category in categories:
        col_name = f"hhsize_{category.replace(' ', '')}"

        REGRESSION_SUBSETS[col_name] = (
            REGRESSION_SUBSETS['hhsize_clean'] == category
        ).astype(float).where(REGRESSION_SUBSETS['hhsize_clean'].notna())

    REGRESSION_SUBSETS["hhsize_ordinal"] = REGRESSION_SUBSETS["hhsize"].replace({
                                                                                "Not available": np.nan})

    REGRESSION_SUBSETS['hhsize_ordinal_clean'] = REGRESSION_SUBSETS['hhsize_ordinal'].map({
        "1 person": "1 person",
        "2 persons": "2 persons",
        '3 persons': "3 persons",
        '4 persons': "4+ persons",
        '5 persons': "4+ persons",
        '6 persons': "4+ persons",
        "7 persons or more": "4+ persons",

    })

    categories = ['1 person', '2 persons', '3 persons', '4+ persons',]
    for category in categories:
        col_name = f"ord_hhsize_{category.replace(' ', '')}"

        REGRESSION_SUBSETS[col_name] = (
            REGRESSION_SUBSETS['hhsize_ordinal_clean'] == category
        ).astype(float).where(REGRESSION_SUBSETS['hhsize_ordinal_clean'].notna())

    categories = ['Newfoundland and Labrador', 'Ontario', 'Quebec', 'Alberta', 'British Columbia',
                  'New Brunswick', 'Manitoba', 'Northern Canada', 'Nova Scotia', 'Saskatchewan', 'Prince Edward Island']
    for category in categories:
        col_name = f"prov_{category.replace(' ', '')}"

        REGRESSION_SUBSETS[col_name] = (
            REGRESSION_SUBSETS['pr'] == category
        ).astype(float).where(REGRESSION_SUBSETS['pr'].notna())

    REGRESSION_SUBSETS["kol_clean"] = REGRESSION_SUBSETS["kol"].replace(
        {"Not available": np.nan})

    categories = ['English only', 'French only',
                  'English and French', 'Neither English nor French',]
    for category in categories:
        col_name = f"kol_{category.replace(' ', '')}"

        REGRESSION_SUBSETS[col_name] = (
            REGRESSION_SUBSETS['kol_clean'] == category
        ).astype(float).where(REGRESSION_SUBSETS['kol_clean'].notna())

    REGRESSION_SUBSETS["immstat_clean"] = REGRESSION_SUBSETS["immstat"].replace({
                                                                                "Not available": np.nan})

    categories = ['Immigrants', 'Non-immigrants', 'Non-permanent residents']
    for category in categories:
        col_name = f"immig_{category.replace(' ', '')}"

        REGRESSION_SUBSETS[col_name] = (
            REGRESSION_SUBSETS['immstat_clean'] == category
        ).astype(float).where(REGRESSION_SUBSETS['immstat_clean'].notna())

    REGRESSION_SUBSETS['indigenous_bin'] = REGRESSION_SUBSETS['aboid'].map({
        'Non-Indigenous identity': 0,
        'First Nations (North American Indian)': 1,
        'Métis': 1,
        'Multiple Indigenous responses': 1,
        'Indigenous responses not included elsewhere': 1,
        'Inuk\xa0(Inuit)': 1

    })

    REGRESSION_SUBSETS['minority_bin'] = REGRESSION_SUBSETS['vismin'].map({
        'Not available': np.nan,
        'South Asian': 1,
        'Not a visible minority': 0,
        'Latin American': 1,
        'Black': 1,
        'Chinese': 1,
        'Filipino': 1,
        'Visible minority, n.i.e.': 1,
        'West Asian': 1,
        'Arab': 1,
        'Korean': 1,
        'Southeast Asian': 1,
        'Multiple visible minorities': 1,
        'Japanese': 1

    })

    return REGRESSION_SUBSETS


if __name__ == "__main__":

    main()
