import pandas as pd

from . import constants


def read_raw(path: str):
    df = pd.read_html(path)[0]
    df.columns = constants.COLUMNS
    return df


def clean_data(raw_data: pd.DataFrame,
               drop_low_outlier: bool = True,
               drop_high_outlier: bool = True,
               drop_unnecessary: bool = True
               ) -> pd.DataFrame:
    df = raw_data.copy()

    cols_to_lower = ['role', 'company', 'city', 'compensation']
    for col in cols_to_lower:
        df[col] = df[col].str.lower()

    df.salary = df.apply(normalize_salary_to_million_idr_monthly, axis=1)
    df.company = df.apply(clean_company, axis=1)

    # drop duplicated
    categorical_cols = ['role', 'company', 'years_of_exp', 'city', 'country',
                        'gender']
    df = df[~df[categorical_cols].duplicated()]
    
    # drop outliers
    q3, q1 = df.salary.quantile(0.75), df.salary.quantile(0.25)
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    if drop_low_outlier:
        df = df[lower <= df.salary]
    if drop_high_outlier:
        df = df[df.salary <= upper]

    if drop_unnecessary:
        cols_to_drop = ['currency', 'mode', 'period']
        df = df.drop(cols_to_drop, axis=1)
        
    df = df.reset_index(drop=True)
    return df


def normalize_salary_to_million_idr_monthly(row: pd.Series) -> int:
    salary = row.salary

    if row.currency != 'IDR':
        salary *= constants.CURRENCY_RATE[row.currency]
    elif salary <= 1000:
        salary *= 1_000_000
    elif salary <= 100_000:
        salary *= 1000

    if row.period == 'Annual':
        salary = salary // 12

    return salary / 1_000_000


def clean_company(row: pd.Series) -> str:
    company = row.company
    return 'other' if company == 'purchase to unlock 👆' else company


def clean_country(row: pd.Series) -> str:
    country = row.country
    return country if country == 'ID' else 'Outside ID'


def clean_city(row):
    city = row.city

    for correct_city, mispelled_cities in constants.FIX_CITIES.items():
        for mispelled_city in mispelled_cities:
            if mispelled_city == city:
                city = correct_city
                break

    for other_city in constants.CITIES:
        if other_city in city:
            city = other_city
            break

    return city


def clean_role(row):
    role = row.role

    for mispelled_role in constants.FIX_ROLE.keys():
        if mispelled_role in role:
            rep = constants.FIX_ROLE[mispelled_role]
            role = role.replace(mispelled_role, rep)

    return role


def limit_years_of_exp(row, limit=10):
    yoe = row.years_of_exp
    if yoe >= limit:
        return str(limit) + '+'
    return str(yoe)
