import os

import numpy as np
import pandas as pd

from lib_gather_data import get_country_name_dicts


def gather_findex_data(findex_data_paths_: dict, question_ids_: dict, root_dir_: str, varname_: str):
    """
    This function gathers and processes data from the FINDEX datasets.

    Data can be obtained from
    https://doi.org/10.48529/JQ97-AJ70 (FINDEX 2021)
    https://doi.org/10.48529/d3cf-fj47 (FINDEX 2017)
    https://doi.org/10.48529/9j25-hr41 (FINDEX 2014)
    https://doi.org/10.48529/wnvd-y919 (FINDEX 2011)

    Parameters:
    findex_data_path (dict): Values contain the paths to the FINDEX data file with the year as dict key.
    outpath (str, optional): The path to the output file. If provided, the result will be saved to this file.

    Returns:
    None
    """
    any_to_wb, iso3_to_wb, iso2_iso3 = get_country_name_dicts(root_dir_)

    findex_datasets = []
    for year, findex_data_path in findex_data_paths_.items():
        # Load the data from the provided path, selecting only the necessary columns
        findex_data = pd.read_csv(findex_data_path, encoding='latin1')

        # Select the correct question for each FINDEX year
        if year in question_ids_:
            question = question_ids_[year]
        else:
            raise ValueError('Unknown FINDEX year: {}'.format(np.unique(findex_data.year).item()))

        if question is not None:
            if year == 2011:
                findex_data.rename({'ecnmycode': 'economycode'}, axis=1, inplace=True)

            if 'year' not in findex_data.columns:
                findex_data['year'] = year

            findex_data.rename({'economycode': 'iso3', 'inc_q': 'income_cat', question: varname_},
                               axis=1, inplace=True)
            findex_data[varname_] = findex_data[varname_].fillna(0)
            findex_data = findex_data[['iso3', 'income_cat', varname_, 'wgt', 'year']].dropna()

            # some country names have changed between FINDEX rounds; therefore, first use iso3, then convert to WB name
            findex_data['country'] = findex_data.iso3.replace(iso3_to_wb)
            findex_data.drop('iso3', axis=1, inplace=True)

            findex_data = findex_data.astype({'country': 'str', 'income_cat': 'int', varname_: 'float',
                                              'wgt': 'float', 'year': 'int'})
            findex_data.income_cat = findex_data.income_cat.apply(lambda x: 'q{}'.format(x))

            # Set the index of the DataFrame to be a combination of country, and income quintile
            findex_data.set_index(['country', 'year', 'income_cat'], inplace=True)

            findex_datasets.append(findex_data)
        else:
            print(f'No question ID provided for FINDEX year {year}. Skipping.')

    # Concatenate the datasets
    findex_data = pd.concat(findex_datasets)

    return findex_data


def gather_axfin_data(root_dir_, findex_data_paths_, write_output_=False):
    # TODO: for the years 2017 and 2021, the variable 'fin17a' can be used to exactly reproduce the World Bank
    #  indicators fin17a.t.d.7 and fin17a.t.d.8 ('The percentage of respondents who report saving or setting aside
    #  any money at a bank or another type of financial institution in the past year, richest 60% (% ages 15+)').
    #  However, these indicators include additional data for the years 2014 and 2011, which do not exactly match
    #  the FINDEX data for questions q18a and q13a, respectively. Yet, the differences are small for those countries
    #  without data for 2017 and 2021. Check later!

    # fin17a (2021): "Saved using an account at a financial institution"
    # fin17a (2017): "In the PAST 12 MONTHS, have you, personally, saved or set aside any money by using an
    #                   account at a bank or another type of formal financial institution (This can include
    #                   using another person’s account)?"
    # q18a (2014): "In the PAST 12 MONTHS, have you, personally, saved or set aside any money by using an
    #               account at a bank or another type of formal financial institution?"
    # q13a (2011): "In the past 12 months, have you saved or set aside money by Using an account at a bank,
    #               credit union (or another financial institution, where applicable – for example, cooperatives
    #               in Latin America), or microfinance institution"
    question_ids = {2021: 'fin17a', 2017: 'fin17a', 2014: 'q18a', 2011: 'q13a'}

    # Gather the data from the FINDEX datasets
    findex_data = gather_findex_data(
        findex_data_paths_=findex_data_paths_,
        question_ids_=question_ids,
        root_dir_=root_dir_,
        varname_='axfin',
    )

    # Replace the values in the 'axfin' column based on the FINDEX codebook
    # 1 if the respondent saved or set aside any money (value=1)
    # 0 if no (=2), don’t know (=3), or refused to answer (=4)
    findex_data.axfin = findex_data.axfin.replace({1: 1, 2: 0, 3: 0, 4: 0})

    # Calculate the result by multiplying all columns (prod), grouping by country, year, and income quintile,
    # summing the groups, and then dividing by the sum of the 'wgt' column for each group
    axfin_data = (findex_data.prod(axis=1).groupby(['country', 'year', 'income_cat']).sum()
                  / findex_data.wgt.groupby(['country', 'year', 'income_cat']).sum()).rename('axfin')

    # If an output path is provided, save the result to a CSV file at that path
    if write_output_:
        axfin_data.to_csv(os.path.join(root_dir_, 'inputs', 'FINDEX', 'findex_axfin.csv'))

    return axfin_data


def gather_savings_data(root_dir_, findex_data_paths_, write_output_=False):
    question_ids = {2021: 'fin24', 2017: 'fin25', 2014: 'q25', 2011: None}

    # Gather the data from the FINDEX datasets
    findex_data = gather_findex_data(
        findex_data_paths_=findex_data_paths_,
        question_ids_=question_ids,
        root_dir_=root_dir_,
        varname_='savings',
    )
    # Replace the values in the 'savings' column based on the FINDEX codebooks:
    # 2021:
    # 1 to 6: Savings / Family, relatives, or friends / Money from working / Borrowing / Selling assets / Other
    # 7 to 9: I could not come up with the money / don’t know / refused to answer
    # 2017:
    # 1 to 6: Savings / Family, relatives, or friends / Money from working / Borrowing / Selling assets / Other
    # 7 to 8: don’t know / refused to answer
    # 2014:
    # 1 to 6: Savings / Family, relatives, friends / Work or employer loan / Borrowing / informal private lender or
    # pawn house / Other
    # 7 to 8: don’t know / refused to answer
    findex_data.savings = findex_data.savings.replace({7: 0, 8: 0, 9: 0})

    # Calculate the result by multiplying all columns (prod), grouping by country, year, and income quintile,
    # summing the groups, and then dividing by the sum of the 'wgt' column for each group
    savings_data = findex_data.groupby(['country', 'year', 'income_cat', 'savings']).wgt.sum() / findex_data.groupby(
        ['country', 'year', 'income_cat']).wgt.sum()

    savings_data = savings_data.unstack('savings').fillna(0)
    savings_data = savings_data.rename(columns={0: 'NA/couldnt', 1: 'savings', 2: 'family_friends', 3: 'work_loan',
                                                4: 'borrowing', 5: 'selling_assets', 6: 'other'})

    if write_output_:
        savings_data.to_csv(os.path.join(root_dir_, 'inputs', 'FINDEX', 'findex_savings.csv'))

    return savings_data


if __name__ == "__main__":
    root_dir = os.getcwd()

    findex_data_paths = {
        2021: os.path.join(root_dir, 'inputs', 'FINDEX', 'WLD_2021_FINDEX_v03_M.csv'),
        2017: os.path.join(root_dir, 'inputs', 'FINDEX', 'WLD_2017_FINDEX_v02_M.csv'),
        2014: os.path.join(root_dir, 'inputs', 'FINDEX', 'WLD_2014_FINDEX_v01_M.csv'),
        2011: os.path.join(root_dir, 'inputs', 'FINDEX', 'WLD_2011_FINDEX_v02_M.csv'),
    }

    gather_axfin_data(root_dir, findex_data_paths, write_output_=True)
    gather_savings_data(root_dir, findex_data_paths, write_output_=True)







































