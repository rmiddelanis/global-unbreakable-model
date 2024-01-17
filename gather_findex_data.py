import numpy as np
import pandas as pd

from lib_gather_data import get_country_name_dicts


def gather_findex_data(findex_data_paths: dict, root_dir: str, axfin_outpath=None):
    """
    This function gathers and processes data from the FINDEX datasets.

    Data can be obtained from
    https://doi.org/10.48529/JQ97-AJ70 (FINDEX 2021)
    https://doi.org/10.48529/d3cf-fj47 (FINDEX 2017)
    https://doi.org/10.48529/9j25-hr41 (FINDEX 2014)
    https://doi.org/10.48529/wnvd-y919 (FINDEX 2011)

    Parameters:
    findex_data_path (dict): Values contain the paths to the FINDEX data file with the year as dict key.
    axfin_outpath (str, optional): The path to the output file. If provided, the result will be saved to this file.

    Returns:
    None
    """
    any_to_wb, iso3_to_wb, iso2_iso3 = get_country_name_dicts(root_dir)

    findex_datasets = []
    for year, findex_data_path in findex_data_paths.items():
        # Load the data from the provided path, selecting only the necessary columns
        findex_data = pd.read_csv(findex_data_path, encoding='latin1')

        # Rename the columns for better readability
        if year in [2017, 2021]:
            varname = 'fin17a'
        elif year == 2014:
            varname = 'q18a'
        elif year == 2011:
            varname = 'q13a'
            findex_data.rename({'ecnmycode': 'economycode'}, axis=1, inplace=True)
        else:
            raise ValueError('Unknown FINDEX year: {}'.format(np.unique(findex_data.year).item()))

        if 'year' not in findex_data.columns:
            findex_data['year'] = year

        findex_data.rename({'economycode': 'iso3', 'inc_q': 'income_cat', varname: 'axfin'},
                           axis=1, inplace=True)
        findex_data = findex_data[['iso3', 'income_cat', 'axfin', 'wgt', 'year']].dropna()

        # some country names have changed between FINDEX rounds; therefore, first use iso3, then convert to WB name
        findex_data['country'] = findex_data.iso3.replace(iso3_to_wb)
        findex_data.drop('iso3', axis=1, inplace=True)

        findex_data = findex_data.astype({'country': 'str', 'income_cat': 'int', 'axfin': 'float',
                                          'wgt': 'float', 'year': 'int'})
        findex_data.income_cat = findex_data.income_cat.apply(lambda x: 'q{}'.format(x))

        # Set the index of the DataFrame to be a combination of country, and income quintile
        findex_data.set_index(['country', 'year', 'income_cat'], inplace=True)

        # Replace the values in the 'axfin' column based on the FINDEX 2021 codebook:
        # 1 if the respondent saved or set aside any money (value=1)
        # 0 if no (=2), don’t know (=3), or refused to answer (=4)
        findex_data.axfin.replace({1: 1, 2: 0, 3: 0, 4: 0}, inplace=True)

        findex_datasets.append(findex_data)

    # Concatenate the datasets
    findex_data = pd.concat(findex_datasets)

    # Calculate the result by multiplying all columns (prod), grouping by country, and income quintile,
    # summing the groups, and then dividing by the sum of the 'wgt' column for each group
    findex_data = (findex_data.prod(axis=1).groupby(['country', 'year', 'income_cat']).sum()
                   / findex_data.wgt.groupby(['country', 'year', 'income_cat']).sum())
    findex_data.name = 'axfin'

    # If an output path is provided, save the result to a CSV file at that path
    if axfin_outpath is not None:
        findex_data.to_csv(axfin_outpath)

    return findex_data
