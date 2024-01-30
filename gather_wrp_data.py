import numpy as np

from lib_gather_data import df_to_iso3
from pandas_helper import load_input_data


def gather_wrp_data(wrp_data_path="WRP/lrf_wrp_2021_full_data.csv.zip", root_dir='.', outfile=None):
    # early warning questions (see lrf_wrp_2021_full_data_dictionary.xlsx):
    # Q19A	Received Warning About Disaster From Internet/Social Media
    # Q19B	Received Warning About Disaster From Local Government or Police
    # Q19C	Received Warning About Disaster From Radio, TV, or Newspapers
    # Q19D	Received Warning About Disaster From Local Community Organization
    early_warning_cols = ['Q19A', 'Q19B', 'Q19C', 'Q19D']

    # read data
    wrp_data = load_input_data(root_dir, wrp_data_path)
    wrp_data = wrp_data[['Country', 'WGT', 'Year'] + early_warning_cols]
    wrp_data.set_index(['Country', 'Year'], inplace=True)

    # drop rows with no early warning data
    wrp_data = wrp_data.replace(' ', np.nan).dropna(how='all', subset=early_warning_cols)

    # 97 = Does Not Apply
    # 98 = Don't Know
    # 99 = Refused
    # consider all of the above as 'no'
    # 1 = Yes
    # 2 = No (set to 0)
    wrp_data[early_warning_cols] = wrp_data[early_warning_cols].astype(int).replace({97: 0, 98: 0, 99: 0, 2: 0})

    wrp_data['ew'] = wrp_data[early_warning_cols].sum(axis=1).astype(bool).astype(int)
    early_warning_cols.append('ew')

    ew_shares = wrp_data[early_warning_cols].mul(wrp_data.WGT, axis=0).groupby(['Country', 'Year']).sum().div(
        wrp_data.WGT.groupby(['Country', 'Year']).sum(), axis=0)
    ew_shares.reset_index(inplace=True)
    ew_shares = ew_shares.loc[ew_shares.groupby('Country').Year.idxmax()].drop('Year', axis=1).set_index('Country')

    if outfile is not None:
        ew_shares.ew.to_csv(outfile)

    return ew_shares
