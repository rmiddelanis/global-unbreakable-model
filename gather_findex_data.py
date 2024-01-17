import pandas as pd


def gather_findex_data(findex_data_path, axfin_outpath=None):
    """
    This function gathers and processes data from the FINDEX dataset.

    Parameters:
    findex_data_path (str): The path to the 2021 FINDEX data file. File can be obtained from https://doi.org/10.48529/JQ97-AJ70
    axfin_outpath (str, optional): The path to the output file. If provided, the result will be saved to this file.

    Returns:
    None
    """

    # Load the data from the provided path, selecting only the necessary columns
    findex_data = pd.read_csv(findex_data_path, encoding='latin1')[['economy', 'economycode', 'inc_q', 'fin17a', 'wgt']]

    # Rename the columns for better readability
    findex_data.rename({'economy': 'country', 'economycode': 'iso3', 'inc_q': 'income_quintile', 'fin17a': 'axfin'},
                       axis=1, inplace=True)

    # Set the index of the DataFrame to be a combination of country, iso3 code, and income quintile
    findex_data.set_index(['country', 'iso3', 'income_quintile'], inplace=True)

    # Replace the values in the 'axfin' column based on the FINDEX 2021 codebook:
    # 1 if the respondent saved or set aside any money (value=1)
    # 0 if no (=2), don’t know (=3), or refused to answer (=4)
    findex_data.axfin.replace({1: 1, 2: 0, 3: 0, 4: 0}, inplace=True)

    # Calculate the result by multiplying all columns (prod), grouping by country, iso3 code, and income quintile,
    # summing the groups, and then dividing by the sum of the 'wgt' column for each group
    res = (findex_data.prod(axis=1).groupby(['country', 'iso3', 'income_quintile']).sum()
           / findex_data.wgt.groupby(['country', 'iso3', 'income_quintile']).sum())
    res.name = 'axfin'

    # If an output path is provided, save the result to a CSV file at that path
    if axfin_outpath is not None:
        res.to_csv(axfin_outpath)
