import os
import pandas as pd
import numpy as np


def process_peb_data(root_dir="./", exposure_data_path="inputs/PEB/exposure bias.dta",
                     poverty_data_path="inputs/PEB/poverty_data/", outfile=None):
    exposure_data_path = os.path.join(root_dir, exposure_data_path)
    poverty_data_path = os.path.join(root_dir, poverty_data_path)

    # load exposure data
    exposure = pd.read_stata(exposure_data_path)
    exposure.rename({'exp': 'hazard', 'code': 'iso3', 'line': 'pov_line', 'nvul': 'pop_a'}, axis=1, inplace=True)
    exposure = exposure[['iso3', 'hazard', 'pop_a', 'pov_line']]
    exposure = exposure.astype({'hazard': str, 'iso3': str, 'pov_line': float, 'pop_a': float})
    exposure.hazard = exposure.hazard.apply(lambda x: str.capitalize(x.replace('exp_', '')))
    exposure.pop_a *= 1e6
    exposure.pov_line = exposure.pov_line.fillna(np.inf) / 100  # use np.inf for full population; convert to percentage
    exposure = exposure.set_index(['iso3', 'hazard', 'pov_line']).squeeze()

    # split cyclone into storm surge and wind
    exposure = exposure.unstack('hazard')
    exposure['Storm surge'] = exposure.Cyclone
    exposure['Wind'] = exposure.Cyclone
    exposure.drop('Cyclone', axis=1, inplace=True)
    exposure = exposure.stack('hazard')

    # add poverty line 0
    exposure = exposure.unstack('pov_line')
    exposure[0.] = 0
    exposure = exposure.stack('pov_line').sort_index().rename('pop_a')

    # load poverty data
    pov_files = [f for f in os.listdir(poverty_data_path) if f.endswith('.csv')]
    pov_data = pd.concat([pd.read_csv(poverty_data_path + f) for f in pov_files], axis=0)
    pov_data = pov_data[['country_code', 'reporting_year', 'poverty_line', 'reporting_pop', 'headcount']]
    pov_data = pov_data.rename({'country_code': 'iso3', 'reporting_year': 'year', 'poverty_line': 'pov_line',
                                'reporting_pop': 'pop', 'headcount': 'pov_headcount'}, axis=1).reset_index(drop=True)
    pov_data = pov_data.astype({'iso3': str, 'year': int, 'pov_line': float, 'pop': int, 'pov_headcount': float})
    pov_data = pov_data.loc[pov_data.groupby(['iso3', 'pov_line']).year.idxmax()].reset_index(drop=True)
    pov_data.drop('year', axis=1, inplace=True)

    # keep population data separately
    pop_data = pov_data[['iso3', 'pop']].drop_duplicates().set_index('iso3').squeeze()

    # keep poverty headcount data separately
    pov_head = pov_data.set_index(['iso3', 'pov_line']).pov_headcount.unstack('pov_line')

    # interpolate missing poverty lines
    missing_pov_lines = np.setdiff1d(exposure.index.get_level_values('pov_line').unique(), pov_head.columns)
    new_cols = pd.DataFrame(
        index=pd.Index(missing_pov_lines, name='pov_line'),
        data={i: np.interp(missing_pov_lines, pov_head.columns, pov_head.loc[i].interpolate().values) for i in pov_head.index.unique()},
        columns=pd.Index(pov_head.index.unique(), name='iso3')
    ).transpose()

    # set headcount for poverty lines 0 to 0 and np.inf (full population) to 1
    new_cols[0] = 0
    new_cols[np.inf] = 1

    pov_head = pd.concat([pov_head, new_cols], axis=1).stack().sort_index().rename('pov_headcount')

    # compute population per poverty line
    pov_data = pd.merge((pop_data * pov_head).rename('pop'), pov_head, left_index=True, right_index=True)

    # merge exposure and poverty data
    exposure = pd.merge(exposure, pov_data, left_index=True, right_index=True, how='left')

    # define population slices, i.e. the population between two poverty lines
    exposure.reset_index(inplace=True)
    exposure['pop_slice'] = exposure.pov_line.replace({p: i for i, p in enumerate(sorted(exposure.pov_line.unique()))})
    exposure = exposure.set_index(['iso3', 'hazard', 'pop_slice']).sort_index()

    # drop hazards without any exposure
    drop_idx = exposure[(exposure.pov_line == np.inf) & (exposure.pop_a == 0)].droplevel('pop_slice').index
    exposure.drop(drop_idx, inplace=True)

    # keep only the average exposure for the total country
    exposure_avg = exposure.loc[exposure.pov_line == np.inf].droplevel('pop_slice')
    exposure_avg = exposure_avg.pop_a / exposure_avg['pop']
    exposure_avg[exposure_avg > 1] = 1  # TODO: use more recent population data for some countries w old values

    # calculate row-wise difference to get values per population slice
    exposure = exposure.drop('pov_line', axis=1).sort_index().groupby(['iso3', 'hazard']).diff().dropna()

    # TODO: some categories have pov_headcount == 0 but pop_a > 0
    # drop data where pov_headcount == 0 (these don't contribute to any income quintile)
    exposure = exposure[exposure.pov_headcount > 0].copy()

    # compute cumulative headcount (needed for calculation of per-quintile exposure)
    exposure['cum_headcount'] = exposure.groupby(['iso3', 'hazard']).pov_headcount.cumsum()

    # compute relative exposure
    exposure['f_a'] = exposure.pop_a / exposure['pop']
    exposure.loc[exposure.f_a > 1, 'f_a'] = 1  # TODO: see row above
    exposure.loc[exposure.f_a < 0, 'f_a'] = np.nan  # some entries have f_a < 0 (only very small headcounts)
    exposure['f_a'] = exposure['f_a'].fillna(exposure_avg)  # exposure for these entries with country avg

    # compute exposure bias
    exposure['exposure_bias'] = exposure.f_a / exposure_avg

    # calculate exposure bias per quintile
    quintiles = ['q1', 'q2', 'q3', 'q4', 'q5']
    q_head = .2
    exp_bias_q = pd.DataFrame(
        index=exposure.droplevel('pop_slice').index.unique(),
        columns=quintiles,
    ).fillna(0).stack().replace(0, np.nan).rename('exposure_bias')
    exp_bias_q.index.names = ['iso3', 'hazard', 'income_cat']
    for quintile_idx, quintile in enumerate(quintiles):
        cum_head_q = q_head * (quintile_idx + 1)
        weights = (
                (exposure.pov_headcount - (exposure.cum_headcount - cum_head_q).clip(lower=0)).clip(lower=0) -
                (exposure.pov_headcount - (exposure.cum_headcount - (cum_head_q - .2)).clip(lower=0)).clip(lower=0)
        ).clip(lower=0)
        exp_bias_q_ = (weights * exposure.exposure_bias).groupby(['iso3', 'hazard']).sum() / weights.groupby(['iso3', 'hazard']).sum()
        exp_bias_q_ = pd.concat([exp_bias_q_], keys=[quintile], names=['income_cat']).swaplevel(0, 2).swaplevel(0, 1).sort_index()
        exp_bias_q.loc[exp_bias_q_.index] = exp_bias_q_.values

    # store exposure bias per quintile
    if outfile is not None:
        exp_bias_q.to_csv(outfile)
        print(f"Exposure bias per quintile stored in {outfile}")

    return exp_bias_q


if __name__ == "__main__":
    process_peb_data(
        root_dir=os.getcwd(),
        exposure_data_path="inputs/PEB/exposure bias.dta",
        poverty_data_path="inputs/PEB/poverty_data/",
        outfile="./inputs/PEB/exposure_bias_per_quintile.csv"
    )
