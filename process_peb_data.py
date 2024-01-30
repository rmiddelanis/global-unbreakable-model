import pandas as pd
import numpy as np
from wb_api_wrapper import get_wb_series
from lib_gather_data import df_to_iso3


def process_peb_data(exposure_data_path, wb_macro_path, outfile=None):
    # load wb data
    wb_macro = pd.read_csv(wb_macro_path, encoding='latin1').set_index('iso3')

    # load exposure data
    exposure = pd.read_stata(exposure_data_path)
    exposure.rename({'exp': 'hazard', 'code': 'iso3', 'line': 'pov_line', 'nvul': 'pop_a'}, axis=1, inplace=True)
    exposure.hazard = exposure.hazard.apply(lambda x: x.replace('exp_', ''))
    exposure = exposure[['iso3', 'hazard', 'pop_a', 'pov_line']]
    exposure.pop_a *= 1e6
    exposure.pov_line = exposure.pov_line.fillna(np.inf)
    exposure = exposure.set_index(['iso3', 'hazard', 'pov_line']).squeeze()
    exposure = exposure.unstack('pov_line')
    exposure[0.] = 0
    exposure = exposure.stack().rename('pop_a').reset_index()

    # load wb headcount data
    pov_head_215 = get_wb_series('SI.POV.DDAY', 215.0).dropna() / 100
    pov_head_365 = get_wb_series('SI.POV.LMIC', 365.0).dropna() / 100
    pov_head_685 = get_wb_series('SI.POV.UMIC', 685.0).dropna() / 100
    pov_head = pd.concat([pov_head_215, pov_head_365, pov_head_685], axis=1)
    pov_head = pov_head.stack().rename('pov_headcount')
    pov_head.index.names = ['country', 'year', 'pov_line']
    pov_head = pov_head.reset_index()
    pov_head = pov_head.loc[pov_head.groupby(['country', 'pov_line']).year.idxmax()]
    pov_head = pov_head.set_index(['country', 'pov_line']).pov_headcount.unstack('pov_line')
    pov_head[0.] = 0
    pov_head = pov_head.stack().rename('pov_headcount')
    pov_head = df_to_iso3(pov_head.reset_index(), 'country')
    pov_head = pov_head.dropna(subset=['iso3']).reset_index(drop=True)
    pov_head.drop('country', axis=1, inplace=True)
    pov_head.pov_line = pov_head.pov_line.astype(object)

    # merge exposure and headcount data
    exposure = pd.merge(exposure, pov_head, on=['iso3', 'pov_line'], how='outer')

    # pov_line == np.inf is the total country exposure --> headcount is 1
    exposure.loc[exposure.pov_line == np.inf, 'pov_headcount'] = 1

    # for some poverty lines, no headcount data is available. drop these
    exposure.dropna(subset=['pov_headcount'], inplace=True)

    # define population slices, i.e. the population between two poverty lines
    exposure['pop_slice'] = exposure.pov_line.replace({p: i for i, p in enumerate(sorted(exposure.pov_line.unique()))})
    exposure = exposure.set_index(['iso3', 'hazard', 'pop_slice']).sort_index()

    # drop hazards without any exposure
    drop_idx = exposure[(exposure.pov_line == np.inf) & (exposure.pop_a == 0)].droplevel('pop_slice').index
    exposure.drop(drop_idx, inplace=True)

    # keep only the average exposure for the total country
    exposure_data_avg = exposure.loc[exposure.pov_line == np.inf].droplevel('pop_slice')
    exposure_data_avg = exposure_data_avg.pop_a / wb_macro['pop']
    exposure_data_avg[exposure_data_avg > 1] = 1  # TODO

    # calculate row-wise difference to get values per population slice
    exposure = exposure.groupby(['iso3', 'hazard']).diff().dropna()

    # TODO: some categories have pov_headcount == 0 but pop_a > 0
    # drop data where pov_headcount == 0 (these don't contribute to any income quintile)
    exposure = exposure[exposure.pov_headcount > 0]

    # compute relative exposure
    exposure['f_a'] = exposure.pop_a / (wb_macro['pop'] * exposure.pov_headcount)
    exposure.dropna(subset=['f_a'], inplace=True)

    # compute exposure bias
    exposure['exposure_bias'] = exposure.f_a / exposure_data_avg

    # compute cumulative headcount (needed for calculation of per-quintile exposure)
    exposure['cum_headcount'] = exposure.groupby(['iso3', 'hazard']).pov_headcount.cumsum()

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
