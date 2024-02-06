# This script provides data input for the resilience indicator multihazard model.
# The script was developed by Adrien Vogt-Schilb and improved by Jinqiang Chen and Robin Middelanis.
import argparse

import tqdm
from scipy.interpolate import NearestNDInterpolator

from gather_gir_data import load_gir_hazard_losses
from lib_gather_data import *
from apply_policy import *
import pandas as pd
from lib_gather_data import get_country_name_dicts


# TODO: Note: compute_recovery_duration does not use the adjusted vulnerability. However, adjusted vulnerability is
#  hazard specific, so it cannot be used for computing recovery duration.
# TODO: discuss standard parameter values
def get_recovery_duration(avg_prod_k_, vulnerability_, discount_rate_, consump_util_=1.5, force_recompute=True,
                          lambda_increment_=.001, years_to_recover_=20, max_duration_=10, store_output=True):
    if not force_recompute and os.path.exists(os.path.join(root_dir, 'intermediate', 'recovery_rates.csv')):
        print("Loading recovery rates from file.")
        df = pd.read_csv(os.path.join(root_dir, 'intermediate', 'recovery_rates.csv'), index_col=['iso3', 'income_cat'])
    else:
        # vulnerability__ = vulnerability_.rename({'v_poor': 'poor', 'v_rich': 'nonpoor'}, axis=1)[['poor', 'nonpoor']]
        # vulnerability__ = vulnerability__.stack().reset_index().rename({'level_1': 'income_cat', 0: 'v'}, axis=1)
        # df = pd.merge(vulnerability__, avg_prod_k_, on='iso3', how='inner')
        # df.set_index(['iso3', 'income_cat'], inplace=True)
        df = pd.merge(vulnerability_, avg_prod_k_, left_index=True, right_index=True, how='inner')
        for row in tqdm.tqdm(df.itertuples(), total=len(df), desc="Computing recovery duration"):
            df.loc[row.Index, 'recovery_rate'] = integrate_and_find_recovery_rate(
                v=row.v, consump_util=consump_util_, discount_rate=discount_rate_, average_productivity=row.avg_prod_k,
                lambda_increment=lambda_increment_, years_to_recover=years_to_recover_
            )
        df_ = df.loc[df['recovery_rate'] < max_duration_]
        interp = NearestNDInterpolator(list(zip(df_.v, df_.avg_prod_k)), df_['recovery_rate'])
        df_error = df.loc[df['recovery_rate'] >= max_duration_]
        df_error['recovery_rate'] = interp(df_error.v, df_error.avg_prod_k)
        df = pd.concat((df_, df_error))

        # calculate years needed to recover to 95% of the initial welfare
        df['recovery_time'] = np.log(1 / 0.05) / df['recovery_rate']

        # recovery years are very high for very low vulnerability, so cap to 20 years for now
        df.loc[df['recovery_time'] > years_to_recover_, 'recovery_time'] = years_to_recover_
        df['recovery_rate'] = np.log(1 / 0.05) / df['recovery_time']  # adjust rates for capped recovery times
        if store_output:
            df[['recovery_rate', 'recovery_time']].to_csv(os.path.join(root_dir, 'intermediate', 'recovery_rates.csv'))
    return df[['recovery_rate', 'recovery_time']]


def get_cat_info_and_tau_tax(wb_data_cat_info_, wb_data_macro_, avg_prod_k_, n_quantiles_, axfin_impact_):
    cat_info_ = wb_data_cat_info_.copy(deep=True)

    # TODO: diversified_share is again recomputed in recompute_after_policy_change()
    # TODO: new variable 'diversified_share' instead of changing 'social' --> need to adjust in the rest of the model
    # from the Paper: "We assume that the fraction of income that is diversified increases by 10% for people who have
    # bank accounts
    cat_info_['diversified_share'] = cat_info_.social + cat_info_.axfin * axfin_impact_

    cat_info_['n'] = 1 / n_quantiles_
    cat_info_['c'] = cat_info_.income_share / cat_info_.n * wb_data_macro_.gdp_pc_pp

    # compute tau tax and gamma_sp from social
    tau_tax_, cat_info_["gamma_SP"] = social_to_tx_and_gsp(econ_scope, cat_info_)

    # compute capital per income category
    cat_info_["k"] = (1 - cat_info_.diversified_share) * cat_info_.c / ((1 - tau_tax_) * avg_prod_k_)

    cat_info_ = cat_info_.loc[cat_info_.drop('n', axis=1).dropna(how='all').index]
    return cat_info_, tau_tax_


# TODO: discuss protection approach with Bramka
# compare to gather_data_old.py: before, protection was simply set to rp=1, since no_protection==True
def load_protection(index_, protection_data="FLOPROS", min_rp=1, hazard_types="Flood+Storm surge",
                    flopros_protection_file="FLOPROS/FLOPROS_protection_processed/flopros_protection_processed.csv",
                    protection_level_assumptions_file="WB_country_classification/protection_level_assumptions.csv",
                    income_groups_file="WB_country_classification/country_classification.xlsx",
                    income_groups_file_historical="WB_country_classification/country_classification_historical.xlsx"):
    # TODO: is this applied to all hazards or just floods?
    # TODO: use FLOPROS V1 for protection; try to infer missing countries from FLOPROS based on GDP-protection correlation
    if 'rp' in index_.names:
        index_ = index_.droplevel('rp').drop_duplicates()
    if 'income_cat' in index_.names:
        index_ = index_.droplevel('income_cat').drop_duplicates()
    prot = pd.Series(index=index_, name="protection", data=0.)
    if protection_data in ['FLOPROS', 'country_income']:
        if protection_data == 'FLOPROS':
            prot_data = load_input_data(root_dir, flopros_protection_file, index_col=0)
            # prot_data = df_to_iso3(prot_data, 'country', any_to_wb)
            # prot_data = prot_data[~prot_data.iso3.isna()]
            # set ISO3 country codes and average over duplicates (for West Bank and Gaza Strip, which are both assigned to
            # ISO3 code PSE)
            prot_data = prot_data.drop('country', axis=1).set_index('iso3')
            prot_data.rename({'MerL_Riv': 'Flood', 'MerL_Co': 'Storm surge'}, axis=1, inplace=True)
        elif protection_data == 'country_income':  # assumed a function of the country's income group
            # note: "protection_level_assumptions.csv" can be found in orig_inputs.
            prot_assumptions = load_input_data(root_dir, protection_level_assumptions_file,
                                               index_col="Income group").squeeze()
            # prot_data = load_input_data(root_dir, income_groups_file, header=4, index_col=3)["Income group"].dropna()
            prot_data = load_input_data(root_dir, income_groups_file, header=0)[["Code", "Income group"]]
            prot_data = prot_data.dropna(how='all').rename({'Code': 'iso3', 'Income group': 'protection'}, axis=1)
            prot_data = prot_data.set_index('iso3').squeeze()
            prot_data.replace(prot_assumptions, inplace=True)

            # use historical data for countries that are not in the income_groups_file
            prot_data_hist = load_input_data(root_dir, income_groups_file_historical,
                                             sheet_name='Country Analytical History', header=5)
            prot_data_hist = prot_data_hist.rename({prot_data_hist.columns[0]: 'iso3'}, axis=1).dropna(subset='iso3')
            prot_data_hist.set_index('iso3', inplace=True)
            prot_data_hist.drop(prot_data_hist.columns[0], axis=1, inplace=True)
            prot_data_hist = prot_data_hist.stack().rename('protection').replace(prot_assumptions)
            prot_data_hist.index.names = ['iso3', 'year']
            prot_data_hist = prot_data_hist.reset_index()
            prot_data_hist = prot_data_hist.loc[prot_data_hist.groupby('iso3')['year'].idxmax()]
            prot_data_hist = prot_data_hist.set_index('iso3')['protection']
            prot_data = prot_data.fillna(prot_data_hist).dropna()

            hazards = hazard_types.split('+')
            prot_data = pd.concat([prot_data] * len(hazards), axis=1)
            prot_data.columns = hazards
        prot_data = prot_data.stack()
        prot_data.index.names = ['iso3', 'hazard']
        # prot_data = pd.concat([prot_data] * 5, keys=[f"q{q}" for q in range(1, 6)], names=['income_cat'])
        # prot_data = prot_data.reset_index().set_index(['iso3', 'hazard', 'income_cat']).squeeze().sort_index()
        prot_data.name = 'protection'
        prot.loc[np.intersect1d(prot_data.index, prot.index)] = prot_data
    elif protection_data == 'None':
        print("No protection data applied.")
    else:
        raise ValueError(f"Unknown value for protection_data: {protection_data}")
    if min_rp is not None:
        prot.loc[prot < min_rp] = min_rp
    # res = pd.merge(hazard_ratios_, prot, left_index=True, right_index=True, how='left').reset_index('rp')
    # res.protection = res.protection > res.rp
    # res = res.set_index('rp', append=True).swaplevel(2, 3).sort_index()
    return prot


def get_early_warning_per_hazard(index_, ew_per_country_, no_ew_hazards="Earthquake"):
    ew_per_hazard_ = pd.Series(index=index_, data=0.0, name='ew')
    common_countries = np.intersect1d(index_.get_level_values('iso3').unique(), ew_per_country_.index)
    ew_per_hazard_.loc[common_countries] += ew_per_country_.loc[common_countries]
    ew_per_hazard_.loc[:, no_ew_hazards.split('+')] = 0
    ew_per_hazard_.fillna(0, inplace=True)
    return ew_per_hazard_


# TODO: update with new PEB data
def apply_poverty_exposure_bias(exposure_fa_, use_avg_pe_, population_data_=None,  # n_quantiles_, poor_categories_,
                                peb_data_path="PEB/exposure_bias_per_quintile.csv",
                                # peb_povmaps_filepath_="PEB_flood_povmaps.xlsx",
                                # peb_deltares_filepath_="PEB_wb_deltares.csv", peb_hazards_="Flood+Storm surge"
                                ):
    # # Exposure bias from WB povmaps study
    # exposure_bias_wb = load_input_data(root_dir, peb_povmaps_filepath_)[["iso", "peb"]].dropna()
    # exposure_bias_wb = exposure_bias_wb.rename({'iso': 'iso3'}, axis=1).set_index('iso3').peb - 1
    #
    # # Exposure bias from older WB DELTARES study
    # exposure_bias_deltares = load_input_data(root_dir, peb_deltares_filepath_, skiprows=[0, 1, 2],
    #                                          usecols=["Country", "Nation-wide"])
    # exposure_bias_deltares = df_to_iso3(exposure_bias_deltares, 'Country', any_to_wb)
    # exposure_bias_deltares = exposure_bias_deltares.set_index('iso3').drop('Country', axis=1)
    # exposure_bias_deltares.rename({'Nation-wide': 'peb'}, axis=1, inplace=True)
    # exposure_bias_deltares.dropna(inplace=True)
    #
    # # Completes with bias from previous study when pov maps not available.
    # peb = pd.merge(exposure_bias_wb, exposure_bias_deltares, how='outer', left_index=True, right_index=True)
    # peb['peb_x'].fillna(peb['peb_y'], inplace=True)
    # peb.drop('peb_y', axis=1, inplace=True)
    # peb.rename({'peb_x': 'peb'}, axis=1, inplace=True)
    # peb = peb.peb

    bias = load_input_data(root_dir, peb_data_path, index_col=[0, 1, 2]).squeeze()
    missing_index = pd.MultiIndex.from_tuples(
        np.setdiff1d(exposure_fa_.index.droplevel('rp').unique(), bias.index.unique()), names=bias.index.names)
    bias = pd.concat((bias, pd.Series(index=missing_index, data=np.nan)), axis=0).rename(bias.name).sort_index()
    bias = bias.loc[np.intersect1d(bias.index, exposure_fa_.index.droplevel('rp').unique())]

    # if use_avg_pe, use averaged pe from global data for countries that don't have PE. Otherwise use 0.
    if use_avg_pe_:
        if population_data_ is not None:
            bias_pop = pd.merge(bias.dropna(), population_data_, left_index=True, right_index=True, how='left')
            bias_wavg = (bias_pop.product(axis=1).groupby(['hazard', 'income_cat']).sum()
                         / bias_pop['pop'].groupby(['hazard', 'income_cat']).sum()).rename('exposure_bias')
            bias = pd.merge(bias, bias_wavg, left_index=True, right_index=True, how='left')
            bias.exposure_bias_x.fillna(bias.exposure_bias_y, inplace=True)
            bias.drop('exposure_bias_y', axis=1, inplace=True)
            bias = bias.rename({'exposure_bias_x': 'exposure_bias'}, axis=1).squeeze()
            bias = bias.swaplevel('income_cat', 'iso3').swaplevel('hazard', 'iso3').sort_index()
            # TODO: averages show exposure bias towards hig-income quintiles. Check when PEB data gets updated with
            #  more poverty lines and harmonized poverty headcount / population data.
        else:
            raise Exception("Population data not available. Cannot use average PE.")
    bias.fillna(1, inplace=True)

    exposure_fa_with_peb_ = (exposure_fa_ * bias).swaplevel('rp', 'income_cat').sort_index().rename(exposure_fa.name)

    # selects just flood and surge
    # peb_hazards_ = peb_hazards_.split("+")

    # exposure_fa_with_peb_.loc[:, peb_hazards_, :, poor_categories_] *= (1 + peb)
    # nonpoor_categories = exposure_fa_with_peb_.index.get_level_values('income_cat').unique().drop(poor_categories_)
    # # (1 / n_quantiles_) = poverty_headcount
    # exposure_fa_with_peb_.loc[:, peb_hazards_, :, nonpoor_categories] *= ((1 - (1 / n_quantiles_) * (1 + peb))
    #                                                             / (1 - (1 / n_quantiles_)))
    return exposure_fa_with_peb_


def compute_exposure_and_adjust_vulnerability(hazard_loss_tot_, vulnerability_, fa_threshold_, event_level_):
    exposure_fa_guessed_ = (hazard_loss_tot_ / vulnerability_.loc[(slice(None), 'tot')]).dropna().rename('fa')

    vulnerability_quintiles = vulnerability_.unstack('income_cat').drop('tot', axis=1).stack().rename('v')

    # merge vulnerability with hazard_ratios
    fa_v_merged = pd.merge(exposure_fa_guessed_, vulnerability_quintiles, left_index=True, right_index=True)

    # clip f_a to fa_threshold; increase v s.th. \Delta K = f_a * v is constant
    if np.any(fa_v_merged.fa > fa_threshold_):
        excess_exposure = fa_v_merged.loc[fa_v_merged.fa > fa_threshold_, 'fa']
        print(f"Exposure values f_a are above fa_threshold for {len(excess_exposure)} entries. Setting them to "
              f"fa_threshold and adjusting vulnerability accordingly.")
        r = excess_exposure / fa_threshold_  # ratio by which fa is above fa_threshold
        r.name = 'r'
        # set f_a to fa_threshold
        fa_v_merged.loc[excess_exposure.index, 'fa'] = fa_threshold_
        # increase v s.th. \Delta K = f_a * v does not change
        fa_v_merged.loc[excess_exposure.index, 'v'] = fa_v_merged.loc[excess_exposure.index, 'v'] * r
        print("Adjusted exposure and vulnerability with factor r:")
        print(pd.concat((fa_v_merged.loc[excess_exposure.index].rename({'fa': 'fa_new', 'v': 'v_new'}, axis=1), r),
                        axis=1))
        if np.any(fa_v_merged.loc[excess_exposure.index, 'v'] > .99):
            # TODO: here, vulnerability is simply clipped. Think more about what this may cause.
            print(f"The adjustment of exposure and vulnerability resulted in excess vulnerability for some entries."
                  f" Setting v_tot =.99 for these regions.")
            excess_vulnerability = fa_v_merged.loc[excess_exposure.index, 'v'][fa_v_merged.loc[excess_exposure.index, 'v'] > .99]
            fa_v_merged.loc[excess_vulnerability.index, 'v'] = .99
            print(f"Clipped excess vulnerability to .99 for {len(excess_vulnerability)} entries.")

    exposure_fa_guessed_ = fa_v_merged.fa
    vulnerability_per_income_cat_adjusted_ = fa_v_merged.v
    return exposure_fa_guessed_, vulnerability_per_income_cat_adjusted_


def load_vulnerability_data(income_shares_, n_quantiles=5,
                            gem_vulnerability_classes_filepath_="GEM_vulnerability/country_vulnerability_classes.csv",
                            building_class_vuln_path="GEM_vulnerability/building_class_to_vulenrability_mapping.csv"):
    house_cats = load_input_data(root_dir, gem_vulnerability_classes_filepath_,
                                     index_col="iso3").drop('country', axis=1)

    # matching vulnerability of buildings and people's income and calculate poor's, rich's and country's vulnerability
    hous_cat_vulnerability = load_input_data(root_dir, building_class_vuln_path, sep=";",
                                             index_col="aggregate_category").squeeze()

    # TODO here, simply assuming that the poorest quantiles occupy the most vulnerable housing. A better approach might
    #  be desirable, e.g. using GMD data
    quantiles = [f"q{q}" for q in range(1, n_quantiles + 1)]
    q_headcount = 1 / n_quantiles
    vulnerability_ = pd.Series(
        index=pd.MultiIndex.from_product((house_cats.index, quantiles), names=['iso3', 'income_cat']),
        name="v", dtype=float
    )
    for cum_head, quantile in zip(np.linspace(q_headcount, 1, n_quantiles), quantiles):
        share_q = (
            # cumulative shares of income categories until the current quantile
            (house_cats - (house_cats.cumsum(axis=1).add(-cum_head, axis=0)).clip(lower=0)).clip(0) -
            # cumulative shares of income categories until the category before (i.e., excluding) the current quantile
            (house_cats - (house_cats.cumsum(axis=1).add(-(cum_head - q_headcount), axis=0)).clip(lower=0)).clip(0)
        ) / q_headcount
        vulnerability_q = (share_q * hous_cat_vulnerability).sum(axis=1, skipna=False)
        vulnerability_.loc[(slice(None), quantile)] = vulnerability_q.values

    # vulnerability weighted with income shares
    # TODO: why weight vulenrability by income shares?
    vulnerability_tot = (vulnerability_ * income_shares_).unstack('income_cat').dropna().sum(axis=1)
    vulnerability_tot.rename('tot', inplace=True)

    vulnerability_ = pd.concat((vulnerability_.unstack('income_cat'), vulnerability_tot), axis=1).dropna().stack()
    vulnerability_.rename('v', inplace=True)
    vulnerability_.index.names = ['iso3', 'income_cat']
    return vulnerability_


def compute_borrowing_ability(credit_ratings_, finance_preparedness_=None, cat_ddo_filepath="CatDDO/catddo.xlsx"):
    borrowing_ability_ = copy.deepcopy(credit_ratings_).rename('borrowing_ability')
    if finance_preparedness_ is not None:
        borrowing_ability_ = pd.concat((borrowing_ability_, finance_preparedness_), axis=1).mean(axis=1).rename('borrowing_ability')
    catddo_countries = pd.concat(load_input_data(root_dir, cat_ddo_filepath, sheet_name=[0, 1, 2], header=0), axis=0)
    catddo_countries = df_to_iso3(catddo_countries, 'Country').iso3.unique()
    catddo_countries = pd.Series(index=catddo_countries, data=1, name='contingent_countries')
    borrowing_ability_ = pd.concat((borrowing_ability_, catddo_countries), axis=1)
    borrowing_ability_ = borrowing_ability_.borrowing_ability.fillna(borrowing_ability_.contingent_countries)
    return borrowing_ability_


def load_hfa_data():
    # HFA (Hyogo Framework for Action) data to assess the role of early warning system
    # TODO: check Côte d'Ivoire naming in HFA
    # 2015 hfa
    hfa15 = load_input_data(root_dir, "HFA_all_2013_2015.csv")
    hfa15 = hfa15.set_index('ISO 3')
    # READ THE LAST HFA DATA
    hfa_newest = load_input_data(root_dir, "HFA_all_2011_2013.csv")
    hfa_newest = hfa_newest.set_index('ISO 3')
    # READ THE PREVIOUS HFA DATA
    hfa_previous = load_input_data(root_dir, "HFA_all_2009_2011.csv")
    hfa_previous = hfa_previous.set_index('ISO 3')
    # most recent values... if no 2011-2013 reporting, we use 2009-2011

    # concat to harmonize indices
    hfa_oldnew = pd.concat([hfa_newest, hfa_previous, hfa15], axis=1, keys=['new', 'old', "15"])
    hfa_data = hfa_oldnew["15"].fillna(hfa_oldnew["new"].fillna(hfa_oldnew["old"]))

    # access to early warning normalized between zero and 1.
    hfa_data["ew"] = 1 / 5 * hfa_data["P2-C3"]

    # q_s in the report, ability to scale up support to affected population after the disaster
    # normalized between zero and 1
    hfa_data["prepare_scaleup"] = (hfa_data["P4-C2"] + hfa_data["P5-C2"] + hfa_data["P4-C5"]) / 3 / 5

    # between 0 and 1	!!!!!!!!!!!!!!!!!!!REMARK: INCONSISTENT WITH THE TECHNICAL PAPER. Q_f=1/2(ratings+P5C3/5)
    # TODO @Bramka @Stephane
    hfa_data["finance_pre"] = (1 + hfa_data["P5-C3"]) / 6

    hfa_data = hfa_data[["ew", "prepare_scaleup", "finance_pre"]]
    hfa_data.fillna(0, inplace=True)
    hfa_data.index.name = 'iso3'

    return hfa_data


def load_wrp_data(wrp_data_path_="WRP/lrf_wrp_2021_full_data.csv.zip", root_dir_='.', outfile=None):
    # previously used HFA indicators:
    # P4C2: Do social safety nets exist to increase the resilience of risk prone households and communities?
    # P5C2: Disaster preparedness plans and contingency plans are in place at all administrative levels, and regular
    #       training drills and rehearsals are held to test and develop disaster response programs.
    # P4C5: Disaster risk reduction measures are integrated into post‐disaster recovery and rehabilitation processes.
    # disaster preparedness related questions (see lrf_wrp_2021_full_data_dictionary.xlsx):

    # WRP indicators for disaster preparedness
    # Q16A	    Well Prepared to Deal With a Disaster: The National Government
    # Q16B	    Well Prepared to Deal With a Disaster: Hospitals
    # (Q16C)	Well Prepared to Deal With a Disaster: You and Your Family
    # Q16D	    Well Prepared to Deal With a Disaster: Local Government
    cat_preparedness_cols = ['Q16A', 'Q16D']

    # early warning related questions (see lrf_wrp_2021_full_data_dictionary.xlsx):
    # Q19A	Received Warning About Disaster From Internet/Social Media
    # Q19B	Received Warning About Disaster From Local Government or Police
    # Q19C	Received Warning About Disaster From Radio, TV, or Newspapers
    # Q19D	Received Warning About Disaster From Local Community Organization
    early_warning_cols = ['Q19A', 'Q19B', 'Q19C', 'Q19D']

    all_cols = cat_preparedness_cols + early_warning_cols

    # read data
    wrp_data_ = load_input_data(root_dir_, wrp_data_path_, dtype=object)[['Country', 'WGT', 'Year'] + all_cols]
    wrp_data_ = wrp_data_.replace(' ', np.nan)
    wrp_data_[all_cols + ['WGT']] = wrp_data_[all_cols + ['WGT']].astype(float)
    wrp_data_[['Country', 'Year']] = wrp_data_[['Country', 'Year']].astype(str)
    wrp_data_.set_index(['Country', 'Year'], inplace=True)

    # drop rows with no data
    wrp_data_ = wrp_data_.dropna(how='all', subset=all_cols)

    # 97 = Does Not Apply
    # 98 = Don't Know
    # 99 = Refused
    # consider all of the above as 'no'
    # 1 = Yes
    # 2 = No (set to 0)
    # 3 = It depends (only for Q16A-D, set to 0.5)
    wrp_data_[all_cols] = wrp_data_[all_cols].replace({97: 0, 98: 0, 99: 0, 2: 0, 3: 0.5})

    # consider early warning available, if at least one of the early warning questions was answered with 'yes'
    wrp_data_['ew'] = wrp_data_[early_warning_cols].sum(axis=1, skipna=False)
    wrp_data_.loc[~wrp_data_.ew.isna(), 'ew'] = (wrp_data_.loc[~wrp_data_.ew.isna(), 'ew'] > 0).astype(float)
    early_warning_cols = ['ew']

    indicators = []
    for indicator, cols in [('prepare_scaleup', cat_preparedness_cols), ('ew', early_warning_cols)]:
        data_ = wrp_data_[cols + ['WGT']].dropna(how='all', subset=cols)

        # calculate weighted mean of each sub-indicator. Then take the mean of the sub-indicators to get the indicator
        data_ = data_[cols].mul(data_.WGT, axis=0).groupby(['Country', 'Year']).sum().div(
            data_.WGT.groupby(['Country', 'Year']).sum(), axis=0).mean(axis=1).rename(indicator)
        data_ = data_.to_frame().reset_index()
        data_ = data_.loc[data_.groupby('Country').Year.idxmax()].drop('Year', axis=1).set_index('Country')
        indicators.append(data_)
    indicators = pd.concat(indicators, axis=1)

    indicators = df_to_iso3(indicators.reset_index(), 'Country', any_to_wb)
    indicators = indicators.set_index('iso3').drop('Country', axis=1)

    if outfile is not None:
        indicators.to_csv(outfile)

    return indicators


def load_credit_ratings(tradingecon_ratings_path="credit_ratings/2023-12-13_tradingeconomics_ratings.csv",
                        cia_ratings_raw_path="credit_ratings/2024-01-30_cia_ratings_raw.txt",
                        ratings_scale_path="credit_ratings/credit_ratings_scale.csv"):

    # Trading Economics ratings
    print("Warning. Check that [date]_tradingeconomics_ratings.csv is up to date. If not, download it from "
          "http://www.tradingeconomics.com/country-list/rating")
    # TODO: Fitch no longer on tradingeconomics.com. Find a different source? Also, here DBRS is not used. Why?
    te_ratings = load_input_data(root_dir, tradingecon_ratings_path, dtype="str", encoding="utf8", na_values=['NR'])
    te_ratings = te_ratings.dropna(how='all')
    te_ratings = te_ratings[["country", "S&P", "Moody's"]]  # TODO: why not DBRS / TE?

    # drop EU
    te_ratings = te_ratings[te_ratings.country != 'European Union']

    # rename Congo to Congo, Dem. Rep.
    te_ratings.replace({'Congo': 'Congo, Dem. Rep.'}, inplace=True)

    # change country name to iso3
    te_ratings = df_to_iso3(te_ratings, "country", any_to_wb)
    te_ratings = te_ratings.set_index("iso3").drop("country", axis=1)

    # make lower case and strip
    te_ratings = te_ratings.map(lambda x: str.strip(x).lower() if type(x) is str else x)

    # CIA ratings
    print("Warning. Check that [date]_cia_ratings_raw.csv is up to date. If not, copy the text from "
            "https://www.cia.gov/the-world-factbook/field/credit-ratings/ into [date]_cia_ratings_raw.csv")
    cia_ratings_raw = load_input_data(root_dir, cia_ratings_raw_path)
    cia_ratings = pd.DataFrame(columns=['country', 'agency', 'rating']).set_index(['country', 'agency'])
    current_country = None
    for line in cia_ratings_raw:
        if "rating:" in line:
            agency, rating_year = line.split(" rating: ")
            rating = rating_year.strip().split(" (")[0]
            cia_ratings.loc[(current_country, agency), 'rating'] = rating.strip().lower()
        elif "note:" not in line and len(line) > 0:
            current_country = line
    cia_ratings.reset_index(inplace=True)
    cia_ratings.replace({"Standard & Poors": "S&P", 'n/a': np.nan, 'nr': np.nan}, inplace=True)

    # drop EU
    cia_ratings = cia_ratings[cia_ratings.country != 'European Union']

    # change country name to iso3
    cia_ratings = df_to_iso3(cia_ratings, "country", any_to_wb)
    cia_ratings = cia_ratings.set_index(["iso3", "agency"]).drop("country", axis=1).squeeze().unstack('agency')

    # merge ratings
    ratings = pd.merge(te_ratings, cia_ratings, left_index=True, right_index=True, how='outer')
    for agency in np.intersect1d(te_ratings.columns, cia_ratings.columns):
        ratings[agency] = ratings[agency + "_x"].fillna(ratings[agency + "_y"])
        ratings.drop([agency + "_x", agency + "_y"], axis=1, inplace=True)

    # Transforms ratings letters into 1-100 numbers
    rating_scale = load_input_data(root_dir, ratings_scale_path)
    ratings["S&P"].replace(rating_scale["s&p"].values, rating_scale["score"].values, inplace=True)
    ratings["Moody's"].replace(rating_scale["moodys"].values, rating_scale["score"].values, inplace=True)
    ratings["Fitch"].replace(rating_scale["fitch"].values, rating_scale["score"].values, inplace=True)

    # average rating over all agencies
    ratings["rating"] = ratings.mean(axis=1) / 100

    # set ratings to 0 for countries with no rating
    if ratings.rating.isna().any():
        print("No rating available for regions:", "; ".join(ratings[ratings.rating.isna()].index), ". Setting rating to 0.")
        ratings.rating.fillna(0, inplace=True)

    return ratings.rating


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Script parameters')
    parser.add_argument('--use_flopros_protection', type=bool, default=True, help='Use FLOPROS for protection')
    parser.add_argument('--no_protection', action='store_true', help='No protection')
    parser.add_argument('--use_avg_pe', type=bool, default=True, help='Use average PE')
    parser.add_argument('--drop_unused_data', type=bool, default=True, help='Drop unused data')
    parser.add_argument('--econ_scope', type=str, default='iso3', help='Economic scope')
    parser.add_argument('--default_rp', type=str, default='default_rp', help='Default return period')
    parser.add_argument('--reconstruction_time', type=float, default=3.0, help='Reconstruction time')
    parser.add_argument('--reduction_vul', type=float, default=0.2, help='Reduction in vulnerability')
    parser.add_argument('--income_elasticity', type=float, default=1.5, help='Income elasticity')
    parser.add_argument('--discount_rate_rho', type=float, default=0.06, help='Discount rate')
    parser.add_argument('--shareable', type=float, default=0.8, help='Asset loss covered')
    parser.add_argument('--max_increased_spending', type=float, default=0.05, help='Maximum support')
    parser.add_argument('--fa_threshold', type=float, default=0.9, help='FA threshold')
    parser.add_argument('--axfin_impact', type=float, default=0.1, help='Increase of the fraction of diversified '
                                                                        'income through financial inclusion')
    parser.add_argument('--root_dir', type=str, default=os.getcwd(), help='Root directory')
    parser.add_argument('--no_optimized_recovery', action='store_true', help='Use fixed recovery duration'
                                                                             'instead of optimization.')
    parser.add_argument('--force_recovery_recompute', action='store_true', help='Force recomputation '
                                                                                'of recovery duration.')
    parser.add_argument('--test_run', action='store_true', help='If true, only runs with country USA')
    args = parser.parse_args()

    use_flopros_protection = args.use_flopros_protection
    no_protection = args.no_protection
    use_avg_pe = args.use_avg_pe
    default_rp = args.default_rp  # TODO: check if this can be removed
    reduction_vul = args.reduction_vul
    income_elasticity = args.income_elasticity
    discount_rate_rho = args.discount_rate_rho
    shareable = args.shareable
    max_increased_spending = args.max_increased_spending
    fa_threshold = args.fa_threshold
    axfin_impact = args.axfin_impact
    no_optimized_recovery = args.no_optimized_recovery
    test_run = args.test_run
    default_reconstruction_time = args.reconstruction_time
    force_recovery_recompute = args.force_recovery_recompute

    econ_scope = args.econ_scope
    event_level = [econ_scope, "hazard", "rp"]  # levels of index at which one event happens

    root_dir = args.root_dir
    input_dir = os.path.join(root_dir, 'inputs')  # get inputs data directory
    intermediate_dir = os.path.join(root_dir, 'intermediate')  # get outputs data directory

    # if the intermediate directory doesn't exist, create one
    if not os.path.exists(intermediate_dir):
        os.makedirs(intermediate_dir)

    any_to_wb, iso3_to_wb, iso2_iso3 = get_country_name_dicts(root_dir)

    # read WB data
    wb_data_macro = load_input_data(root_dir, "WB_socio_economic_data/wb_data_macro.csv").set_index(econ_scope)
    wb_data_cat_info = load_input_data(root_dir, "WB_socio_economic_data/wb_data_cat_info.csv").set_index(
        [econ_scope, 'income_cat'])
    # TODO: remove later
    if test_run:
        wb_data_macro_full = wb_data_macro
        wb_data_macro = wb_data_macro.loc[['USA']]
        wb_data_cat_info = wb_data_cat_info.loc[['USA']]

    n_quantiles = len(wb_data_cat_info.index.get_level_values('income_cat').unique())

    # read HFA data
    hfa_data = load_hfa_data()
    # # TODO: remove later
    # if test_run:
    #     hfa_data = hfa_data.loc[['USA']]
    # TODO: WRP data contains about 30 countries less than HFA data
    wrp_data = load_wrp_data("WRP/lrf_wrp_2021_full_data.csv.zip", root_dir)

    # merge HFA and WRP data: take the mean of the two data sets where both are available (i.e., ew + prepare_scaleup)
    disaster_preparedness = hfa_data[['finance_pre']].copy()
    disaster_preparedness['ew'] = pd.concat((hfa_data['ew'], wrp_data['ew']), axis=1).mean(axis=1)
    disaster_preparedness['prepare_scaleup'] = pd.concat((hfa_data['prepare_scaleup'], wrp_data['prepare_scaleup']), axis=1).mean(axis=1)

    # read credit ratings
    credit_ratings = load_credit_ratings()
    # TODO: remove later
    if test_run:
        credit_ratings = credit_ratings.loc[['USA']]

    # compute country borrowing ability
    borrowing_ability = compute_borrowing_ability(credit_ratings, disaster_preparedness.finance_pre)
    # borrowing_ability = compute_borrowing_ability(credit_ratings)

    # load average productivity of capital
    avg_prod_k = gather_capital_data(root_dir).avg_prod_k

    # load building vulnerability classification and compute vulnerability per income class
    vulnerability = load_vulnerability_data(income_shares_=wb_data_cat_info.income_share,
                                            n_quantiles=len(wb_data_cat_info.index.get_level_values('income_cat').unique()))

    # load total hazard losses per return period (on the country level)
    hazard_loss_tot = load_gir_hazard_losses(root_dir,
                                             "GIR_hazard_loss_data/export_all_metrics.csv.zip", default_rp,
                                             extrapolate_rp=False)
    # TODO: remove later
    if test_run:
        hazard_loss_tot = hazard_loss_tot.loc[['USA']]

    # compute exposure and adjust vulnerability (per income category) for excess exposure
    exposure_fa, vulnerability_per_income_cat_adjusted = compute_exposure_and_adjust_vulnerability(
        hazard_loss_tot, vulnerability, fa_threshold, event_level
    )

    # apply poverty exposure bias
    exposure_fa_with_peb = apply_poverty_exposure_bias(exposure_fa, use_avg_pe,
                                                       wb_data_macro['pop'] if not test_run else wb_data_macro_full['pop'])
    # TODO: figure out what these data are and if to keep them
    # TODO: constant_fa does not come in five income categories; ignoring for now
    # if update_exposure_with_constant_fa:
    #     constant_fa = load_input_data(root_dir, 'constant_fa.csv')
    #     constant_fa = df_to_iso3(constant_fa, 'country', any_to_wb)
    #     constant_fa.set_index(event_level + ['income_cat'], inplace=True)
    #     exposure_fa_with_peb.update(constant_fa.fa)

    # expand the early warning to all hazard rp's, income categories, and countries; set to 0 for specific hazrads
    # (Earthquake)
    # early_warning_per_hazard = get_early_warning_per_hazard(exposure_fa_with_peb.index)
    early_warning_per_hazard = get_early_warning_per_hazard(exposure_fa_with_peb.index, disaster_preparedness.ew)

    # concatenate exposure, vulnerability and early warning into one dataframe
    hazard_ratios = pd.concat((exposure_fa_with_peb, vulnerability_per_income_cat_adjusted, early_warning_per_hazard),
                              axis=1).dropna()

    # get data per income categories
    cat_info, tau_tax = get_cat_info_and_tau_tax(wb_data_cat_info_=wb_data_cat_info, wb_data_macro_=wb_data_macro,
                                                 avg_prod_k_=avg_prod_k, n_quantiles_=n_quantiles,
                                                 axfin_impact_=axfin_impact)

    if no_optimized_recovery:
        cat_info['recovery_time'] = default_reconstruction_time
    else:
        recovery = get_recovery_duration(avg_prod_k, vulnerability, discount_rate_rho,
                                         force_recompute=force_recovery_recompute)
        cat_info = pd.merge(cat_info, recovery, left_index=True, right_index=True)
        # TODO: remove later
        if test_run:
            recovery = recovery.loc[['USA']]
    if no_protection:
        hazard_protection = load_protection(hazard_ratios.index, protection_data="None", min_rp=0)
    else:
        if use_flopros_protection:
            hazard_protection = load_protection(hazard_ratios.index, protection_data="FLOPROS", min_rp=1)
        else:
            hazard_protection = load_protection(hazard_ratios.index, protection_data="country_income", min_rp=1)

    # TODO: before, also included pov_head, pi (vulnerability reduction with early warning), income_elast, rho,
    #  shareable, max_increased_spending, protection; check if this is necessary
    macro = wb_data_macro
    macro = macro.join(disaster_preparedness['prepare_scaleup'], how='left')
    macro = macro.join(borrowing_ability, how='left')
    macro = macro.join(avg_prod_k, how='left')
    macro = macro.join(tau_tax, how='left')
    # TODO: these global parameters should eventually be moved elsewhere and not stored in macro!
    macro['rho'] = discount_rate_rho
    macro['max_increased_spending'] = max_increased_spending
    macro['shareable'] = shareable
    macro['income_elasticity'] = income_elasticity

    # clean and harmonize data frames
    macro.dropna(inplace=True)
    cat_info.dropna(inplace=True)
    hazard_ratios.dropna(inplace=True)
    hazard_protection.dropna(inplace=True)
    common_regions = [c for c in macro.index if c in cat_info.index and c in hazard_ratios.index and c in hazard_protection.index]
    macro = macro.loc[common_regions]
    cat_info = cat_info.loc[common_regions]
    hazard_ratios = hazard_ratios.loc[common_regions]
    hazard_protection = hazard_protection.loc[common_regions]

    # TODO: check that different policies still work
    # for pol_str, pol_opt in [[None, None], ['bbb_complete', 1], ['borrow_abi', 2], ['unif_poor', None], ['bbb_incl', 1],
    #                          ['bbb_fast', 1], ['bbb_fast', 2], ['bbb_fast', 4], ['bbb_fast', 5], ['bbb_50yrstand', 1]]:
    for pol_str, pol_opt in [[None, None]]:

        # apply policy
        scenario_macro, scenario_cat_info, scenario_hazard_ratios, pol_desc = apply_policy(
            macro_=macro.copy(deep=True),
            cat_info_=cat_info.copy(deep=True),
            hazard_ratios_=hazard_ratios.copy(deep=True),
            event_level=event_level,
            policy_name=pol_str,
            policy_opt=pol_opt
        )

        # TODO: ideally, all variables that are computed in gather_data.py should be computed in
        #  recomputed_after_policy_change(); before, only input data should be loaded to avoid double computation and
        #  recomputation errors.

        # TODO: check that scenario_hazard_ratios_rec (return periods extrapolated to protection rp's) has the same AAL
        #  as scenario_hazard_ratios (which does not include protection return periods)

        # clean data
        # TODO: can be removed after compute_resilience_and_risk.py is updated to use variable 'diversified_share'
        scenario_macro_rec, scenario_cat_info_rec, scenario_hazard_ratios_rec = recompute_after_policy_change(
            macro_=scenario_macro,
            cat_info_=scenario_cat_info,
            hazard_ratios_=scenario_hazard_ratios,
            econ_scope_=econ_scope,
            axfin_impact_=axfin_impact,
            pi_=reduction_vul,
        )
        scenario_cat_info_rec.drop('social', axis=1, inplace=True)

        # Save all data
        print(scenario_macro_rec.shape[0], 'countries in analysis')
        outstring = (f"_{pol_str}" if pol_str is not None else '') + (str(pol_opt) if pol_opt is not None else '')

        # TODO: should save vulenrability_per_income_cat_adjusted instead of vulnerability here!
        # save vulnerability (total, poor, rich) by country
        vulnerability.to_csv(
            os.path.join(intermediate_dir + "/scenario__vulnerability_unadjusted" + outstring + ".csv"),
            encoding="utf-8",
            header=True
        )

        # save protection by country and hazard
        hazard_protection.to_csv(
            os.path.join(intermediate_dir + "/scenario__hazard_protection" + outstring + ".csv"),
            encoding="utf-8",
            header=True
        )

        # save macro-economic country economic data
        scenario_macro_rec.to_csv(
            os.path.join(intermediate_dir + "/scenario__macro" + outstring + ".csv"),
            encoding="utf-8",
            header=True
        )

        # save consumption, access to finance, gamma, capital, exposure, early warning access by country and income category
        scenario_cat_info_rec.to_csv(
            os.path.join(intermediate_dir + "/scenario__cat_info" + outstring + ".csv"),
            encoding="utf-8",
            header=True
        )

        # save exposure, vulnerability, and access to early warning by country, hazard, return period, income category
        scenario_hazard_ratios_rec.to_csv(
            os.path.join(intermediate_dir + "/scenario__hazard_ratios" + outstring + ".csv"),
            encoding="utf-8",
            header=True
        )
