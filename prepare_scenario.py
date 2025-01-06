# This script provides data input for the resilience indicator multihazard model.
# The script was developed by Robin Middelanis based on previous work by Adrien Vogt-Schilb and Jinqiang Chen.
import argparse

import numpy as np

from gather_findex_data import get_liquidity_from_findex, gather_axfin_data
from gather_gem_data import gather_gem_data
from gather_gir_data import load_gir_hazard_loss_rel
from get_wb_data import get_wb_data
from lib_prepare_scenario import *
from apply_policy import *
import pandas as pd
from lib import get_country_name_dicts, df_to_iso3
from wb_api_wrapper import get_wb_mrv
from time import time


def get_cat_info_and_tau_tax(cat_info_, wb_data_macro_, avg_prod_k_, n_quantiles_, axfin_impact_,
                             scale_non_diversified_income_, min_diversified_share_):
    print("Computing diversified income share and tax...")
    cat_info_['diversified_share'] = cat_info_.social + cat_info_.axfin * axfin_impact_
    cat_info_['diversified_share'] = (1 - (1 - cat_info_['diversified_share']) * scale_non_diversified_income_).clip(upper=1, lower=min_diversified_share_)

    cat_info_['n'] = 1 / n_quantiles_
    cat_info_['c'] = cat_info_.income_share / cat_info_.n * wb_data_macro_.gdp_pc_pp

    # compute tau tax and gamma_sp from social
    tau_tax_, cat_info_["gamma_SP"] = social_to_tx_and_gsp(cat_info_)

    # compute capital per income category
    cat_info_["k"] = (1 - cat_info_.diversified_share) * cat_info_.c / ((1 - tau_tax_) * avg_prod_k_)

    cat_info_ = cat_info_.loc[cat_info_.drop('n', axis=1).dropna(how='all').index]
    return cat_info_, tau_tax_


def load_protection(index_, root_dir_, protection_data="FLOPROS", min_rp=1, hazard_types="Flood+Storm surge",
                    flopros_protection_file="FLOPROS/FLOPROS_protection_processed/flopros_protection_processed.csv",
                    protection_level_assumptions_file="WB_country_classification/protection_level_assumptions.csv",
                    income_groups_file="WB_country_classification/country_classification.xlsx",
                    income_groups_file_historical="WB_country_classification/country_classification_historical.xlsx"):
    print("Loading protection data...")
    if 'rp' in index_.names:
        index_ = index_.droplevel('rp').drop_duplicates()
    if 'income_cat' in index_.names:
        index_ = index_.droplevel('income_cat').drop_duplicates()
    prot = pd.Series(index=index_, name="protection", data=0.)
    if protection_data in ['FLOPROS', 'country_income']:
        if protection_data == 'FLOPROS':
            prot_data = pd.read_csv(os.path.join(root_dir_, "inputs/raw/", flopros_protection_file), index_col=0)
            prot_data = prot_data.drop('country', axis=1).set_index('iso3')
            prot_data.rename({'MerL_Riv': 'Flood', 'MerL_Co': 'Storm surge'}, axis=1, inplace=True)
        elif protection_data == 'country_income':  # assumed a function of the country's income group
            prot_assumptions = pd.read_csv(os.path.join(root_dir_, "inputs/raw/", protection_level_assumptions_file),
                                               index_col="Income group").squeeze()
            prot_data = pd.read_csv(os.path.join(root_dir_, "inputs/raw/", income_groups_file), header=0)[["Code", "Income group"]]
            prot_data = prot_data.dropna(how='all').rename({'Code': 'iso3', 'Income group': 'protection'}, axis=1)
            prot_data = prot_data.set_index('iso3').squeeze()
            prot_data.replace(prot_assumptions, inplace=True)

            # use historical data for countries that are not in the income_groups_file
            prot_data_hist = pd.read_excel(os.path.join(root_dir_, "inputs/raw/", income_groups_file_historical),
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
        else:
            raise ValueError(f"Unknown value for protection_data: {protection_data}")
        prot_data = prot_data.stack()
        prot_data.index.names = ['iso3', 'hazard']
        prot_data.name = 'protection'
        prot.loc[np.intersect1d(prot_data.index, prot.index)] = prot_data
    elif protection_data == 'None':
        print("No protection data applied.")
    else:
        raise ValueError(f"Unknown value for protection_data: {protection_data}")
    if min_rp is not None:
        prot.loc[prot < min_rp] = min_rp
    return prot

def load_findex_liquidity_and_axfin(root_dir_, any_to_wb_, force_recompute_=True, verbose=True):
    outpath = os.path.join(root_dir_, 'inputs', 'processed', 'findex_liquidity_and_axfin.csv')
    if not force_recompute_ and os.path.exists(outpath):
        print("Loading liquidity and axfin data from file...")
        liquidity_and_axfin = pd.read_csv(outpath)
        return liquidity_and_axfin.set_index(['iso3', 'income_cat'])
    print("Recomputing liquidity and axfin data from FINDEX...")
    findex_data_paths = {
        2021: os.path.join(root_dir_, "inputs/raw/", 'FINDEX', 'WLD_2021_FINDEX_v03_M.csv'),
        2017: os.path.join(root_dir_, "inputs/raw/", 'FINDEX', 'WLD_2017_FINDEX_v02_M.csv'),
        2014: os.path.join(root_dir_, "inputs/raw/", 'FINDEX', 'WLD_2014_FINDEX_v01_M.csv'),
        2011: os.path.join(root_dir_, "inputs/raw/", 'FINDEX', 'WLD_2011_FINDEX_v02_M.csv'),
    }
    liquidity_ = get_liquidity_from_findex(root_dir_, any_to_wb_, findex_data_paths, write_output_=False, verbose=verbose)
    liquidity_ = liquidity_[['liquidity_share', 'liquidity']].prod(axis=1).rename('liquidity')
    liquidity_ = liquidity_.iloc[liquidity_.reset_index().groupby(['iso3', 'income_cat']).year.idxmax()]
    liquidity_ = liquidity_.droplevel('year')

    axfin_ = gather_axfin_data(root_dir_, any_to_wb_, findex_data_paths, write_output_=False, verbose=verbose)
    axfin_ = axfin_.iloc[axfin_.reset_index().groupby(['iso3', 'income_cat']).year.idxmax()].axfin.droplevel('year')

    liquidity_and_axfin = pd.merge(liquidity_, axfin_, left_index=True, right_index=True, how='inner')

    liquidity_and_axfin.to_csv(outpath)
    return liquidity_and_axfin


def calc_reconstruction_share_sigma(root_dir_, any_to_wb_, scale_self_employment=1, imf_capital_data_file="IMF_capital/IMFInvestmentandCapitalStockDataset2021.xlsx",
                                    reconstruction_capital_='prv', verbose=True, force_recompute=True,
                                    labor_share_data_file="ILO_labor_share_of_GDP/LAP_2GDP_NOC_RT_A-filtered-2024-04-23.csv"):
    capital_shares_before_adjustment_path = os.path.join(root_dir_, "inputs/processed/capital_shares.csv")
    if not force_recompute and os.path.exists(capital_shares_before_adjustment_path):
        print("Loading capital shares from file...")
        capital_shares = pd.read_csv(capital_shares_before_adjustment_path, index_col='iso3')
    else:
        print("Recomputing capital shares...")
        imf_data = pd.read_excel(os.path.join(root_dir_, "inputs/raw/", imf_capital_data_file), sheet_name='Dataset')
        imf_data = imf_data.rename(columns={'isocode': 'iso3'}).set_index(['iso3'])[['year', 'kgov_n', 'kpriv_n', 'kppp_n']]
        imf_data['kpub_n'] = imf_data[['kgov_n', 'kppp_n']].sum(axis=1, skipna=True)
        imf_data = imf_data.replace(0, np.nan)
        imf_data = imf_data.rename({'kpriv_n': 'knonpub_n'}, axis=1)[['year', 'kpub_n', 'knonpub_n']].dropna(axis=0, how='any')
        imf_data = imf_data.iloc[imf_data.reset_index().groupby('iso3').year.idxmax()].drop('year', axis=1)
        imf_data['k_pub_share_kappa'] = imf_data['kpub_n'] / (imf_data['knonpub_n'] + imf_data['kpub_n'])

        labor_share_of_gdp = pd.read_csv(os.path.join(root_dir_, "inputs/raw/", labor_share_data_file))
        labor_share_of_gdp = labor_share_of_gdp.rename({'ref_area.label': 'country'}, axis=1).set_index('country').obs_value
        labor_share_of_gdp = labor_share_of_gdp.rename('labor_share_of_gdp') / 100
        labor_share_of_gdp.drop(['MENA', 'Central Africa'], inplace=True)

        capital_share = (1 - labor_share_of_gdp).rename('capital_elasticity_alpha')
        capital_share = df_to_iso3(capital_share.reset_index(), 'country', any_to_wb_, verbose_=verbose)
        capital_share = capital_share.dropna().set_index('iso3')['capital_elasticity_alpha']

        self_employment = get_wb_mrv('SL.EMP.SELF.ZS', 'self_employment') / 100
        self_employment = df_to_iso3(self_employment.reset_index(), 'country', any_to_wb_, verbose_=verbose).dropna(subset='iso3')
        self_employment = self_employment.set_index('iso3').drop('country', axis=1).squeeze()

        capital_shares = pd.merge(capital_share, imf_data.k_pub_share_kappa, left_index=True, right_index=True, how='inner')
        capital_shares = pd.merge(capital_shares, self_employment, left_index=True, right_index=True, how='inner')

        capital_shares.to_csv(capital_shares_before_adjustment_path)

    capital_shares['self_employment'] *= scale_self_employment

    capital_shares['k_pub_share'] = capital_shares.k_pub_share_kappa
    denominator = capital_shares.capital_elasticity_alpha + capital_shares.self_employment * (
                1 - capital_shares.capital_elasticity_alpha)
    capital_shares['k_oth_share'] = capital_shares.capital_elasticity_alpha * (
                1 - capital_shares.k_pub_share_kappa) / denominator
    capital_shares['k_prv_share'] = (1 - capital_shares.k_pub_share_kappa) * capital_shares.self_employment * (
                1 - capital_shares.capital_elasticity_alpha) / denominator

    if reconstruction_capital_ == 'prv':
        capital_shares['reconstruction_share_sigma_h'] = capital_shares.k_prv_share
    elif reconstruction_capital_ == 'prv_oth':
        capital_shares['reconstruction_share_sigma_h'] = capital_shares.k_prv_share + capital_shares.k_oth_share

    return capital_shares



def get_early_warning_per_hazard(index_, ew_per_country_, no_ew_hazards="Earthquake+Tsunami"):
    ew_per_hazard_ = pd.Series(index=index_, data=0.0, name='ew')
    common_countries = np.intersect1d(index_.get_level_values('iso3').unique(), ew_per_country_.index)
    ew_per_hazard_.loc[common_countries] += ew_per_country_.loc[common_countries]
    ew_per_hazard_.loc[:, no_ew_hazards.split('+')] = 0
    ew_per_hazard_.fillna(0, inplace=True)
    return ew_per_hazard_


def apply_poverty_exposure_bias(root_dir_, exposure_fa_, use_avg_pe_, population_data_=None,
                                peb_data_path="PEB/exposure_bias_per_quintile.csv"):
    bias = pd.read_csv(os.path.join(root_dir_, "inputs/raw/", peb_data_path), index_col=[0, 1, 2]).squeeze()
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
        else:
            raise Exception("Population data not available. Cannot use average PE.")
    bias.fillna(1, inplace=True)
    exposure_fa_with_peb_ = (exposure_fa_ * bias).swaplevel('rp', 'income_cat').sort_index().rename(exposure_fa_.name)
    return exposure_fa_with_peb_


def compute_exposure_and_vulnerability(root_dir_, fa_threshold_, n_quantiles, verbose=True, force_recompute_=False,
                                       apply_exposure_bias=True, population_data=None):
    outpath = os.path.join(root_dir_, 'inputs', 'processed')
    exposure_path = os.path.join(outpath, 'exposure_fa_guessed.csv')
    vuln_adj_path = os.path.join(outpath, 'vulnerability_adjusted.csv')
    vuln_unadj_path = os.path.join(outpath, 'vulnerability_unadjusted.csv')
    if not force_recompute_ and os.path.exists(exposure_path) and os.path.exists(vuln_adj_path) and os.path.exists(vuln_unadj_path):
            print("Loading exposure and vulnerability from file...")
            exposure_fa_ = pd.read_csv(exposure_path, index_col=[0, 1, 2, 3]).squeeze()
            vulnerability_adjusted_ = pd.read_csv(vuln_adj_path, index_col=[0, 1, 2, 3])
            vulnerability_unadjusted_ = pd.read_csv(vuln_unadj_path, index_col=[0, 1, 2])
    else:
        print("Computing exposure and vulnerability...")
        # load total hazard losses per return period (on the country level)
        hazard_loss_rel = load_gir_hazard_loss_rel(
            gir_filepath_=os.path.join(root_dir_, "inputs/raw/GIR_hazard_loss_data/export_all_metrics.csv.zip"),
            extrapolate_rp_=False,
            climate_scenario="Existing climate",
            verbose=verbose,
        )

        # load building vulnerability classification and compute vulnerability per income class
        vulnerability_unadjusted_ = load_vulnerability_data(
            root_dir_=root_dir_,
            n_quantiles_=n_quantiles,
            use_gmd_to_distribute=True,
            fill_missing_gmd_with_country_average=False,
            vulnerability_bounds='gem_extremes',
            verbose=verbose
        )

        exposure_fa_ = (hazard_loss_rel / vulnerability_unadjusted_.xs('tot', level='income_cat')).dropna().rename('fa')

        vulnerability_quintiles = vulnerability_unadjusted_.drop('tot', level='income_cat')

        # merge vulnerability with hazard_ratios
        fa_v_merged = pd.merge(exposure_fa_, vulnerability_quintiles, left_index=True, right_index=True)

        # clip f_a to fa_threshold; increase v s.th. \Delta K = f_a * v is constant
        if np.any(fa_v_merged.fa > fa_threshold_):
            excess_exposure = fa_v_merged.loc[fa_v_merged.fa > fa_threshold_, 'fa']
            if verbose:
                print(f"Exposure values f_a are above fa_threshold for {len(excess_exposure)} entries. Setting them to "
                      f"fa_threshold={fa_threshold_} and adjusting vulnerability accordingly.")
            r = excess_exposure / fa_threshold_  # ratio by which fa is above fa_threshold
            r.name = 'r'
            # set f_a to fa_threshold
            fa_v_merged.loc[excess_exposure.index, 'fa'] = fa_threshold_
            # increase v s.th. \Delta K = f_a * v does not change
            fa_v_merged.loc[excess_exposure.index, 'v'] = fa_v_merged.loc[excess_exposure.index, 'v'] * r
            if verbose:
                print("Adjusted exposure and vulnerability with factor r:")
                print(pd.concat((fa_v_merged.loc[excess_exposure.index].rename({'fa': 'fa_new', 'v': 'v_new'}, axis=1), r),
                                axis=1))
            if np.any(fa_v_merged.loc[excess_exposure.index, 'v'] > .99):
                excess_vulnerability = fa_v_merged.loc[excess_exposure.index, 'v'][fa_v_merged.loc[excess_exposure.index, 'v'] > .99]
                fa_v_merged.loc[excess_vulnerability.index, 'v'] = .99
                if verbose:
                    print(f"The adjustment of exposure and vulnerability resulted in excess vulnerability for some entries.")
                    print(f"Clipped excess vulnerability to .99 for {len(excess_vulnerability)} entries.")

        exposure_fa_ = fa_v_merged.fa
        vulnerability_adjusted_ = fa_v_merged.v

        exposure_fa_.to_csv(exposure_path)
        vulnerability_adjusted_.to_csv(vuln_adj_path)
        vulnerability_unadjusted_.to_csv(vuln_unadj_path)

    # apply poverty exposure bias
    if apply_exposure_bias:
        exposure_fa_ = apply_poverty_exposure_bias(
            root_dir_=root_dir_,
            exposure_fa_=exposure_fa_,
            use_avg_pe_=True,
            population_data_=population_data
        )
    return exposure_fa_, vulnerability_adjusted_, vulnerability_unadjusted_


def load_vulnerability_data(
        root_dir_,
        n_quantiles_=5,
        use_gmd_to_distribute=True,
        fill_missing_gmd_with_country_average=False,
        vulnerability_bounds='gem_extremes',
        gem_repo_root_dir="inputs/raw/GEM_vulnerability/global_exposure_model",
        hazus_gem_mapping_path="inputs/raw/GEM_vulnerability/hazus-gem_mapping.csv",
        gem_fields_path = "./inputs/raw/GEM_vulnerability/gem_taxonomy_fields.json",
        vulnarebility_class_mapping = "./inputs/raw/GEM_vulnerability/gem-to-vulnerability_mapping_per_hazard.xlsx",
        building_class_vuln_path="GEM_vulnerability/building_class_to_vulenrability_mapping.csv",
        gmd_vulnerability_distribution_path="GMD_vulnerability_distribution/Dwelling quintile vul ratio.xlsx",
        force_reompute=False,
        verbose=True
):
    # load distribution of vulnerability classes per country
    _, building_classes = gather_gem_data(
        gem_repo_root_dir_=os.path.join(root_dir_, gem_repo_root_dir),
        hazus_gem_mapping_path_=hazus_gem_mapping_path,
        gem_fields_path_=gem_fields_path,
        vuln_class_mapping_=vulnarebility_class_mapping,
        vulnerability_class_output_=None,
        weight_by='replacement_cost',
        verbose=verbose
    )
    building_classes.columns.names = ['hazard', 'vulnerability_class']
    building_classes = building_classes.droplevel('country', axis=0)
    if 'default' in building_classes.columns:
        building_classes = building_classes.drop('default', axis=1, level=0)

    # load vulnerability of building classes
    building_class_vuln = pd.read_csv(os.path.join(root_dir_, "inputs/raw/", building_class_vuln_path), index_col=0)
    building_class_vuln.index.name = 'hazard'
    building_class_vuln.columns.name = 'vulnerability_class'

    # compute extreme case of vuln. distribution, assuming that the poorest live in the most vulnerable buildings
    quantiles = [f"q{q}" for q in range(1, n_quantiles_ + 1)]
    q_headcount = 1 / n_quantiles_
    v_gem = pd.DataFrame(
        index=pd.MultiIndex.from_product((building_classes.index, quantiles), names=['iso3', 'income_cat']),
        columns=pd.Index(building_classes.columns.get_level_values(0).unique(), name='hazard'),
        dtype=float
    )
    for cum_head, quantile in zip(np.linspace(q_headcount, 1, n_quantiles_), quantiles):
        for hazard in v_gem.columns:
            share_h_q = (
                (building_classes[hazard] - (building_classes[hazard].cumsum(axis=1).add(-cum_head, axis=0)).clip(lower=0)).clip(0) -
                (building_classes[hazard] - (building_classes[hazard].cumsum(axis=1).add(-(cum_head - q_headcount), axis=0)).clip(lower=0)).clip(0)) / q_headcount
            v_gem_h_q = (share_h_q * building_class_vuln.loc[hazard]).sum(axis=1, skipna=True)
            v_gem.loc[(slice(None), quantile), hazard] = v_gem_h_q.values
    v_gem = v_gem.stack().rename('v')
    v_gem = v_gem.reorder_levels(['iso3', 'hazard', 'income_cat']).sort_index()

    # compute total vulnerability per country as the weighted sum of vulnerability classes
    vulnerability_tot = building_classes.mul(building_class_vuln.stack(), axis=1).T.groupby(level='hazard').sum().T
    vulnerability_tot = vulnerability_tot.stack().rename('tot').to_frame()
    vulnerability_tot.columns.name = 'income_cat'
    vulnerability_tot = vulnerability_tot.stack()
    vulnerability_tot = vulnerability_tot.reorder_levels(['iso3', 'hazard', 'income_cat']).sort_index()
    vulnerability_tot = vulnerability_tot.rename('v')

    # always use GEM vulnerability as the baseline
    vulnerability_ = v_gem.copy()

    # use GMD data to distribute GEM national vulnerability
    if use_gmd_to_distribute:
        # load vulnerability distribution as per GMD
        vuln_distr = pd.read_excel(os.path.join(root_dir_, "inputs/raw/", gmd_vulnerability_distribution_path), sheet_name='Data')
        vuln_distr = vuln_distr.loc[vuln_distr.groupby('code')['year'].idxmax()]
        vuln_distr.rename(
            {'code': 'iso3', 'ratio1': 'q1', 'ratio2': 'q2', 'ratio3': 'q3', 'ratio4': 'q4', 'ratio5': 'q5'},
            axis=1, inplace=True
        )
        vuln_distr = vuln_distr.set_index('iso3')[['q1', 'q2', 'q3', 'q4', 'q5']]

        # fill missing countries in the GMD data with average
        if fill_missing_gmd_with_country_average:
            missing_index = np.setdiff1d(vulnerability_.index.get_level_values('iso3').unique(), vuln_distr.index)
            vuln_distr = pd.concat((vuln_distr, pd.DataFrame(index=pd.Index(missing_index, name='iso3'),
                                                             columns=vuln_distr.columns, data=np.nan)), axis=0)
            vuln_distr.fillna(vuln_distr.mean(axis=0), inplace=True)
        vuln_distr.columns.name = 'income_cat'
        vuln_distr = vuln_distr.stack().rename('v_rel').sort_index()

        # compute vulnerability per income quintile as the product of national-level vulnerability and the relative
        # vulnerability distribution
        vulnerability_.update((vulnerability_tot.droplevel('income_cat') * vuln_distr).rename('v'))

        # merge vulnerability with the maximum (fragile or from GEM) and minimum (robust or from GEM) vulnerability, as
        # well as the extreme case of vulnerability; then, clip with the min / max vulnerability
        if vulnerability_bounds:
            if vulnerability_bounds == 'class_extremes':
                vulnerability_ = pd.merge(
                    vulnerability_,
                    building_class_vuln.rename({'fragile': 'v_max', 'robust': 'v_min'}, axis=1)[['v_min', 'v_max']],
                    left_index=True, right_index=True, how='left'
                )
            elif vulnerability_bounds == 'gem_extremes':
                v_gem_minmax = v_gem.unstack('income_cat')
                v_gem_minmax = v_gem_minmax.rename({'q1': 'v_max', 'q5': 'v_min'}, axis=1)[['v_min', 'v_max']]
                vulnerability_ = pd.merge(vulnerability_, v_gem_minmax, left_index=True, right_index=True, how='left')
            else:
                raise ValueError(f"Unknown value for vulnerability_bounds: {vulnerability_bounds}")
            vulnerability_ = vulnerability_['v'].clip(lower=vulnerability_['v_min'], upper=vulnerability_['v_max'])

            # recompute vulnerability_tot due to clipping
            vulnerability_tot = vulnerability_.groupby(['iso3', 'hazard']).mean().to_frame()
            vulnerability_tot['income_cat'] = 'tot'
            vulnerability_tot = vulnerability_tot.set_index('income_cat', append=True).squeeze()
        elif vulnerability_[vulnerability_ > 1].any():
            print(f"Warning. No vulnerability bounds provided. {len(vulnerability_[vulnerability_ > 1])} entries with"
                  f"excess vulnerability values > 1!")
    vulnerability_ = pd.concat((vulnerability_, vulnerability_tot), axis=0).sort_index()
    return vulnerability_


def compute_borrowing_ability(root_dir_, any_to_wb_, finance_preparedness_=None, cat_ddo_filepath="CatDDO/catddo.xlsx",
                              verbose=True, force_recompute=True):
    outpath = os.path.join(root_dir_, "inputs/processed/borrowing_ability.csv")
    if not force_recompute and os.path.exists(outpath):
        print("Loading borrowing ability data from file...")
        borrowing_ability_ = pd.read_csv(outpath, index_col='iso3')
        return borrowing_ability_
    print("Computing borrowing ability...")
    credit_ratings = load_credit_ratings(root_dir_, any_to_wb_, verbose=verbose)
    borrowing_ability_ = credit_ratings.rename('borrowing_ability')
    if finance_preparedness_ is not None:
        borrowing_ability_ = pd.concat((borrowing_ability_, finance_preparedness_), axis=1).mean(axis=1).rename('borrowing_ability')
    catddo_countries = pd.concat(pd.read_excel(os.path.join(root_dir_, "inputs/raw/", cat_ddo_filepath), sheet_name=[0, 1, 2], header=0), axis=0)
    catddo_countries = df_to_iso3(catddo_countries, 'Country', verbose_=verbose).iso3.unique()
    catddo_countries = pd.Series(index=pd.Index(catddo_countries, name='iso3'), data=1, name='contingent_countries')
    borrowing_ability_ = pd.concat((borrowing_ability_, catddo_countries), axis=1)
    borrowing_ability_ = borrowing_ability_.borrowing_ability.fillna(borrowing_ability_.contingent_countries)
    borrowing_ability_.to_csv(outpath)
    return borrowing_ability_


def load_hfa_data(root_dir_):
    # HFA (Hyogo Framework for Action) data to assess the role of early warning system
    # 2015 hfa
    hfa15 = pd.read_csv(os.path.join(root_dir_, "inputs/raw/HFA/HFA_all_2013_2015.csv"), index_col='ISO 3')
    # READ THE LAST HFA DATA
    hfa_newest = pd.read_csv(os.path.join(root_dir_, "inputs/raw/HFA/HFA_all_2011_2013.csv"), index_col='ISO 3')
    # READ THE PREVIOUS HFA DATA
    hfa_previous = pd.read_csv(os.path.join(root_dir_, "inputs/raw/HFA/HFA_all_2009_2011.csv"), index_col='ISO 3')
    # most recent values... if no 2011-2013 reporting, we use 2009-2011

    # concat to harmonize indices
    hfa_oldnew = pd.concat([hfa_newest, hfa_previous, hfa15], axis=1, keys=['new', 'old', "15"])
    hfa_data = hfa_oldnew["15"].fillna(hfa_oldnew["new"].fillna(hfa_oldnew["old"]))

    # access to early warning normalized between zero and 1.
    # P2-C3: "Early warning systems are in place for all major hazards, with outreach to communities"
    hfa_data["ew"] = 1 / 5 * hfa_data["P2-C3"]

    # q_s in the report, ability to scale up support to affected population after the disaster
    # normalized between zero and 1
    # P4C2: Do social safety nets exist to increase the resilience of risk prone households and communities?
    # P5C2: Disaster preparedness plans and contingency plans are in place at all administrative levels, and regular
    # training drills and rehearsals are held to test and develop disaster response programs.
    # P4C5: Disaster risk reduction measures are integrated into post‐disaster recovery and rehabilitation processes.
    hfa_data["prepare_scaleup"] = (hfa_data["P4-C2"] + hfa_data["P5-C2"] + hfa_data["P4-C5"]) / 3 / 5

    # P5-C3: "Financial reserves and contingency mechanisms are in place to enable effective response and recovery when
    # required"
    hfa_data["finance_pre"] = (1 + hfa_data["P5-C3"]) / 6

    hfa_data = hfa_data[["ew", "prepare_scaleup", "finance_pre"]]
    hfa_data.fillna(0, inplace=True)
    hfa_data.index.name = 'iso3'

    return hfa_data


def load_wrp_data(any_to_wb_, wrp_data_path_, root_dir_, outfile=None, verbose=True):
    # previously used HFA indicators for "ability to scale up the support to affected population after the disaster":
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

    # previously used HFA indicators for early warning:
    # P2-C3: "Early warning systems are in place for all major hazards, with outreach to communities"

    # early warning related questions (see lrf_wrp_2021_full_data_dictionary.xlsx):
    # Q19A	Received Warning About Disaster From Internet/Social Media
    # Q19B	Received Warning About Disaster From Local Government or Police
    # Q19C	Received Warning About Disaster From Radio, TV, or Newspapers
    # Q19D	Received Warning About Disaster From Local Community Organization
    early_warning_cols = ['Q19A', 'Q19B', 'Q19C', 'Q19D']

    all_cols = cat_preparedness_cols + early_warning_cols

    # read data
    wrp_data_ = pd.read_csv(str(os.path.join(root_dir_, 'inputs/raw/', wrp_data_path_)), dtype=object)[['Country', 'WGT', 'Year'] + all_cols]
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

    indicators = df_to_iso3(indicators.reset_index(), 'Country', any_to_wb_, verbose_=verbose)
    indicators = indicators.set_index('iso3').drop('Country', axis=1)

    if outfile is not None:
        indicators.to_csv(outfile)

    return indicators


def load_credit_ratings(root_dir_, any_to_wb_, tradingecon_ratings_path="credit_ratings/2023-12-13_tradingeconomics_ratings.csv",
                        cia_ratings_raw_path="credit_ratings/2024-01-30_cia_ratings_raw.txt",
                        ratings_scale_path="credit_ratings/credit_ratings_scale.csv",
                        verbose=True):

    # Trading Economics ratings
    if verbose:
        print("Warning. Check that [date]_tradingeconomics_ratings.csv is up to date. If not, download it from "
              "http://www.tradingeconomics.com/country-list/rating")
    te_ratings = pd.read_csv(os.path.join(root_dir_, "inputs/raw/", tradingecon_ratings_path), dtype="str", encoding="utf8", na_values=['NR'])
    te_ratings = te_ratings.dropna(how='all')
    te_ratings = te_ratings[["country", "S&P", "Moody's"]]

    # drop EU
    te_ratings = te_ratings[te_ratings.country != 'European Union']

    # rename Congo to Congo, Dem. Rep.
    te_ratings.replace({'Congo': 'Congo, Dem. Rep.'}, inplace=True)

    # change country name to iso3
    te_ratings = df_to_iso3(te_ratings, "country", any_to_wb_, verbose_=verbose)
    te_ratings = te_ratings.set_index("iso3").drop("country", axis=1)

    # make lower case and strip
    te_ratings = te_ratings.map(lambda x: str.strip(x).lower() if type(x) is str else x)

    # CIA ratings
    if verbose:
        print("Warning. Check that [date]_cia_ratings_raw.csv is up to date. If not, copy the text from "
                "https://www.cia.gov/the-world-factbook/field/credit-ratings/ into [date]_cia_ratings_raw.csv")
    cia_ratings_raw = open(os.path.join(root_dir_, "inputs/raw/", cia_ratings_raw_path), 'r').read().strip().split('\n')
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
    cia_ratings = df_to_iso3(cia_ratings, "country", any_to_wb_, verbose_=verbose)
    cia_ratings = cia_ratings.set_index(["iso3", "agency"]).drop("country", axis=1).squeeze().unstack('agency')

    # merge ratings
    ratings = pd.merge(te_ratings, cia_ratings, left_index=True, right_index=True, how='outer')
    for agency in np.intersect1d(te_ratings.columns, cia_ratings.columns):
        ratings[agency] = ratings[agency + "_x"].fillna(ratings[agency + "_y"])
        ratings.drop([agency + "_x", agency + "_y"], axis=1, inplace=True)

    # Transforms ratings letters into 1-100 numbers
    rating_scale = pd.read_csv(os.path.join(root_dir_, "inputs/raw/", ratings_scale_path))
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


def load_disaster_preparedness_data(root_dir_, any_to_wb_, include_hfa_data=True, guess_missing_countries=True, ew_year=2018, ew_decade=None,
                                    income_groups_file="WB_country_classification/country_classification.xlsx",
                                    forecast_accuracy_file_yearly="Linsenmeier_Shrader_forecast_accuracy/Linsenmeier_Shrader_forecast_accuracy_per_country_yearly.csv",
                                    forecast_accuracy_file_decadal="Linsenmeier_Shrader_forecast_accuracy/Linsenmeier_Shrader_forecast_accuracy_per_country_decadal.csv",
                                    force_recompute=True, verbose=True):
    outpath = os.path.join(root_dir_, "inputs/processed/disaster_preparedness.csv")
    if not force_recompute and os.path.exists(outpath) and (ew_year == 2018) and (ew_decade is None):
        print("Loading disaster preparedness data from file...")
        disaster_preparedness_ = pd.read_csv(outpath, index_col='iso3')
        return disaster_preparedness_

    print("Recomputing disaster preparedness data...")
    if ew_decade is not None or ew_year != 2018:
        print("Recomputing disaster preparedness because of change in EW year or decade.")
    disaster_preparedness_ = load_wrp_data(any_to_wb_, "WRP/lrf_wrp_2021_full_data.csv.zip", root_dir_,
                                           verbose=verbose)
    if include_hfa_data:
        # read HFA data
        hfa_data = load_hfa_data(root_dir_=root_dir_)
        # merge HFA and WRP data: mean of the two data sets where both are available
        disaster_preparedness_['ew'] = pd.concat((hfa_data['ew'], disaster_preparedness_['ew']), axis=1).mean(axis=1)
        disaster_preparedness_['prepare_scaleup'] = pd.concat(
            (hfa_data['prepare_scaleup'], disaster_preparedness_['prepare_scaleup']), axis=1
        ).mean(axis=1)
        disaster_preparedness_ = pd.merge(disaster_preparedness_, hfa_data.finance_pre, left_index=True, right_index=True, how='outer')
    if guess_missing_countries:
        income_groups = pd.read_excel(os.path.join(root_dir_, "inputs/raw/", income_groups_file), header=0)[["Code", "Region", "Income group"]]
        income_groups = income_groups.dropna().rename({'Code': 'iso3'}, axis=1)
        income_groups = income_groups.set_index('iso3').squeeze()
        income_groups.loc['VEN'] = ['Latin America & Caribbean', 'Upper middle income']
        if verbose:
            print("Manually setting region and income group for VEN to 'LAC' and 'UM'.")
        merged = pd.merge(disaster_preparedness_, income_groups, left_index=True, right_index=True, how='outer')
        fill_values = merged.groupby(['Region', 'Income group']).mean()
        fill_values = fill_values.fillna(merged.drop('Region', axis=1).groupby('Income group').mean())
        merged = merged.fillna(merged.apply(lambda x: fill_values.loc[(x['Region'], x['Income group'])] if not x[['Region', 'Income group']].isna().any() else x, axis=1))
        disaster_preparedness_ = merged[disaster_preparedness_.columns]
    if ew_year is not None and ew_decade == '':
        forecast_accuracy = pd.read_csv(os.path.join(root_dir_, "inputs/raw/", forecast_accuracy_file_yearly), index_col=[0, 1]).squeeze()
        if ew_year not in forecast_accuracy.index.get_level_values('year'):
            raise ValueError(f"Forecast accuracy data not available for year {ew_year}.")
        ew_factor = forecast_accuracy.xs(ew_year, level='year') / forecast_accuracy.xs(forecast_accuracy.index.get_level_values('year').max(), level='year')
    elif ew_year is None and ew_decade != '':
        forecast_accuracy = pd.read_csv(os.path.join(root_dir_, "inputs/raw/", forecast_accuracy_file_decadal), index_col=[0, 1]).squeeze()
        if ew_decade not in forecast_accuracy.index.get_level_values('decade'):
            raise ValueError(f"Forecast accuracy data not available for decade {ew_decade}.")
        ew_factor = forecast_accuracy.xs(ew_decade, level='decade') / forecast_accuracy.xs('2011-2020', level='decade')
    else:
        raise ValueError("Exactly one of ew_year or ew_decade must be set.")
    disaster_preparedness_.loc[:, 'ew'] = (disaster_preparedness_['ew'] * ew_factor).copy()
    disaster_preparedness_ = disaster_preparedness_.dropna()
    disaster_preparedness_.to_csv(outpath)
    return disaster_preparedness_


def gather_data(intermediate_dir_=None, force_recompute_=False, hazard_protection_='FLOPROS', reduction_vul_=.2,
                income_elasticity_eta_=1.5, discount_rate_rho_=.06, max_increased_spending_=.05,
                fa_threshold_=.9, axfin_impact_=.1, no_exposure_bias_=False, reconstruction_capital_='prv',
                ew_year_=2018, ew_decade_='', scale_self_employment_=1, scale_non_diversified_income_=1,
                min_diversified_share_=0, scale_gdp_pc_pp_=1, pol_opt_=None, verbose=True, countries=None,
                scale_liquidity_=1, hazards_='all'):
    root_dir = os.path.dirname(os.path.abspath(__file__))
    any_to_wb, iso3_to_wb, iso2_iso3 = get_country_name_dicts(root_dir)

    timestamp = time()
    # read WB data
    wb_data_macro, wb_data_cat_info = get_wb_data(
        root_dir=root_dir,
        include_remittances=True,
        use_additional_data=True,
        drop_incomplete=True,
        force_recompute=force_recompute_,
        verbose=verbose
    )

    # print duration
    if force_recompute_:
        print(f"Duration: {time() - timestamp:.2f} seconds.\n")
        timestamp = time()

    # load liquidity from Findex data
    liquidity_and_axfin = load_findex_liquidity_and_axfin(
        root_dir_=root_dir,
        any_to_wb_=any_to_wb,
        force_recompute_=force_recompute_,
        verbose=verbose
    )
    liquidity_and_axfin['liquidity'] *= scale_liquidity_
    if force_recompute_:
        print(f"Duration: {time() - timestamp:.2f} seconds.\n")
        timestamp = time()

    cat_info = pd.merge(wb_data_cat_info, liquidity_and_axfin, left_index=True, right_index=True, how='inner')

    # scale GDPpc
    wb_data_macro['gdp_pc_pp'] *= scale_gdp_pc_pp_

    n_quantiles = len(wb_data_cat_info.index.get_level_values('income_cat').unique())

    disaster_preparedness = load_disaster_preparedness_data(
        root_dir_=root_dir,
        any_to_wb_=any_to_wb,
        include_hfa_data=True,
        guess_missing_countries=True,
        ew_year=ew_year_,
        ew_decade=ew_decade_,
        force_recompute=force_recompute_,
        verbose=verbose
    )
    if force_recompute_:
        print(f"Duration: {time() - timestamp:.2f} seconds.\n")
        timestamp = time()

    # compute country borrowing ability
    borrowing_ability = compute_borrowing_ability(
        root_dir_=root_dir,
        any_to_wb_=any_to_wb,
        finance_preparedness_=disaster_preparedness.finance_pre,
        verbose=verbose,
        force_recompute=force_recompute_,
    )
    if force_recompute_:
        print(f"Duration: {time() - timestamp:.2f} seconds.\n")
        timestamp = time()

    # load average productivity of capital
    avg_prod_k = gather_capital_data(
        root_dir_=root_dir,
        force_recompute=force_recompute_
    ).avg_prod_k
    if force_recompute_:
        print(f"Duration: {time() - timestamp:.2f} seconds.\n")
        timestamp = time()

    # Load capital shares
    capital_shares = calc_reconstruction_share_sigma(
        root_dir_=root_dir,
        any_to_wb_=any_to_wb,
        reconstruction_capital_=reconstruction_capital_,
        scale_self_employment=scale_self_employment_,
        verbose=verbose,
        force_recompute=force_recompute_,
    )
    if force_recompute_:
        print(f"Duration: {time() - timestamp:.2f} seconds.\n")
        timestamp = time()

    # compute exposure and adjust vulnerability (per income category) for excess exposure
    exposure_fa, vulnerability_adjusted_, vulnerability_unadjusted_ = compute_exposure_and_vulnerability(
        root_dir_=root_dir,
        fa_threshold_=fa_threshold_,
        n_quantiles=n_quantiles,
        force_recompute_=force_recompute_,
        verbose=verbose,
        apply_exposure_bias=not no_exposure_bias_,
        population_data=wb_data_macro['pop'],
    )
    if force_recompute_:
        print(f"Duration: {time() - timestamp:.2f} seconds.\n")
        timestamp = time()

    # expand the early warning to all hazard rp's, income categories, and countries; set to 0 for specific hazrads
    # (Earthquake and Tsunami)
    early_warning_per_hazard = get_early_warning_per_hazard(exposure_fa.index, disaster_preparedness.ew)

    # concatenate exposure, vulnerability and early warning into one dataframe
    hazard_ratios = pd.concat((exposure_fa, vulnerability_adjusted_, early_warning_per_hazard), axis=1).dropna()
    hazard_ratios["v_ew"] = hazard_ratios["v"] * (1 - reduction_vul_ * hazard_ratios["ew"])

    # get data per income categories
    cat_info, tau_tax = get_cat_info_and_tau_tax(
        cat_info_=cat_info,
        wb_data_macro_=wb_data_macro,
        avg_prod_k_=avg_prod_k,
        n_quantiles_=n_quantiles,
        axfin_impact_=axfin_impact_,
        scale_non_diversified_income_=scale_non_diversified_income_,
        min_diversified_share_=min_diversified_share_
    )
    if force_recompute_:
        print(f"Duration: {time() - timestamp:.2f} seconds.\n")
        timestamp = time()

    hazard_protection = load_protection(
        index_=hazard_ratios.index,
        root_dir_=root_dir,
        protection_data=hazard_protection_,
        min_rp=0
    )
    if force_recompute_:
        print(f"Duration: {time() - timestamp:.2f} seconds.\n")
        timestamp = time()

    macro = wb_data_macro
    macro = macro.join(disaster_preparedness, how='left')
    macro = macro.join(borrowing_ability, how='left')
    macro = macro.join(avg_prod_k, how='left')
    macro = macro.join(tau_tax, how='left')
    macro = macro.join(capital_shares, how='left')
    macro['rho'] = discount_rate_rho_
    macro['max_increased_spending'] = max_increased_spending_
    macro['income_elasticity_eta'] = income_elasticity_eta_

    # clean and harmonize data frames
    macro.dropna(inplace=True)
    cat_info.dropna(inplace=True)
    hazard_ratios.dropna(inplace=True)
    hazard_protection.dropna(inplace=True)

    common_regions = [c for c in macro.index if c in cat_info.index and c in hazard_ratios.index and c in hazard_protection.index]
    if countries is not None:
        countries = countries.split('+')
        if len(np.intersect1d(countries, common_regions)) == 0:
            print("None of the selected countries found in data. Keeping all countries.")
        else:
            common_regions = np.intersect1d(countries, common_regions)
    macro = macro.loc[common_regions]
    cat_info = cat_info.loc[common_regions]
    hazard_ratios = hazard_ratios.loc[common_regions]
    hazard_protection = hazard_protection.loc[common_regions]

    if hazards_ != 'all':
        hazard_ratios = hazard_ratios[hazard_ratios.index.get_level_values('hazard').isin(hazards_.split('+'))]
        hazard_protection = hazard_protection[hazard_protection.index.get_level_values('hazard').isin(hazards_.split('+'))]

    if pol_opt_ == '':
        pol_name = None
        pol_opt_param = None
    else:
        pol_name = pol_opt_.split('+')[0]
        if pol_name == 'None':
            pol_name = None
        pol_opt_param = pol_opt_.split('+')[1]
        if pol_opt_param == 'None':
            pol_opt_param = None
        else:
            pol_opt_param = float(pol_opt_param)

    # apply policy
    scenario_macro, scenario_cat_info, scenario_hazard_ratios, pol_name, pol_desc = apply_policy(
        macro_=macro.copy(deep=True),
        cat_info_=cat_info.copy(deep=True),
        hazard_ratios_=hazard_ratios.copy(deep=True),
        policy_name=pol_name,
        policy_opt=pol_opt_param
    )


    # Save all data
    print(scenario_macro.shape[0], 'countries in analysis')

    if intermediate_dir_ is not None:
        sim_name = pol_name
        sim_name += f"_EW-{ew_year_ if ew_decade_ == '' else ew_decade_}" if (
                    ew_year_ != 2018 or ew_decade_ != '') else ''
        sim_name += '_no-EB' if no_exposure_bias_ else ''
        sim_name += f"_scale_self_employment_{scale_self_employment_}" if scale_self_employment_ != 1. else ''
        sim_name += f"_scale_non_diversified_income_{scale_non_diversified_income_}" if scale_non_diversified_income_ != 1 else ''
        sim_name += f"_min_diversified_share-{min_diversified_share_}" if min_diversified_share_ != 0 else ''
        sim_name += f"_scale_GDPpc_{scale_gdp_pc_pp_}" if scale_gdp_pc_pp_ != 1 else ''

        outpath = str(os.path.join(intermediate_dir_, 'scenarios', sim_name).replace(' ', '_'))
        if not os.path.exists(outpath):
            os.makedirs(outpath)

        # save adjusted vulnerability by country and income category
        pd.merge(
            vulnerability_unadjusted_.rename({'v': 'v_unadj'}),
            pd.merge(vulnerability_adjusted_.rename({'v': 'v_adj'}), scenario_hazard_ratios.v_ew,
                     left_index=True, right_index=True, how='inner'),
            left_index=True, right_index=True, how='inner',
        ).to_csv(
            os.path.join(outpath, "scenario__vulnerability.csv"),
            encoding="utf-8",
            header=True
        )

        # save protection by country and hazard
        hazard_protection.to_csv(
            os.path.join(outpath, "scenario__hazard_protection.csv"),
            encoding="utf-8",
            header=True
        )

        # save macro-economic country economic data
        scenario_macro.to_csv(
            os.path.join(outpath, "scenario__macro.csv"),
            encoding="utf-8",
            header=True
        )

        # save consumption, access to finance, gamma, capital, exposure, early warning access by country and income category
        scenario_cat_info.to_csv(
            os.path.join(outpath, "scenario__cat_info.csv"),
            encoding="utf-8",
            header=True
        )

        # save exposure, vulnerability, and access to early warning by country, hazard, return period, income category
        scenario_hazard_ratios.to_csv(
            os.path.join(outpath, "scenario__hazard_ratios.csv"),
            encoding="utf-8",
            header=True
        )
    return scenario_macro, scenario_cat_info, scenario_hazard_ratios, hazard_protection


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Script parameters')
    parser.add_argument('--hazard_protection', type=str, default='FLOPROS', help='Select hazard protection data.')
    parser.add_argument('--drop_unused_data', type=bool, default=True, help='Drop unused data')
    parser.add_argument('--reduction_vul', type=float, default=0.2, help='Reduction in vulnerability')
    parser.add_argument('--income_elasticity_eta', type=float, default=1.5, help='Income elasticity')
    parser.add_argument('--discount_rate_rho', type=float, default=0.06, help='Discount rate')
    parser.add_argument('--max_increased_spending', type=float, default=0.05, help='Maximum support')
    parser.add_argument('--fa_threshold', type=float, default=.9, help='FA threshold')
    parser.add_argument('--axfin_impact', type=float, default=0.1, help='Increase of the fraction of diversified '
                                                                        'income through financial inclusion')
    parser.add_argument('--root_dir', type=str, default=os.getcwd(), help='Root directory')
    parser.add_argument('--force_recompute', action='store_true', help='Force recomputation of data.')
    parser.add_argument('--no_exposure_bias', action='store_true', help='Do not apply exposure bias.')
    parser.add_argument('--reconstruction_capital', type=str, default='prv', help="Fraction of the reconstruction"
                                                                                  " capital that needs to be paid for "
                                                                                  "by households. Options: 'prv' / "
                                                                                  "'prv_oth'.")
    parser.add_argument('--scale_self_employment', type=float, default=1., help='Factor to scale self-employment rate.')
    parser.add_argument('--scale_non_diversified_income', type=float, default=1, help='Scale the share of income that does not come from diversified sources.')
    parser.add_argument('--min_diversified_share', type=float, default=0, help='Set lower bound for the share of income diversified sources.')
    parser.add_argument('--scale_gdp_pc_pp', type=float, default=1, help='Scale per capita GDP.')
    parser.add_argument('--ew_year', type=int, default=2018, help='Year of early warning data')
    parser.add_argument('--ew_decade', type=str, default=None, help='Decade of early warning data')
    parser.add_argument('--pol_opt', type=str, default='None+None', help='Policy selection')
    parser.add_argument('--do_not_execute', action='store_true', help='Do not execute the script. '
                                                                      'Only load the parameters.')
    parser.add_argument('--verbose', action='store_true', help='Print additional information')
    parser.add_argument('--out_dir', type=str, default=None, help='Output directory')
    parser.add_argument('--countries', type=str, default=None, help='List of countries to include in the analysis.')
    parser.add_argument('--scale_liquidity', type=float, default=1, help='Reduce available liquidity by some factor.')
    parser.add_argument('--hazards', type=str, default='all', help='Hazards to include in the analysis.')
    args = parser.parse_args()

    if not args.do_not_execute:
        scenario_macro_data, scenario_cat_info_data, scenario_hazard_ratios_data, scenario_hazard_protection_data = gather_data(
            force_recompute_=args.force_recompute, hazard_protection_=args.hazard_protection,
            reduction_vul_=args.reduction_vul, income_elasticity_eta_=args.income_elasticity_eta,
            discount_rate_rho_=args.discount_rate_rho, max_increased_spending_=args.max_increased_spending,
            fa_threshold_=args.fa_threshold, axfin_impact_=args.axfin_impact,
            no_exposure_bias_=args.no_exposure_bias, reconstruction_capital_=args.reconstruction_capital,
            ew_year_=args.ew_year if args.ew_year != -1 else None, ew_decade_=args.ew_decade,
            scale_self_employment_=args.scale_self_employment,
            scale_non_diversified_income_=args.scale_non_diversified_income,
            min_diversified_share_=args.min_diversified_share, scale_gdp_pc_pp_=args.scale_gdp_pc_pp,
            pol_opt_=args.pol_opt, verbose=args.verbose, countries=args.countries,
            scale_liquidity_=args.scale_liquidity, hazards_=args.hazards)
