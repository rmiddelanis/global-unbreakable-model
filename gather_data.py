# This script provides data input for the resilience indicator multihazard model.
# The script was developed by Adrien Vogt-Schilb and improved by Jinqiang Chen and Robin Middelanis.
import argparse
import copy

from gather_gir_data import load_gir_hazard_losses
from lib_gather_data import *
from apply_policy import *
import pandas as pd
from pandas import isnull
from lib_gather_data import replace_with_warning, get_country_name_dicts

# TODO: switch to continuous use of ISO3 for better consistency


def get_cat_info_and_tau_tax(wb_data_, poverty_head_, avg_prod_k_):
    axfin = wb_data_[['axfin_p', 'axfin_r']].rename({'axfin_p': 'poor', 'axfin_r': 'nonpoor'}, axis=1).stack(dropna=False)
    axfin.name = 'axfin'
    axfin.index.names = ['country', 'income_cat']
    social = wb_data_[['social_p', 'social_r']].rename({'social_p': 'poor', 'social_r': 'nonpoor'}, axis=1).stack(dropna=False)
    social.name = 'social'
    social.index.names = ['country', 'income_cat']
    cat_info_ = pd.concat([axfin, social], axis=1)

    cat_info_.loc[(slice(None), 'poor'), "n"] = poverty_head_
    cat_info_.loc[(slice(None), 'nonpoor'), "n"] = 1 - poverty_head_

    c_poor = wb_data_.share1 / poverty_head_ * wb_data_.gdp_pc_pp
    c_poor.name = 'poor'
    c_nonpoor = (1 - wb_data_.share1) / (1 - poverty_head_) * wb_data_.gdp_pc_pp
    c_nonpoor.name = 'nonpoor'
    c = pd.concat([c_poor, c_nonpoor], axis=1).stack(dropna=False)
    c.name = 'c'

    cat_info_["c"] = c

    # compute tau tax and gamma_sp from social_poor and social_nonpoor.
    tau_tax_, cat_info_["gamma_SP"] = social_to_tx_and_gsp(econ_scope, cat_info_)

    # compute capital per income category
    cat_info_["k"] = (1 - cat_info_.social) * cat_info_.c / ((1 - tau_tax_) * avg_prod_k_)

    # TODO: fa and shew were previously also included in cat_info. IMO averaging here makes no sense, leaving it out
    #  for now.
    # cat_info["fa"] = hazard_ratios.fa.groupby(level=["country", "income_cat"]).mean()
    # cat_info["shew"] = hazard_ratios.shew.drop("earthquake", level="hazard").groupby(
    #     level=["country", "income_cat"]).mean()
    return cat_info_, tau_tax_


# TODO: discuss protection approach with Bramka
# compare to gather_data_old.py: before, protection was simply set to rp=1, since no_protection==True
def load_protection(index_, protection_data="FLOPROS", min_rp=1,
                    flopros_protection_file="protection_national_from_flopros.csv",
                    protection_level_assumptions_file="protection_level_assumptions.csv",
                    income_groups_file="income_groups.csv"):
    # TODO: is this applied to all hazards or just floods?
    # TODO: use FLOPROS V1 for protection; try to infer missing countries from FLOPROS based on GDP-protection correlation
    prot = pd.Series(index=index_, name="protection", data=min_rp).reset_index()
    if protection_data == 'FLOPROS':
        prot_data = load_input_data(root_dir, flopros_protection_file, index_col="country").squeeze()
    elif protection_data == 'country_income':  # assumed a function of the country's income group
        # note: "protection_level_assumptions.csv" can be found in orig_inputs.
        prot_assumptions = load_input_data(root_dir, protection_level_assumptions_file,
                                           index_col="Income group").squeeze()
        prot_data = load_input_data(root_dir, income_groups_file, header=4, index_col=2)["Income group"].dropna()
        prot_data.replace(prot_assumptions, inplace=True)
    else:
        raise ValueError("Unknown protection_data: {}".format(protection_data))
    prot_data.replace(any_to_wb, inplace=True)
    prot.protection = prot.country.apply(lambda x: max(min_rp, prot_data.loc[x] if x in prot_data.index else 0))
    prot.set_index(index_.names, inplace=True)
    return prot


def get_early_warning_per_hazard(index_, ew_per_country_, no_ew_hazards="Earthquake"):
    common_idx = index_.drop(np.setdiff1d(index_.get_level_values('country').unique(), ew_per_country_.index))
    ew_per_hazard_ = pd.Series(
        index=common_idx,
        data=ew_per_country_.loc[common_idx.get_level_values('country')].values,
        name='ew'
    )
    ew_per_hazard_.index.names = index_.names
    ew_per_hazard_.loc[:, no_ew_hazards.split('+')] = 0
    ew_per_hazard_.fillna(0, inplace=True)
    return ew_per_hazard_


def apply_poverty_exposure_bias(exposure_fa_, use_avg_pe_, poverty_headcount_, population_data_=None,
                                peb_povmaps_filepath_="PEB_flood_povmaps.xlsx",
                                peb_deltares_filepath_="PEB_wb_deltares.csv", peb_hazards_="Flood+Storm surge"):
    # Exposure bias from WB povmaps study
    exposure_bias_wb = load_input_data(root_dir, peb_povmaps_filepath_)[["iso", "peb"]].dropna()
    exposure_bias_wb = exposure_bias_wb.set_index(exposure_bias_wb.iso.replace(iso3_to_wb)).peb - 1
    exposure_bias_wb.index.name = 'country'

    # Exposure bias from older WB DELTARES study
    exposure_bias_deltares = load_input_data(root_dir, peb_deltares_filepath_, skiprows=[0, 1, 2],
                                             usecols=["Country", "Nation-wide"])

    # Replace with warning is used for columns, for index set_index is needed.
    exposure_bias_deltares["country"] = replace_with_warning(exposure_bias_deltares["Country"], any_to_wb)
    exposure_bias_deltares = exposure_bias_deltares.set_index('country').drop('Country', axis=1)
    exposure_bias_deltares.rename({'Nation-wide': 'peb'}, axis=1, inplace=True)
    exposure_bias_deltares.dropna(inplace=True)

    # Completes with bias from previous study when pov maps not available.
    peb = pd.merge(exposure_bias_wb, exposure_bias_deltares, how='outer', left_index=True, right_index=True)
    peb['peb_x'].fillna(peb['peb_y'], inplace=True)
    peb.drop('peb_y', axis=1, inplace=True)
    peb.rename({'peb_x': 'peb'}, axis=1, inplace=True)
    peb = peb.peb

    # incorporates exposure bias, but only for (riverine) flood and surge, and gets an updated fa for income_cats
    # fa_hazard_cat = broadcast_simple(fa_guessed_gar,index=income_cats) #fraction of assets affected per hazard and
    # income categories fa_with_pe = concat_categories(fa_guessed_gar*(1+pe),
    # fa_guessed_gar*(1-df.pov_head*(1+pe))/(1-df.pov_head), index=income_cats)
    # NB: fa_guessed_gar*(1+pe) gives f_p^a and fa_guessed_gar*(1-df.pov_head*(1+pe))/(1-df.pov_head) gives f_r^a.TESTED
    # fa_guessed_gar.update(fa_with_pe) #updates fa_guessed_gar where necessary

    # selects just flood and surge
    peb_hazards_ = peb_hazards_.split("+")

    fa_with_peb = exposure_fa_.copy(deep=True)
    fa_with_peb = fa_with_peb.reset_index()
    fa_with_peb['peb'] = fa_with_peb.country.apply(lambda x: peb.loc[x] if x in peb.index else np.nan)

    # if use_avg_pe, use averaged pe from global data for countries that don't have PE. Otherwise use 0.
    if use_avg_pe_:
        if population_data_ is not None:
            fa_with_peb.peb = fa_with_peb.peb.fillna(wavg(peb, population_data_))
        else:
            raise Exception("Population data not available. Cannot use average PE.")
    else:
        peb.fillna(0, inplace=True)

    selector = (fa_with_peb.hazard.isin(peb_hazards_)) & (fa_with_peb.income_cat == 'poor')
    fa_with_peb.loc[selector, "peb_factor"] = 1 + fa_with_peb.loc[selector, "peb"]
    selector = (fa_with_peb.hazard.isin(peb_hazards_)) & (fa_with_peb.income_cat == 'nonpoor')
    fa_with_peb.loc[selector, "peb_factor"] = ((1 - poverty_headcount_ * (1 + fa_with_peb.loc[selector, "peb"]))
                                               / (1 - poverty_headcount_))
    fa_with_peb.peb_factor.fillna(1, inplace=True)
    fa_with_peb['fa'] = fa_with_peb['fa'] * fa_with_peb['peb_factor']

    fa_with_peb = fa_with_peb.set_index(event_level + ['income_cat']).fa

    return fa_with_peb


def compute_exposure_and_adjust_vulnerability(hazard_loss_tot_, vulnerability_, fa_threshold_, event_level_):
    # fa_guessed_gar_ = (hazard_loss_tot_ / vulnerability_.v_tot).dropna().to_frame()
    exposure_fa_guessed_ = (hazard_loss_tot_ / vulnerability_.v_tot).dropna()
    exposure_fa_guessed_.name = 'fa'

    # merge vulnerability with hazard_ratios
    fa_v_merged = pd.merge(
        exposure_fa_guessed_.reset_index(),
        vulnerability_.drop('v_tot', axis=1).rename({'v_poor': 'poor', 'v_rich': 'nonpoor'}, axis=1).reset_index(),
        on=econ_scope
    )

    fa_v_merged = fa_v_merged.set_index(event_level_ + ['fa']).stack()
    fa_v_merged.index.names = event_level_ + ['fa', 'income_cat']
    fa_v_merged.name = 'v'
    fa_v_merged = fa_v_merged.reset_index('fa')

    # clip f_a to fa_threshold; increase v s.th. \Delta K = f_a * v is constant
    if np.any(fa_v_merged.fa > fa_threshold_):
        excess_exposure = fa_v_merged.loc[fa_v_merged.fa > fa_threshold_, 'fa']
        print(f"Exposure values f_a are above fa_threshold for {len(excess_exposure)} countries. Setting them to "
              f"fa_threshold and adjusting vulnerability accordingly.")
        r = excess_exposure / fa_threshold_  # ratio by which fa is above fa_threshold
        r.name = 'r'
        fa_v_merged.loc[excess_exposure.index, 'fa'] = fa_threshold_  # set f_a to fa_threshold
        fa_v_merged.loc[excess_exposure.index, 'v'] = fa_v_merged.loc[excess_exposure.index, 'v'] * r  # increase v s.th. \Delta K = f_a * v does not change
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


def load_vulnerability_data(poverty_headcount_, income_share_poor_,
                            gem_vulnerability_classes_filepath_="GEM_vulnerability/country_vulnerability_classes.csv",
                            aggregate_category_to_vulnerability_filepath_="aggregate_category_to_vulnerability.csv"):
    hous_share_tot = load_input_data(root_dir, gem_vulnerability_classes_filepath_,
                                     index_col="country").drop('iso3', axis=1)

    # matching vulnerability of buildings and people's income and calculate poor's, rich's and country's vulnerability
    hous_cat_vulnerability = load_input_data(root_dir, aggregate_category_to_vulnerability_filepath_, sep=";",
                                             index_col="aggregate_category").squeeze()

    # REMARK: NEED TO BE CHANGED....Stephane I've talked to @adrien_vogt_schilb and don't want you to go over our whole
    # conversation. Here is the thing: in your model, you assume that the bottom 20% of population gets the 20% of buildings
    # of less quality. I don't think it's a fair justification, because normally poor people live in buildings of less
    # quality but in a more crowded way,i.e., it could be the bottom 40% of population get the 10% of buildings of less
    # quality. I think we need to correct this matter. @adrien_vogt_schilb also agreed on that, if he didn't change his
    # opinion. How to do that? I think once we incorporate household data, we can allocate buildings on the decile of
    # households, rather than population. I think it's a more realistic assumption.
    # TODO @Bramka @Stephane --> GMD data?
    p = (hous_share_tot.cumsum(axis=1).add(-poverty_headcount_, axis=0)).clip(lower=0)
    poor = (hous_share_tot - p).clip(lower=0)
    rich = hous_share_tot - poor
    vulnerability_poor_ = ((poor * hous_cat_vulnerability).sum(axis=1, skipna=False) / poverty_headcount_)
    vulnerability_poor_.name = "v_poor"
    vulnerability_rich_ = (rich * hous_cat_vulnerability).sum(axis=1, skipna=False) / (1 - poverty_headcount_)
    vulnerability_rich_.name = "v_rich"

    # vulnerability weighted with income shares
    # TODO: why? shouldn't this be simply (hous_share_tot * hous_cat_vulnerability).sum(axis=1) ?
    vulnerability_tot_ = vulnerability_poor_ * income_share_poor_ + vulnerability_rich_ * (1 - income_share_poor_)
    vulnerability_tot_.name = "v_tot"
    vulnerability_tot_.index.name = "country"

    vulnerability = pd.concat([vulnerability_tot_, vulnerability_poor_, vulnerability_rich_], axis=1)
    return vulnerability


def compute_borrowing_ability(credit_ratings_, finance_preparedness_, cat_ddo_filepath="contingent_finance_countries.csv"):
    borrowing_ability_ = credit_ratings_.add(finance_preparedness_, fill_value=0) / 2
    contingent_countries = load_input_data(root_dir, cat_ddo_filepath).iloc[:, 0].values
    borrowing_ability_.loc[contingent_countries] = 1
    borrowing_ability_.name = 'borrowing_ability'
    return borrowing_ability_


def load_hfa_data():
    # HFA (Hyogo Framework for Action) data to assess the role of early warning system
    # TODO: check Côte d'Ivoire naming in HFA
    # 2015 hfa
    hfa15 = load_input_data(root_dir, "HFA_all_2013_2015.csv")
    hfa15 = hfa15.set_index(replace_with_warning(hfa15["Country name"], any_to_wb))
    # READ THE LAST HFA DATA
    hfa_newest = load_input_data(root_dir, "HFA_all_2011_2013.csv")
    hfa_newest = hfa_newest.set_index(replace_with_warning(hfa_newest["Country name"], any_to_wb))
    # READ THE PREVIOUS HFA DATA
    hfa_previous = load_input_data(root_dir, "HFA_all_2009_2011.csv")
    hfa_previous = hfa_previous.set_index(replace_with_warning(hfa_previous["Country name"], any_to_wb))
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
    hfa_data.index.name = 'country'

    return hfa_data


def load_credit_ratings(credit_ratings_file="credit_ratings_scrapy.csv"):
    print("Warning. Check that credit_ratings_scrapy.csv is up to date. If not, download it from "
          "http://www.tradingeconomics.com/country-list/rating")

    # TODO: Fitch no longer on tradingeconomics.com. Find a different source? Also, here DBRS is not used. Why?
    # drop rows where only all columns are NaN.
    ratings_raw = load_input_data(root_dir, credit_ratings_file, dtype="str", encoding="utf8", na_values=['NR']).dropna(
        how="all")

    # Rename "Unnamed: 0" to "country_in_ratings" and pick only columns with country_in_ratings, S&P, Moody's and Fitch.
    ratings_raw = ratings_raw.rename(columns={"Unnamed: 0": "country_in_ratings"})
    ratings_raw = ratings_raw[["country_in_ratings", "S&P", "Moody's", "Fitch"]]

    # The credit rating sources calls DR Congo just Congo. Here str.strip() is needed to remove any space in the raw data.
    # In the raw data, Congo has some spaces after "o". If not used str.strip(), nothing is replaced.
    ratings_raw.country_in_ratings = ratings_raw.country_in_ratings.str.strip().replace(["Congo"], ["Congo, Dem. Rep."])

    # change country name to wb's name
    ratings_raw["country"] = replace_with_warning(ratings_raw.country_in_ratings.apply(str.strip), any_to_wb)

    ratings_raw = ratings_raw.set_index("country").drop("country_in_ratings", axis=1)

    # mystriper is a function in lib_gather_data. To lower case and strips blanks.
    ratings_raw = ratings_raw.applymap(mystriper)

    # Transforms ratings letters into 1-100 numbers
    rat_disc = load_input_data(root_dir, "cred_rat_dict.csv")
    ratings = ratings_raw
    ratings["S&P"].replace(rat_disc["s&p"].values, rat_disc["s&p_score"].values, inplace=True)
    ratings["Moody's"].replace(rat_disc["moodys"].values, rat_disc["moodys_score"].values, inplace=True)
    ratings["Fitch"].replace(rat_disc["fitch"].values, rat_disc["fitch_score"].values, inplace=True)

    # average rating over all agencies
    ratings["rating"] = ratings.mean(axis=1) / 100

    print("No rating available for regions:\n" + "; ".join(
        ratings[ratings.rating.isna()].index) + ".\nSetting rating to 0.")
    ratings.rating.fillna(0, inplace=True)

    return ratings.rating


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Script parameters')
    parser.add_argument('--use_flopros_protection', type=bool, default=True, help='Use FLOPROS for protection')
    parser.add_argument('--no_protection', type=bool, default=True, help='No protection')
    parser.add_argument('--use_GLOFRIS_flood', type=bool, default=False, help='Use GLOFRIS for flood')
    parser.add_argument('--use_avg_pe', type=bool, default=True, help='Use average PE')
    parser.add_argument('--use_newest_wdi_findex_aspire', type=bool, default=False,
                        help='Use newest WDI, Findex, Aspire')
    parser.add_argument('--drop_unused_data', type=bool, default=True, help='Drop unused data')
    parser.add_argument('--update_exposure_with_constant_fa', type=bool, default=True,
                        help='Update exposure data with constant_fa.csv')
    parser.add_argument('--econ_scope', type=str, default='country', help='Economic scope')
    parser.add_argument('--default_rp', type=str, default='default_rp', help='Default return period')
    parser.add_argument('--poverty_head', type=float, default=0.2, help='Poverty headcount')
    parser.add_argument('--reconstruction_time', type=float, default=3.0, help='Reconstruction time')
    parser.add_argument('--reduction_vul', type=float, default=0.2, help='Reduction in vulnerability')
    parser.add_argument('--inc_elast', type=float, default=1.5, help='Income elasticity')
    parser.add_argument('--discount_rate', type=float, default=0.06, help='Discount rate')
    parser.add_argument('--asset_loss_covered', type=float, default=0.8, help='Asset loss covered')
    parser.add_argument('--max_support', type=float, default=0.05, help='Maximum support')
    parser.add_argument('--fa_threshold', type=float, default=0.9, help='FA threshold')
    parser.add_argument('--root_dir', type=str, default=os.getcwd(), help='Root directory')
    parser.add_argument('--income_categories', type=str, default="poor+nonpoor", help='Income categories.')
    args = parser.parse_args()

    # use_flopros_protection = args.use_flopros_protection
    # no_protection = args.no_protection
    # use_GLOFRIS_flood = args.use_GLOFRIS_flood
    use_avg_pe = args.use_avg_pe
    # use_newest_wdi_findex_aspire = args.use_newest_wdi_findex_aspire
    # drop_unused_data = args.drop_unused_data
    default_rp = args.default_rp  # TODO: check if this can be removed
    poverty_headcount = args.poverty_head
    # reconstruction_time = args.reconstruction_time
    # reduction_vul = args.reduction_vul
    # inc_elast = args.inc_elast
    # discount_rate = args.discount_rate
    # asset_loss_covered = args.asset_loss_covered
    # max_support = args.max_support
    fa_threshold = args.fa_threshold
    update_exposure_with_constant_fa = args.update_exposure_with_constant_fa
    income_categories = args.income_categories.split("+")

    econ_scope = args.econ_scope
    event_level = [econ_scope, "hazard", "rp"]  # levels of index at which one event happens

    root_dir = args.root_dir
    input_dir = os.path.join(root_dir, 'inputs')  # get inputs data directory
    intermediate_dir = os.path.join(root_dir, 'intermediate')  # get outputs data directory

    # if the intermediate directory doesn't exist, create one
    if not os.path.exists(intermediate_dir):
        os.makedirs(intermediate_dir)

    any_to_wb, iso3_to_wb, iso2_iso3 = get_country_name_dicts(root_dir)

    # read HFA data
    hfa_data = load_hfa_data()

    # read credit ratings
    credit_ratings = load_credit_ratings()

    # compute country borrowing ability
    borrowing_ability = compute_borrowing_ability(credit_ratings, hfa_data.finance_pre,
                                                  cat_ddo_filepath="contingent_finance_countries.csv")
    # load average productivity of capital
    avg_prod_k = gather_capital_data(root_dir).avg_prod_k

    # read WB data
    wb_data = load_input_data(root_dir, "WB_socio_economic_data/wb_data.csv").set_index(econ_scope)

    # load building vulnerability classification and compute vulnerability per income class
    vulnerability = load_vulnerability_data(poverty_headcount, wb_data.share1)

    # load total hazard losses per return period (on the country level)
    hazard_loss_tot = load_gir_hazard_losses(root_dir,
                                             "GIR_hazard_loss_data/export_all_metrics.csv.zip", default_rp)

    # compute exposure and adjust vulnerability (per income category) for excess exposure
    exposure_fa, vulnerability_per_income_cat_adjusted = compute_exposure_and_adjust_vulnerability(
        hazard_loss_tot, vulnerability, fa_threshold, event_level
    )

    # apply poverty exposure bias
    exposure_fa = apply_poverty_exposure_bias(exposure_fa, use_avg_pe, poverty_headcount, wb_data["pop"])
    if update_exposure_with_constant_fa:
        # TODO: figure out what these data are and if to keep them
        exposure_fa.update(load_input_data(root_dir, 'constant_fa.csv',
                                           index_col=['country', 'hazard', 'rp', 'income_cat']).fa)

    # expand the early warning to all hazard rp's, income categories, and countries; set to 0 for specific hazrads
    # (Earthquake)
    early_warning_per_hazard = get_early_warning_per_hazard(exposure_fa.index, hfa_data.ew)

    # concatenate exposure, vulnerability and early warning into one dataframe
    # TODO: In the original code, a 'protection' column was added if no_protection was set to False. However, this
    #  variable was set to True, therefore this option is not implemented here. Refer to gather_data_old.py.
    hazard_ratios = pd.concat((exposure_fa, vulnerability_per_income_cat_adjusted, early_warning_per_hazard),
                              axis=1)

    # get protection
    protection = load_protection(exposure_fa.index, protection_data="FLOPROS", min_rp=1)

    # get data per income categories TODO: previously also contained fa and early warning, check if this is necessary
    cat_info, tau_tax = get_cat_info_and_tau_tax(wb_data, poverty_headcount, avg_prod_k)

    # TODO: include recovery duration
    # TODO: before, also included pov_head, pi (vulnerability reduction with early warning), income_elast, rho,
    #  shareable, max_increased_spending, protection; check if this is necessary
    macro = wb_data.join(hfa_data, how='left')
    macro = macro.join(credit_ratings, how='left')
    macro = macro.join(borrowing_ability, how='left')
    macro = macro.join(avg_prod_k, how='left')
    macro = macro.join(tau_tax, how='left')

    for pol_str, pol_opt in [[None, None], ['bbb_complete', 1], ['borrow_abi', 2], ['unif_poor', None], ['bbb_incl', 1],
                             ['bbb_fast', 1], ['bbb_fast', 2], ['bbb_fast', 4], ['bbb_fast', 5], ['bbb_50yrstand', 1]]:

        # apply policy
        pol_df, pol_cat_info, pol_hazard_ratios, pol_desc = apply_policy(macro.copy(deep=True), cat_info.copy(deep=True),
                                                                         hazard_ratios.copy(deep=True), event_level,
                                                                         pol_str, pol_opt)

        # clean up and save out
        # drop_unused_data:
        pol_cat_info = pol_cat_info.drop(np.intersect1d(["social"], pol_cat_info.columns), axis=1).dropna()
        pol_df_in = pol_df.drop(np.intersect1d(["social_p", "social_r", "pov_head", "pe", "vp", "vr", "axfin_p",
                                                "axfin_r", "rating", "finance_pre"], pol_df.columns),
                                axis=1).dropna()

        pol_df_in = pol_df_in.drop(np.intersect1d(["es"], pol_df_in.columns), axis=1, errors="ignore").dropna()

        # Save all data
        print(pol_df_in.shape[0], 'countries in analysis')
        outstring = (f"_{pol_str}" if pol_str is not None else '') + (str(pol_opt) if pol_opt is not None else '')

        # # save exposure and vulnerability by return country, hazard, return period, income category
        # fa_guessed_gar.to_csv(
        #     os.path.join(intermediate_dir, "fa_guessed_from_GAR_and_PAGER_shaved" + outstring + ".csv"),
        #     encoding="utf-8",
        #     header=True
        # )

        # save vulnerability (total, poor, rich) by country
        vulnerability.to_csv(
            os.path.join(intermediate_dir + "/v_pr_fromPAGER_shaved_GAR" + outstring + ".csv"),
            encoding="utf-8",
            header=True
        )

        # save macro-economic country economic data
        pol_df_in.to_csv(
            os.path.join(intermediate_dir + "/macro" + outstring + ".csv"),
            encoding="utf-8",
            header=True
        )

        # save consumption, access to finance, gamma, capital, exposure, early warning access by country and income category
        pol_cat_info.to_csv(
            os.path.join(intermediate_dir + "/cat_info" + outstring + ".csv"),
            encoding="utf-8",
            header=True
        )

        # save exposure, vulnerability, and access to early warning by country, hazard, return period, income category
        pol_hazard_ratios.to_csv(
            os.path.join(intermediate_dir + "/hazard_ratios" + outstring + ".csv"),
            encoding="utf-8",
            header=True
        )
