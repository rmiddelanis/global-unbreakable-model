# This script provides data input for the resilience indicator multihazard model.
# The script was developed by Adrien Vogt-Schilb and improved by Jinqiang Chen.
import copy

from gather_gir_data import load_gir_hazard_losses
from lib_gather_data import *
from apply_policy import *
import pandas as pd
from pandas import isnull
from lib_gather_data import replace_with_warning, get_country_name_dicts

warnings.filterwarnings("always", category=UserWarning)

# define directory
use_2016_inputs = False
use_2016_ratings = False
constant_fa = True

year_str = ''
if use_2016_inputs:
    year_str = 'orig_'

root_dir = os.getcwd()  # get current directory
input_dir = os.path.join(root_dir, year_str + 'inputs')  # get inputs data directory
intermediate_dir = os.path.join(root_dir, year_str + 'intermediate')  # get outputs data directory

# ^ if the depository directory doesn't exist, create one
if not os.path.exists(intermediate_dir):
    os.makedirs(intermediate_dir)

# Options and parameters
use_flopros_protection = True  # FLOPROS is a global database of flood potection standards. Used in Protection.
no_protection = True  # Used in Protection.
use_GLOFRIS_flood = False  # else uses GAR (True does not work i think)
use_avg_pe = True  # otherwise 0 when no data
use_newest_wdi_findex_aspire = False  # too late to include new data just before report release
drop_unused_data = True  # if true removes from df and cat_info the intermediate variables
econ_scope = "country"  # province, deparmtent
event_level = [econ_scope, "hazard", "rp"]  # levels of index at which one event happens
default_rp = "default_rp"  # return period to use when no rp is provided (mind that this works with protection)
income_cats = pd.Index(["poor", "nonpoor"], name="income_cat")  # categories of households
affected_cats = pd.Index(["a", "na"], name="affected_cat")  # categories for social protection
helped_cats = pd.Index(["helped", "not_helped"], name="helped_cat")

poverty_head = 0.2  # poverty headcount
reconstruction_time = 3.0  # Reconstruction time
reduction_vul = 0.2  # how much early warning reduces vulnerability
inc_elast = 1.5  # income elasticity
discount_rate = 0.06  # discount rate
asset_loss_covered = 0.8  # target of asset losses to be covered by scale up
max_support = 0.05  # 5% of GDP in post-disaster support maximum, if everything is ready
fa_threshold = 0.9  # if fa (=exposure) is above this threshold, then it is capped at this value

any_to_wb, iso3_to_wb, iso2_iso3 = get_country_name_dicts(root_dir)

# Run GAR preprocessing
gar_preprocessing(root_dir, intermediate_dir, default_rp=default_rp)

# Read data
##Macro data
###Economic data from the world bank
df = load_input_data(root_dir, "WB_socio_economic_data/wb_data.csv").set_index(econ_scope)
print("Note: check that the wb_data.csv file is up to date. If not, download it using the script download_wb_data.py")
df = df[['gdp_pc_pp', 'pop', 'share1', 'urbanization_rate', 'gdp_pc_cd', 'axfin_p', 'axfin_r', 'social_p', 'social_r']]

# Define parameters
df["pov_head"] = poverty_head  # poverty head
ph = df.pov_head
df["T_rebuild_K"] = reconstruction_time  # Reconstruction time
df["pi"] = reduction_vul  # how much early warning reduces vulnerability
df["income_elast"] = inc_elast  # income elasticity
df["rho"] = discount_rate  # discount rate
df["shareable"] = asset_loss_covered  # target of asset losses to be covered by scale up
df["max_increased_spending"] = max_support  # 5% of GDP in post-disaster support maximum, if everything is ready


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
hfa_oldnew = pd.concat([hfa_newest, hfa_previous, hfa15], axis=1,keys=['new', 'old', "15"])
hfa = hfa_oldnew["15"].fillna(hfa_oldnew["new"].fillna(hfa_oldnew["old"]))

# access to early warning normalized between zero and 1.
hfa["shew"] = 1 / 5 * hfa["P2-C3"]

# q_s in the report, ability to scale up support to affected population after the disaster
# normalized between zero and 1
hfa["prepare_scaleup"] = (hfa["P4-C2"] + hfa["P5-C2"] + hfa["P4-C5"]) / 3 / 5

# between 0 and 1	!!!!!!!!!!!!!!!!!!!REMARK: INCONSISTENT WITH THE TECHNICAL PAPER. Q_f=1/2(ratings+P5C3/5)
# TODO @Bramka @Stephane
hfa["finance_pre"] = (1 + hfa["P5-C3"]) / 6
df[["shew", "prepare_scaleup", "finance_pre"]] = hfa[["shew", "prepare_scaleup", "finance_pre"]]

# fill NA values with 0. assumes no reporting is bad situation
# (caution! do the fillna after inputing to df to get the largest set of index)
df[["shew", "prepare_scaleup", "finance_pre"]] = df[["shew", "prepare_scaleup", "finance_pre"]].fillna(0)

# Income group
# df["income_group"]=pd.read_csv(inputs+"income_groups.csv",header=4,index_col=2)["Income group"].dropna()

# Country Ratings
credit_ratings_file = "credit_ratings_scrapy.csv"
if use_2016_inputs or use_2016_ratings:
    credit_ratings_file = "cred_rat.csv"
print("Warning. Check that credit_ratings_scrapy.csv is up to date. If not, download it from "
      "http://www.tradingeconomics.com/country-list/rating")

# TODO: Fitch no longer on tradingeconomics.com. Find a different source? Also, here DBRS is not used. Why?
# drop rows where only all columns are NaN.
ratings_raw = load_input_data(root_dir, credit_ratings_file, dtype="str", encoding="utf8", na_values=['NR']).dropna(how="all")

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
df["rating"] = ratings["rating"]
if True:
    print("No rating available for regions:\n" + "; ".join(df.loc[isnull(df.rating)].index) + ".\nSetting rating to 0.")
df["rating"].fillna(0, inplace=True)  # assumes no rating is bad rating

# Ratings + HFA
# Ability and willingness to improve transfers after the disaster
# df["finance_pre"] =
df["borrow_abi"] = (df["rating"] + df["finance_pre"]) / 2

# If contingent finance instrument then borrow_abo = 1
# TODO: @Bramka what are these countries? @Stephane
#   catDDO is an indicator whether countries have received aids previously from the WB
contingent_file = "contingent_finance_countries.csv"
if use_2016_inputs:
    contingent_file = "contingent_finance_countries_orig.csv"
contingent_countries = load_input_data(root_dir, contingent_file, dtype="str", encoding="utf8").set_index("country")
contingent_countries["catDDO"] = 1
df = pd.merge(df.reset_index(), contingent_countries.reset_index(), on="country", how="outer").set_index("country")
df.loc[df.catDDO == 1, "borrow_abi"] = 1
df.drop(["catDDO"], axis=1, inplace=True)



# \mu in the technical paper -- average productivity of capital
df["avg_prod_k"] = gather_capital_data(root_dir).avg_prod_k

# Hazards data
# # Vulnerability from Pager data
# pager_category_matching = load_input_data(root_dir, "pager_description_to_aggregate_category.csv",
#                                       index_col="pager_description").squeeze()
# pager_excel_file = "PAGER_Inventory_database_v2.0.xlsx"
# pager_data = load_input_data(root_dir, "PAGER_Inventory_database_v2.0.xlsx", sheet_name="Release_Notes", usecols="B:C",
#                              skiprows=56).dropna().squeeze()
#
# # removes spaces and dots from PAGER description
# pager_data.Description = pager_data.Description.str.strip(". ")
#
# # replace double spaces with single spaces
# pager_data.Description = pager_data.Description.str.replace("  ", " ")
# pager_data = pager_data.set_index("PAGER-STR")
#
# # results in a table with PAGER-STR index and associated category (fragile, median etc.)
# housing_categories = replace_with_warning(pager_data.Description, pager_category_matching, joiner="\n")
#
# # total share of each category of building per country
# hous_share_rur_res = get_share_from_sheet(load_input_data(root_dir, pager_excel_file, sheet_name='Rural_Res'), housing_categories, iso3_to_wb)
# hous_share_rur_nonres = get_share_from_sheet(load_input_data(root_dir, pager_excel_file, sheet_name='Rural_Non_Res'), housing_categories, iso3_to_wb)
# hous_share_rur = .5 * hous_share_rur_res + .5 * hous_share_rur_nonres
#
# hous_share_urb_res = get_share_from_sheet(load_input_data(root_dir, pager_excel_file, sheet_name='Urban_Res'), housing_categories, iso3_to_wb)
# hous_share_urb_nonres = get_share_from_sheet(load_input_data(root_dir, pager_excel_file, sheet_name='Urban_Non_Res'), housing_categories, iso3_to_wb)
# hous_share_urb = .5 * hous_share_urb_res + .5 * hous_share_urb_nonres
#
# # the sum(axis=1) of hous_share_rur is equal to 1, so hous_share_rur needs to be weighted by the 1-urbanization_rate
# # same for hous_share_urb
# hous_share_tot = (hous_share_rur.stack() * (1 - df.urbanization_rate) + hous_share_urb.stack() * df.urbanization_rate).unstack().dropna()
# hous_share_tot = hous_share_tot[hous_share_tot.index.isin(iso3_to_wb)]  # the share of building inventory for fragile, median and robust

hous_share_tot = load_input_data(root_dir, "GEM_vulnerability/country_vulnerability_classes.csv",
                                 index_col="country").drop('iso3', axis=1)

# matching vulnerability of buildings and people's income and calculate poor's, rich's and country's vulnerability
hous_cat_vulnerability = load_input_data(root_dir, "aggregate_category_to_vulnerability.csv", sep=";",
                                     index_col="aggregate_category").squeeze()

# REMARK: NEED TO BE CHANGED....Stephane I've talked to @adrien_vogt_schilb and don't want you to go over our whole
# conversation. Here is the thing: in your model, you assume that the bottom 20% of population gets the 20% of buildings
# of less quality. I don't think it's a fair justification, because normally poor people live in buildings of less
# quality but in a more crowded way,i.e., it could be the bottom 40% of population get the 10% of buildings of less
# quality. I think we need to correct this matter. @adrien_vogt_schilb also agreed on that, if he didn't change his
# opinion. How to do that? I think once we incorporate household data, we can allocate buildings on the decile of
# households, rather than population. I think it's a more realistic assumption.
# TODO @Bramka @Stephane --> GMD data?
p = (hous_share_tot.cumsum(axis=1).add(-df["pov_head"], axis=0)).clip(lower=0)
poor = (hous_share_tot - p).clip(lower=0)
rich = hous_share_tot - poor
vulnerability_poor = ((poor * hous_cat_vulnerability).sum(axis=1, skipna=False) / df["pov_head"])
vulnerability_rich = (rich * hous_cat_vulnerability).sum(axis=1, skipna=False) / (1 - df["pov_head"])

# vulnerability weighted with income shares
# TODO: why? shouldn't this be simply (hous_share_tot * hous_cat_vulnerability).sum(axis=1) ?
vulnerability_tot = vulnerability_poor * df.share1 + vulnerability_rich * (1 - df.share1)
vulnerability_tot.name = "v"
vulnerability_tot.index.name = "country"

# apply \delta_K = f_a * V, and use destroyed capital from GAR data, and fa_threshold to recalculate vulnerability
# \delta_K, Generated by pre_process\ GAR.ipynb
frac_value_destroyed_gar = load_gir_hazard_losses(root_dir, "GIR_hazard_loss_data/export_all_metrics.csv.zip",
                                                  default_rp)

# frac_value_destroyed_gar = pd.read_csv(
#     os.path.join(intermediate_dir, "frac_value_destroyed_gar_completed.csv"),
#     index_col=["country", "hazard", "rp"]
# ).squeeze()

# fa is the fraction of asset affected (=exposure): \Delta K = f_a * V; with f_a the exposure.
# broadcast_simple, substitute the value in frac_value_destroyed_gar by values in vulnerability_tot.
# Previously, it assumed that vulnerability for all types of disasters are the same.
# fa_guessed_gar = exposure/vulnerability. Now we will change this to event-specific vulnerabilities...
fa_guessed_gar = (
    (frac_value_destroyed_gar / broadcast_simple(vulnerability_tot, frac_value_destroyed_gar.index)).dropna()).to_frame()
fa_guessed_gar.columns = ['fa']

# merge vulnerability with hazard_ratios
fa_guessed_gar = pd.merge(fa_guessed_gar.reset_index(), vulnerability_rich.reset_index(), on=econ_scope)
fa_guessed_gar = pd.merge(fa_guessed_gar, vulnerability_poor.reset_index(), on=econ_scope)
fa_guessed_gar.columns = ['country', 'hazard', 'rp', 'fa', 'nonpoor', 'poor']

fa_guessed_gar = fa_guessed_gar.set_index(event_level + ['fa']).stack()
fa_guessed_gar.index.names = event_level + ['fa', 'income_cat']
fa_guessed_gar = fa_guessed_gar.reset_index().set_index(event_level + ['income_cat'])
fa_guessed_gar.columns = ['fa', 'v']

# clip f_a to fa_threshold
excess = fa_guessed_gar.loc[fa_guessed_gar.fa > fa_threshold, 'fa']
for c in excess.index:
    r = (excess / fa_threshold)[c]  # ratio by which fa is above fa_threshold
    print('\nExcess case:\n', c, '\n', r, '\n', fa_guessed_gar.loc[c], '\n\n')
    fa_guessed_gar.loc[c, 'fa'] = fa_guessed_gar.loc[c, 'fa'] / r  # set f_a to fa_threshold
    fa_guessed_gar.loc[c, 'v'] = fa_guessed_gar.loc[c, 'v'] * r  # increase v s.th. \Delta K = f_a * v does not change
    print('r=', r)
    print('changed to:', fa_guessed_gar.loc[c], '\n\n')
    if fa_guessed_gar.loc[c, 'v'] > .99:
        print(f"\nExcess vulnerability: v={fa_guessed_gar.loc[c, 'v']}. Setting =.99")
        fa_guessed_gar.loc[c, 'v'] = .99

fa_guessed_gar = fa_guessed_gar.reset_index().set_index(event_level)


# Exposure bias from PEB

# Exposure bias from WB povmaps study
exposure_bias_wb = load_input_data(root_dir, "PEB_flood_povmaps.xlsx")[["iso", "peb"]].dropna()
df["pe"] = exposure_bias_wb.set_index(exposure_bias_wb.iso.replace(iso3_to_wb)).peb - 1
# Exposure bias from older WB DELTARES study
exposure_bias_deltares = load_input_data(root_dir, "PEB_wb_deltares.csv", skiprows=[0, 1, 2],
                                     usecols=["Country", "Nation-wide"])
# Replace with warning is used for columns, for index set_index is needed.
exposure_bias_deltares["country"] = replace_with_warning(exposure_bias_deltares["Country"], any_to_wb)
# Completes with bias from previous study when pov maps not available. squeeze is needed or else it's impossible to
# fillna with a dataframe
df["pe"] = df["pe"].fillna(exposure_bias_deltares.set_index("country").drop(["Country"], axis=1).squeeze())

# if use_avg_pe, use averaged pe from global data for countries that don't have PE. Otherwise use 0.
if use_avg_pe:
    df["pe"] = df["pe"].fillna(wavg(df["pe"], df["pop"]))
else:
    df["pe"].fillna(0)
pe = df.pop("pe")


# incorporates exposure bias, but only for (riverine) flood and surge, and gets an updated fa for income_cats
# fa_hazard_cat = broadcast_simple(fa_guessed_gar,index=income_cats) #fraction of assets affected per hazard and income
# categories fa_with_pe = concat_categories(fa_guessed_gar*(1+pe),fa_guessed_gar*(1-df.pov_head*(1+pe))/(1-df.pov_head),
# index=income_cats)
# NB: fa_guessed_gar*(1+pe) gives f_p^a and fa_guessed_gar*(1-df.pov_head*(1+pe))/(1-df.pov_head) gives f_r^a. TESTED
# fa_guessed_gar.update(fa_with_pe) #updates fa_guessed_gar where necessary

# selects just flood and surge
fa_with_pe = fa_guessed_gar.query("hazard in ['flood','surge']")[['income_cat', 'fa']].copy()
fa_with_pe.loc[fa_with_pe['income_cat'] == 'poor', 'fa'] = (
        fa_with_pe.loc[fa_with_pe['income_cat'] == 'poor', 'fa'] * (1 + pe))
fa_with_pe.loc[fa_with_pe['income_cat'] == 'nonpoor', 'fa'] = (
        fa_with_pe.loc[fa_with_pe['income_cat'] == 'nonpoor', 'fa'] * (1 - df.pov_head * (1 + pe)) / (1 - df.pov_head)
)

fa_with_pe = fa_with_pe.reset_index().set_index(event_level + ['income_cat'])
fa_guessed_gar = fa_guessed_gar.reset_index().set_index(event_level + ['income_cat'])

fa_guessed_gar['fa'].update(fa_with_pe['fa'])
if constant_fa:
    if use_2016_inputs:
        fa_guessed_gar.to_csv(input_dir + 'constant_fa.csv', header=True)
    else:
        # TODO: @Bramka what is this constant_fa.csv in orig_inputs? keeping it for now, but need to clean this.
        #   --> can probably stay unused in the update
        fa_guessed_gar['fa'].update(
            load_input_data(root_dir, 'constant_fa.csv', index_col=['country', 'hazard', 'rp', 'income_cat'])['fa'])

# gathers hazard ratios
hazard_ratios = copy.deepcopy(fa_guessed_gar)
hazard_ratios = pd.merge(hazard_ratios.reset_index(), df['shew'].reset_index(), on=['country']).set_index(
    event_level + ['income_cat'])

# set shew (early warning) to 0 for earthquake
hazard_ratios["shew"] = hazard_ratios.shew.unstack("hazard").assign(earthquake=0).stack(
    "hazard").reset_index().set_index(event_level + ["income_cat"])
if not no_protection:
    # protection at 0 for earthquake and wind
    hazard_ratios["protection"] = 1
    # TODO: but this sets protection of earthquake and wind to 1?!
    hazard_ratios["protection"] = hazard_ratios.protection.unstack("hazard").assign(earthquake=1, wind=1).stack(
        "hazard").reset_index().set_index(event_level)
hazard_ratios = hazard_ratios.drop("Finland")  # because Finland has fa=0 everywhere.

# Protection
# TODO: is this applied to all hazards or just floods?
# TODO: use FLOPROS V1 for protection; try to infer missing countries from FLOPROS based on GDP-protection correlation
if use_flopros_protection:  # in this code, this protection is overwritten by no_protection
    minrp = 1 / 2  # assumes nobody is flooded more than twice a year
    df["protection"] = load_input_data(root_dir, "protection_national_from_flopros.csv",
                                   index_col="country").squeeze().clip(lower=minrp)
else:  # assumed a function of the country's income group
    # note: "protection_level_assumptions.csv" can be found in orig_inputs.
    protection_assumptions = load_input_data(root_dir, "protection_level_assumptions.csv",
                                         index_col="Income group").squeeze()
    df["protection"] = load_input_data(root_dir, "income_groups.csv", header=4, index_col=2)[
        "Income group"].dropna().replace(protection_assumptions)
if no_protection:
    p = hazard_ratios.reset_index("rp").rp.min()
    df.protection = p
    print("PROTECTION IS ", p)

# Data by income categories
cat_info = pd.DataFrame()
cat_info["n"] = concat_categories(ph, (1 - ph), index=income_cats)  # number
cp = df["share1"] / ph * df["gdp_pc_pp"]  # consumption levels, by definition.
cr = (1 - df["share1"]) / (1 - ph) * df["gdp_pc_pp"]
cat_info["c"] = concat_categories(cp, cr, index=income_cats)
cat_info["social"] = concat_categories(df.social_p, df.social_r, index=income_cats)  # diversification
cat_info["axfin"] = concat_categories(df.axfin_p, df.axfin_r, index=income_cats)  # access to finance
cat_info = cat_info.dropna()

# Taxes, redistribution, capital

# compute tau tax and gamma_sp from social_poor and social_nonpoor. CHECKED!
df["tau_tax"], cat_info["gamma_SP"] = social_to_tx_and_gsp(econ_scope, cat_info)

# here k in cat_info has poor and non poor, while that from capital_data.csv has only k, regardless of poor or nonpoor
cat_info["k"] = (1 - cat_info["social"]) * cat_info["c"] / ((1 - df["tau_tax"]) * df["avg_prod_k"])
# TODO: check; cat_info["k"].groupby('country').sum() is very different from k_data["k"] for some countries.
#  Shoudln't they be the same?

# Exposure
# TODO: why average over all hazards and return periods?!
cat_info["fa"] = hazard_ratios.fa.groupby(level=["country", "income_cat"]).mean()

# access to early warnings
# TODO: again, why average over all hazards and return periods?!
cat_info["shew"] = hazard_ratios.shew.drop("earthquake", level="hazard").groupby(level=["country", "income_cat"]).mean()


# apply policies
for pol_str, pol_opt in [[None, None], ['bbb_complete', 1], ['borrow_abi', 2], ['unif_poor', None], ['bbb_incl', 1],
                         ['bbb_fast', 1], ['bbb_fast', 2], ['bbb_fast', 4], ['bbb_fast', 5], ['bbb_50yrstand', 1]]:

    # apply policy
    pol_df, pol_cat_info, pol_hazard_ratios, pol_desc = apply_policy(df.copy(deep=True), cat_info.copy(deep=True),
                                                                     hazard_ratios.copy(deep=True), event_level,
                                                                     pol_str, pol_opt)

    # clean up and save out
    if drop_unused_data:
        pol_cat_info = pol_cat_info.drop(np.intersect1d(["social"], pol_cat_info.columns), axis=1).dropna()
        pol_df_in = pol_df.drop(np.intersect1d(["social_p", "social_r", "pov_head", "pe", "vp", "vr", "axfin_p",
                                                "axfin_r", "rating", "finance_pre"], pol_df.columns), axis=1).dropna()
    else:
        pol_df_in = pol_df.dropna()
    pol_df_in = pol_df_in.drop(np.intersect1d(["shew"], pol_df_in.columns), axis=1, errors="ignore").dropna()

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
    pd.DataFrame(
        [vulnerability_poor, vulnerability_rich, vulnerability_tot], index=["vp", "vr", "v"]).T.to_csv(
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
