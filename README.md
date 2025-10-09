# Global Model for Socio-economic Resilience to Natural Disasters

## Contents

1. [Overview](#overview)
2. [Usage](#usage)
   - [Prerequisites](#prerequisites)
   - [Model settings](#model-settings)
   - [Run the model](#run-the-model)
3. [Data Availability](#data-availability)



## Overview

This package contains the code and data to calculate global estimates of socio-economic resilience. 

If you use this code or the data it produces as, please cite the following paper:

Robin Middelanis, Bramka Arga Jafino, Ruth Hill, Minh Cong Nguyen, and Stéphane Hallegatte (2025). "Global Socio-economic Resilience to Natural Disasters". _Policy Research Working Paper_.

As not all raw datasets are publicly available, this package contains pre-processed data files to reproduce the results presented in the paper. Raw data files are not included in this package, but the sources are indicated in the [data availability section](#data-availability). 

This package is structured as follows: 
```
├── data/                      # data directory
│   ├── raw/                   # directory for raw data (data files not included)
│   └── processed/             # pre-processed data to run the model
│
├── example/                   
│   └── settings_example.yml   # example settings file to run the model
│
├── src/unbreakable/           # source code
│       ├── misc/              # utility functions and miscellaneous scripts
│       ├── model/             # core model implementation
│       ├── post_processing/   # post-processing of model output
│       └── scenario/          # prepares simulation data
│
├── environment.yml            # dependencies
└── README.md                  # this README file
```


## Usage

### Prerequisites
The model and data processing run on Python. To install the required dependencies, create a conda environment from the `environment.yml` file and activate it:
```bash
conda env create -n disaster_resilience -f environment.yml
conda activate disaster_resilience
```

### Model settings
The model is configured via a settings file in YAML format. An example settings file is provided in `./example/settings_example.yml`.
The following table provides an overview of the settings options.

| **Parameter**                                          | **Description**                                                  | **Example / Default**                                      |
|--------------------------------------------------------|------------------------------------------------------------------|------------------------------------------------------------|
| **model_params**                                       |                                                                  |                                                            |
| `outpath`                                              | Directory for model output files.                                | `./results`                                                |
| `run_params.verbose`                                   | Print detailed output during model run.                          | `false`                                                    |
| `pds_params.covered_loss_share`                        | Fraction of losses covered by post-disaster support.             | `0.2` for 20%                                              |
| `pds_params.pds_borrowing_ability`                     | Country borrowing ability.                                       | Scale 0-1                                                  |
| `pds_params.pds_lending_rate`                          | Fraction of GDP that can be borrowed for PDF.                    | [`0.05` for 5% of GDP, `unlimited`]                        |
| `pds_params.pds_scope`                                 | Population share intervals for targeting or policy scaling.      | e.g. `[[0, 0.2], [0.2, 0.4]` for bottom two quintiles      |
| `pds_params.pds_targeting`                             | Targeting quality of post-disaster support.                      | [`perfect`, `data`]                                        |
| `pds_params.pds_variant`                               | Type of post-disaster support.                                   | [`no`, `proportional`, ...]                                |
| **scenario_params**                                    |                                                                  |                                                            |
| `run_params.countries`                                 | Countries to include.                                            | `all`                                                      |
| `run_params.download`                                  | Whether to (re)download input data from the WB API.              | `false`                                                    |
| `run_params.exclude_countries`                         | List of countries to exclude.                                    | `[THA, LUX]`                                               |
| `run_params.hazards`                                   | Hazards to include.                                              | `all`                                                      |
| `run_params.include_poverty_data`                      | Include poverty-related data in output.                          | `false`                                                    |
| `run_params.num_cores`                                 | Number of CPU cores to use.                                      | `1`                                                        |
| `run_params.pip_reference_year`                        | Reference year for poverty data.                                 | `2021`                                                     |
| `run_params.poverty_line`                              | Poverty line in USD (PPP).                                       | `3.0`                                                      |
| `run_params.recompute`                                 | Whether to recompute model input data.                           | `false`                                                    |
| `run_params.recompute_hazard_protection`               | Whether to recompute hazard protection data.                     | `false`                                                    |
| `run_params.resolution`                                | Population resolution.                                           | e.g. `0.2` for quintiles                                   |
| `run_params.verbose`                                   | Print detailed output during input processing.                   | `false`                                                    |
| `data_params.transfers_regression_params.type`         | Regression type for transfer shares estimation.                  | `LassoCV` for cross-validated Lasso Regression             |
| `data_params.transfers_regression_params.features`     | Features used in regression model.                               | `[GDP_{pc}, REM, SOC, UNE, income_group, region, FSY]`     |
| `hazard_params.fa_threshold`                           | Maximum allowed national exposure.                               | `0.99`                                                     |
| `hazard_params.hazard_protection`                      | Hazard protection dataset.                                       | `FLOPROS`                                                  |
| `hazard_params.no_exposure_bias`                       | Disable exposure bias adjustment.                                | `false`                                                    |
| `hazard_params.zero_rp`                                | Assumed natural protection level.                                | `2`                                                        |
| `macro_params.axfin_impact`                            | Effect of financial access on diversified income.                | `0.1`                                                      |
| `macro_params.discount_rate_rho`                       | Discount rate for welfare aggregation.                           | `0.06`                                                     |
| `macro_params.early_warning_file`                      | Optional file path for external early warning benefit estimates. | `null`                                                     |
| `macro_params.income_elasticity_eta`                   | Elasticity of income with respect to consumption.                | `1.5`                                                      |
| `macro_params.reconstruction_capital`                  | Capital fraction reconstructed by households.                    | `self_hous`                                                |
| `macro_params.reduction_vul`                           | Reduction in vulnerability due to early warning.                 | `0.2`                                                      |
| `policy_params.min_diversified_share.parameter`        | Minimum diversified income policy parameter.                     | `0` (= policy not applied)                                 |
| `policy_params.min_diversified_share.scope`            | Population scope for minimum diversified share policy.           | `[[0, 0.2], [0.2, 0.4], [0.4, 0.6], [0.6, 0.8], [0.8, 1]]` |
| `policy_params.scale_exposure.parameter`               | Scaling factor for exposure.                                     | `0.95` (= policy not applied)                              |
| `policy_params.scale_exposure.scale_total`             | Whether scaling applies to total exposure.                       | `true`                                                     |
| `policy_params.scale_exposure.scope`                   | Population scope for scaling exposure.                           | `[[0, 0.2]]`                                               |
| `policy_params.scale_income.parameter`                 | Scaling factor for national income.                              | `1` (= policy not applied)                                 |
| `policy_params.scale_income.scope`                     | Population scope for scaling income.                             | `[[0,0.2],[0.2,0.4],[0.4,0.6],[0.6,0.8],[0.8,1]]`          |
| `policy_params.scale_liquidity.parameter`              | Scaling factor for liquidity.                                    | `1` (= policy not applied)                                 |
| `policy_params.scale_liquidity.scope`                  | Population scope for scaling liquidity.                          | `[[0,0.2],[0.2,0.4],[0.4,0.6],[0.6,0.8],[0.8,1]]`          |
| `policy_params.scale_non_diversified_income.parameter` | Scaling factor for non-diversified income share.                 | `1` (= policy not applied)                                 |
| `policy_params.scale_non_diversified_income.scope`     | Population scope for scaling non-diversified income.             | `[[0,0.2],[0.2,0.4],[0.4,0.6],[0.6,0.8],[0.8,1]]`          |
| `policy_params.scale_self_employment.parameter`        | Scaling factor for self-employment share.                        | `1` (= policy not applied)                                 |
| `policy_params.scale_self_employment.scope`            | Population scope for scaling self-employment.                    | `[[0,0.2],[0.2,0.4],[0.4,0.6],[0.6,0.8],[0.8,1]]`          |
| `policy_params.scale_vulnerability.parameter`          | Scaling factor for vulnerability.                                | `1` (= policy not applied)                                 |
| `policy_params.scale_vulnerability.scale_total`        | Whether scaling applies to total vulnerability.                  | `false`                                                    |
| `policy_params.scale_vulnerability.scope`              | Population scope for scaling vulnerability.                      | `[[0,0.2],[0.2,0.4],[0.4,0.6],[0.6,0.8],[0.8,1]]`          |

### Run the model

To run the model, execute 
```bash
python ./src/model/run_model.py <path_to_settings_file>
```
where `<path_to_settings_file>` is the path to a settings file that specifies the model's parameters for simulation, including the parameterization of policy scenarios. An example settings file is provided in `./code/model/settings_example.yml`. 

The model will automatically create scenario model inputs using the script `./code/scenario/prepare_scenario.py`. These model inputs are derived from preprocessed input data in `./data/processed`. Optionally, the preprocessed data can be re-generated by setting the `recompute` and `recompute_hazard_protection` flags in the settings file. 
To also download the most recent data available from the World Bank's API, set the `download` flag. When executed, the model will recompute preprocessed data at runtime from the locally stored raw data in `./data/raw`. Note that not all raw data files are included in this package.

After preparing the simulation inputs, the model will run the simulation and generate output files in the directory specified by the settings file.

## Data Availability

This section outlines where and how the data supporting the findings of the study can be accessed and used.

### Manually downloaded raw data
This section provides an overview of the raw data used in the model. Each data set must be stored in a subdirectory of `./data/raw/` as described below. 
```
  ├── data/                      
  │   ├── raw/     
→ │   │    ├── <...>/                             
  │   └── processed/             
```

[1] Global Financial Inclusion (Global Findex) Database - editions 2011, 2014, 2017, and 2021
  - Reference: Development Research Group, Finance and Private Sector Development Unit. (2022). Global Financial Inclusion (Global Findex) Database 2021 [Data set]. World Bank, Development Data Group. 
  - identifiers WLD_2011_FINDEX_v02_M, WLD_2014_FINDEX_v01_M, WLD_2017_FINDEX_v02_M, WLD_2021_FINDEX_v03_M
  - filenames: `WLD_2011_FINDEX_v02_M.csv`, `WLD_2014_FINDEX_v01_M.csv`, `WLD_2017_FINDEX_v02_M.csv`, `WLD_2021_FINDEX_v03_M.csv`
  - location: `./data/raw/FINDEX/`
  - URL: https://microdata.worldbank.org/
  - Accessed on: 2025-02-18
  - License: World Bank Microdata Research License (https://datacatalog.worldbank.org/public-licenses?fragment=research) under "Public use" classification

[2] General government actual expenditure on social protection including and excluding health care, latest available year (percentage of GDP)
  - Reference: International Labour Office. (2024). _World Social Protection Report 2024-2026: Universal Social Protection for Climate
Action and a Just Transition_. Geneva: International Labour Office.
  - filename: data export renamed to `ILO_WSPR_SP_exp.csv`
  - location: `./data/raw/social_share_regression/`
  - URL: https://www.social-protection.org/gimi/ShowWiki.action?id=52
  - Accessed on: 2025-02-25
  - License: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)


[3] Penn World Table v10.01
  - Reference: Feenstra, R. C., Inklaar, R., & Timmer, M. P. (2015). The Next Generation of the Penn World Table. _American Economic Review_, 105(10), 3150–3182.
  - filename: `pwt1001.xlsx`
  - location: `./data/raw/PWT/`
  - URL: https://www.rug.nl/ggdc/productivity/pwt/
  - Accessed on: 2023-12-13
  - License: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)

[4] Modeled asset losses from the Global Infrastructure and Resilience Index
  - Reference: CDRI. (2023). _Global Infrastructure Resilience: Capturing the Resilience
Dividend - A Biennial Report from the Coalition for Disaster Resilient Infrastructure_. New Delhi.
  - filename: `export_all_metrics.csv.zip`
  - location: `./data/raw/GIRI/`
  - URL: https://giri.unepgrid.ch
  - Accessed on: 2023-12-21
  - License: [CC BY 3.0 IGO](http://creativecommons.org/licenses/by/3.0/igo)

[5] Flood protection levels
  - Reference: Scussolini, P., Aerts, J. C. J. H., Jongman, B., Bouwer, L. M., Winsemius, H. C., de Moel, H., & Ward, P. J. (2016). FLOPROS: an evolving global database of flood protection standards. _Natural Hazards and Earth System Sciences_, 16, 1049–1061. https://doi.org/10.5194/nhess-16-1049-2016
  - filenames: all files in `nhess-16-1049-2016-supplement.zip/Scussolini_etal_Suppl_info/FLOPROS_shp_V1`
  - location: `./data/raw/FLOPROS/Scussolini_et_al_FLOPROS_shp_V1` 
  - URL: https://nhess.copernicus.org/articles/16/1049/2016/nhess-16-1049-2016-supplement.zip
  - Accessed on: 2024-01-19
  - License: [CC BY 3.0](http://creativecommons.org/licenses/by/3.0/)

[6] Modeled coastal flood protection layer
  - Reference: Tiggeloven, T. (2020). Benefit-cost analysis of adaptation objectives to coastal flooding at the global scale (Version 2) [Data set]. Zenodo.
  - filename: `Results_adaptation_objectives.zip`
  - location: unzip to `./data/raw/FLOPROS/Tiggeloven_et_al_2020_data/`
  - URL: https://doi.org/10.5281/zenodo.4275517
  - Accessed on: 2024-01-19
  - License: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)

[7] World Bank Official Boundaries
  - Reference: World Bank. (2025). _World Bank Official Boundaries_.
  - filename: `WB_GAD_ocean_mask.zip`, `WB_GAD_Lines.zip`, `WB_GAD_ADM0_complete.zip` 
  - location: `./data/raw/WB_shapes/`
  - URL: https://datacatalog.worldbank.org/infrastructure-data/search/dataset/0038272/World-Bank-Official-Administrative-Boundaries
  - Accessed on: 2025-10-08
  - License: CC BY-4.0

[8] Gridded population density data
  - Reference: CIESIN. (2018). _Gridded Population of the World, Version 4 (GPWv4): Population Density Adjusted to Match 2015 Revision UN WPP Country Totals, Revision 11 (Version 4.11)_. Palisades, NY: NASA Socioeconomic Data and Applications Center (SEDAC).
  - filename: `gpw_v4_population_density_adjusted_rev11_2pt5_min.nc.zip`
  - location: `./data/raw/GPW/`
  - URL: https://doi.org/10.7927/H4F47M65
  - Accessed on: 2024-01-15
  - License: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)

[9] Global Exposure Model data set
  - Reference: Yepes-Estrada, C., Calderon, A., Costa, C., Crowley, H., Dabbeek, J., Hoyos, M., Martins, L., Paul, N., Rao, A., & Silva, V. (2023). Global Building Exposure Model for Earthquake Risk Assessment. _Earthquake Spectra_. https://doi.org/10.1177/87552930231194048
  - URL: https://github.com/gem/global_exposure_model
  - clone repository to `./data/raw/GEM_vulnerability/global_exposure_model/`
  - Accessed on: 2025-01-06
  - License: [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/)

[10] Relative per-quintile vulnerabilities derived from the Global Monitoring Database (GMD)
  - Reference: World Bank. (2023). _Global Monitoring Database (GMD)_.
  - filename: `Dwelling quintile vul ratio.xlsx`'
  - location: `./data/raw/GMD/`
  - Accessed on: 2024-02-28
  - not publicly available

[11] Hyogo Framework for Action performance indicators
  - Reference: UNISDR. (2007). _Hyogo framework for action 2005-2015: building the resilience of nations and communities to disasters._ Geneva: United Nations Office for Disaster Risk Reduction. https://www.unisdr.org/we/inform/publications/43291
  - filenames: `HFA_all_2009_2011.csv`, `HFA_all_2011_2013.csv`, `HFA_all_2013_2015.csv`
  - location: `./data/raw/HFA/`
  - Compiled list of indicators not publicly available

[12] World Risk Poll
  - Reference: Lloyd's Register Foundation. (2022). _2021 World Risk Poll_. London: Lloyd's Register Foundation.
  - filename: `lrf_wrp_2021_full_data.csv.zip`
  - location: `./data/raw/WRP/`
  - Accessed on: 2024-01-30
  - URL: https://www.lrfoundation.org.uk/en/research/world-risk-poll/

[13] Estimated number of exposed people by poverty line and hazard
  - Reference: Doan, M. K., Hill, R., Hallegatte, S., Corral Rodas, P. A., Brunckhorst, B. J., Nguyen, M., Freije-Rodriguez, S., & Naikal, E. G. (2023). Counting People Exposed to, Vulnerable to, or at High Risk From Climate Shocks — A Methodology. _Policy Research working paper_, 10619. http://documents.worldbank.org/curated/en/099602511292336760 
  - filename: `exposure bias.dta`
  - location: `./data/raw/PEB/`
  - not publicly available

[14] General government, private, and public-private partnership capital stock data 
  - Reference: IMF. (2022). _Investment and Capital Stock Dataset (ICSD), 1960-2021_.
  - filename: `IMFInvestmentandCapitalStockDataset2021.xlsx`
  - location: `./data/raw/IMF/`
  - Accessed on: 2025-02-13
  - URL: https://infrastructuregovern.imf.org/content/dam/PIMA/Knowledge-Hub/dataset/IMFInvestmentandCapitalStockDataset2021.xlsx

Data from the United Nations Statistics Division
  - [15] "Households in housing units by type of housing unit, tenure of household and urban/rural residence"
    - filename: data export renamed to `2025-02-18_household_tenure.csv`
    - Accessed on: 2025-02-18
  - [16] "Table 2.4 Value added by industries at constant prices (ISIC Rev. 4)"
    - filename: data export renamed to `2025-02-13_value_added_by_industry.csv`
    - Accessed on: 2025-02-13
  - URL: https://data.un.org/
  - location: `./data/raw/UNdata/`

Data from Eurostat:
  - [17] "Capital stocks by industry (NACE Rev.2) and detailed asset type" (indicator nama_10_nfa_st)
    - filename: data export renamed to `eurostat__nama_10_nfa_st__capital_stock.csv`
    - location: `./data/raw/Eurostat/`
    - Accessed on: 2025-02-10
  - [18] "Gross value added and income by detailed industry (NACE Rev.2)" (indicator nama_10_a64)
    - filename: data export renamed to `eurostat__nama_10_a64__value_added.csv`
    - location: `./data/raw/Eurostat/`
    - Accessed on: 2025-02-03
  - [19] "Distribution of population by tenure status, type of household and income group" (indicator ilc_lvho02)
    - values stored in sheet "Eurostat" of `./data/raw/Home_ownership_rates/home_ownership_rates.xlsx` (ilc_lvho02)
    - Accessed on: 2025-02-13
  - URL: https://ec.europa.eu/eurostat/web/main/data/database
  - License: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)
 

[20] OECD home ownership rates
  - Reference: OECD. (2025). _OECD Affordable Housing Database_. Paris: OECD.
  - URL: https://webfs.oecd.org/Els-com/Affordable_Housing_Database/HM1-3-Housing-tenures.xlsx
  - filename: `HM1-3-Housing-tenures.xlsx` 
  - location: values stored in sheet "OECD" of `./data/raw/Home_ownership_rates/home_ownership_rates.xlsx` (ilc_lvho02)
  - Accessed on: 2025-02-13
  - License: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)

[21] CAHF home ownership rates
  - Reference: Centre for Affordable Housing Finance in Africa. (2024). _2024 Yearbook: Housing Finance in Africa_. Johannesburg: Centre for Affordable Housing Finance in Africa, 2024.
  - URL: https://housingfinanceafrica.org/
  - location: values stored in sheet "CAHF" of `./data/raw/Home_ownership_rates/home_ownership_rates.xlsx` (ilc_lvho02)
  - Accessed on: 2025-02-13
  - License: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)

[22] World Bank Country and Lending Groups
  - Reference: World Bank. (2025). _World Bank Country and Lending Groups_. Washington, D.C.: World Bank.
  - URL: https://datacatalogfiles.worldbank.org/ddh-published/0037712/DR0090755/CLASS.xlsx
  - filename: `world_bank_countries.csv`
  - location: `./data/raw/WB_socio_economic_data/API/`
  - Accessed: 2025-10-08
  - License: [Terms and Conditions](https://www.worldbank.org/en/about/legal/terms-and-conditions)

[23] Population headcount at various poverty lines
  - Reference: World Bank (2025), _Poverty and Inequality Platform (version 20250401_2021_01_02_PROD)_ [data set].
  - URL: https://pip.worldbank.org
  - filenames: data exports renamed to `<pl>_povline.csv` with `<pl>` being one of [215, 320, 325, 365, 430, 545, 550, 685, 700, 750, 1000, 1500], corresponding to poverty lines 
    of 2.15, 3.20, 3.25, 3.65, 4.30, 5.45, 5.50, 6.85, 7.00, 7.50, 10.00, and 15.00 USD per day
  - location: `./data/raw/PEB/poverty_data/`
  - Accessed on: 2025-02-05
  - License: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)

[24] Mapping of the GEM Building Taxonomy to the HAZUS Building Taxonomy
  - Reference: Brzev, S., Scawthorn, C., Charleson, A.W., Allen, L., Greene, M., Jaiswal, K., Silva, V. (2013). _GEM Building Taxonomy Version 2.0, GEM Technical Report 2013-02 V1.0.0_. Pavia, Italy: GEM Foundation. doi: 10.13117/GEM.EXP-MOD.TR2013.02
  - URL: https://cloud-storage.globalquakemodel.org/public/wix-new-website/pdf-collections-wix/publications/GEM%20Building%20Taxonomy%20Version%202.0.pdf
  - filename: data from Table D-2 stored as `hazus-gem_mapping.csv`
  - location: `./data/raw/GEM_vulnerability/`
  - Accessed on: 2023-12-19
  - License: [CC BY 3.0 Unported](http://creativecommons.org/licenses/by/3.0/)

### Dynamically downloaded raw data 

The following data sets are downloaded from the World Bank DataBamk by the model when the `recompute` flag is set to `True` in the settings file. 

  - Datasets used:
    - [25] "Income share held by lowest / second / third / fourth / highest 20%" (identifiers SI.DST.FRST.20, SI.DST.02nd.20, SI.DST.03rd.20, SI.DST.04th.20, SI.DST.05th.20)
    - [26] "Population, total" (identifier SP.POP.TOTL)
    - [27] "Personal remittances, received (% of GDP)" (identifier BX.TRF.PWKR.DT.GD.ZS)
    - [28] "Self-employed, total (% of total employment) (modeled ILO estimate)" (identifier SL.EMP.SELF.ZS)
    - [29] "Unemployment, total (% of total labor force) (modeled ILO estimate)" (identifier SL.UEM.TOTL.ZS)
    - [30] "GDP per capita, PPP (constant 2017 international $)" (identifier NY.GDP.PCAP.PP.KD)
    - [31] "GNI per capita, PPP (constant 2017 international $)" (identifier NY.GNP.PCAP.PP.KD)
    - [32] "Coverage in 1st / 2nd / 3rd / 4th / 5th quintile (%) -All Social Protection and Labor" (identifiers 
    per_allsp.cov_q1_tot, per_allsp.cov_q2_tot, per_allsp.cov_q3_tot, per_allsp.cov_q4_tot, per_allsp.cov_q5_tot)
    - [33] "Adequacy of benefits in 1st / 2nd / 3rd / 4th / 5th quintile (%) -All Social Protection and Labor" 
    (identifiers per_allsp.adq_q1_tot, per_allsp.adq_q2_tot, per_allsp.adq_q3_tot, per_allsp.adq_q4_tot, per_allsp.adq_q5_tot)
    - [34] "Coverage in 1st / 2nd / 3rd / 4th / 5th quintile (%) - All Private Transfers" (identifiers 
    per_pr_allpr.cov_q1_tot, per_pr_allpr.cov_q2_tot, per_pr_allpr.cov_q3_tot, per_pr_allpr.cov_q4_tot, per_pr_allpr.cov_q5_tot)
    - [35] "Adequacy of benefits in 1st / 2nd / 3rd / 4th / 5th quintile (%) - All Private Transfers"
      (identifiers per_pr_allpr.adq_q1_tot, per_pr_allpr.adq_q2_tot, per_pr_allpr.adq_q3_tot, per_pr_allpr.adq_q4_tot, per_pr_allpr.adq_q5_tot)
  - Accessed on: 2025-10-08
  - URL: https://databank.worldbank.org
  - License: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)

### Pre-processed data
Preprocessed data to reproduce the manuscript results are provided as `.csv` files in `./data/processed`: 

```
  ├── data/                      
  │   ├── raw/                                  
→ │   └── processed/
  │       └── <...>.csv             
```

All files are re-generated from the raw data files or from dynamically downloaded data when the `force_recompute` flag is set to `True` in the settings file.

A set of pre-processed data is provided in this package, allowing to reproduce the findings of the study. The following table provides information on this data, The data in the present files correspond to variables in the manuscript as follows:

| <file_name> / <column_name>                              | Description                                                                                                                                 | Paper Variable                           | Raw data sources   |
|----------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------|------------------------------------------|--------------------|
| hazard_ratios.csv / fa                                   | Exposure, not accounting for exposure biases. See equation (8).                                                                             | $f^a$                                    | [4, 9, 10]         |
| hazard_ratios.csv / v                                    | Vulnerability, not accounting for early warning. See data sections "Country-level vulnerability" and "Household-level vulnerability".       | $V_{GEM}\cdot v_{q,rel}$                 | [9, 10]            |
| avg_prod_k.csv / avg_prod_k                              | Average productivity of capital. See data section "Capital productivity".                                                                   | $\Pi$                                    | [3]                |
| capital_shares.csv / self_employment                     | Self-employment rate. See data section "Asset shares".                                                                                      | $\gamma_s$                               | [28]               |
| capital_shares.csv / real_est_k_to_va_shares_ratio       | Ratio of real-estate capital share to real-estate value added share. See data section "Asset shares".                                       | $\kappa^r$                               | [17, 18]           |
| capital_shares.csv / k_labor_share                       | Share of capital used for self-employed or employed labor. See data section "Asset shares".                                                 | $1 - \kappa^p - \kappa^r \cdot \gamma_h$ | [14-21]            |
| capital_shares.csv / owner_occupied_share_of_value_added | Share of value added from owner-occupied housing. See data section "Asset shares".                                                          | $\kappa^r \cdot \gamma_h$                | [15, 17-21]        |
| disaster_preparedness.csv / ew                           | Availability of early warning systems. See data section "Early warning".                                                                    | $q_{EW}$                                 | [11, 12]           |
| exposure_bias_per_quintile.csv / exposure_bias           | Exposure bias by income quintile and hazard type. See data section "Exposure bias".                                                         | $EB_q$                                   | [13, 23]           |
| findex_liquidity_and_axfin.csv / liquidity               | Estimate of available savings / liquidity. See data section "Household liquidity".                                                          | $S_q^{sav}$                              | [1]                |
| findex_liquidity_and_axfin.csv / axfin                   | Fraction with savings at a financial institution. See data section "Diversified income".                                                    | $\gamma_q^{fin}$                         | [1]                |
| flopros_protection_processed.csv / MerL_Riv              | Protection level against riverine floods. See data section "Hazard protection".                                                             | $HP_{\text{Flood}}$                      | [5, 7, 8]          |
| flopros_protection_processed.csv / MerL_Co               | Protection level against coastal flooding. See data section "Hazard protection".                                                            | $HP_{\text{Storm surge}}$                | [6, 7, 8]          |
| wb_data_cat_info.csv / income_share                      | Share of national income by income quintile. See data section "Income shares".                                                              | $\frac{c_q}{\sum_q c_q}$                 | [25]               |
| wb_data_cat_info.csv / transfers                         | Income share from social protection and private remittances, not accounting for financial inclusion. See data section "Diversified income". | $\gamma_q^{sp,pt}$                       | [2, 27, 29, 32-35] |
| wb_data_macro.csv / gdp_pc_pp                            | GDP per capita (PPP). See data section "GDP and population".                                                                                | $\sum_q n_q\cdot c_q$                    | [30]               |
| wb_data_macro.csv / pop                                  | Population. See data section "GDP and population".                                                                                          | $P$                                      | [26]               |
| wb_data_macro.csv / gni_pc_pp                            | GNI per capita (PPP). See data section "Household liquidity".                                                                               | $GNIpc$                                  | [31]               |
| wb_data_macro.csv / region                               | Region.                                                                                                                                     |                                          | [22]               |
| wb_data_macro.csv / income_group                         | Income group.                                                                                                                               |                                          | [22]               |

