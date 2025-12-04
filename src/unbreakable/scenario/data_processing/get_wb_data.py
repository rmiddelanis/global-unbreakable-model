"""
  Copyright (c) 2023-2025 Robin Middelanis <rmiddelanis@worldbank.org>

  This file is part of the global Unbreakable model. It is based on
  previous work by Adrien Vogt-Schilb, Jinqiang Chen, Brian Walsh,
  and Jun Rentschler. See https://github.com/walshb1/gRIMM.

  Permission is hereby granted, free of charge, to any person obtaining a copy
  of this software and associated documentation files (the "Software"), to deal
  in the Software without restriction, including without limitation the rights
  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
  copies of the Software, and to permit persons to whom the Software is
  furnished to do so, subject to the following conditions:

  The above copyright notice and this permission notice shall be included in all
  copies or substantial portions of the Software.

  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
  SOFTWARE.
"""


import os
import pandas as pd
from pandas_datareader import wb
import statsmodels.api as sm
from scipy.stats import norm
from scipy.optimize import minimize
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LassoCV
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from unbreakable.misc.helpers import get_country_name_dicts, df_to_iso3, update_data_coverage, get_world_bank_countries
import numpy as np

from datetime import date

YEAR_TODAY = date.today().year

AGG_REGIONS = ['Africa Eastern and Southern', 'Africa Western and Central', 'Arab World', 'Caribbean small states',
               'Central Europe and the Baltics', 'Early-demographic dividend', 'East Asia & Pacific',
               'East Asia & Pacific (excluding high income)', 'East Asia & Pacific (IDA & IBRD countries)',
               'Euro area', 'Europe & Central Asia', 'Europe & Central Asia (excluding high income)',
               'Europe & Central Asia (IDA & IBRD countries)', 'European Union', 'Fragile and conflict affected situations',
               'Heavily indebted poor countries (HIPC)', 'High income', 'IBRD only', 'IDA & IBRD total', 'IDA blend',
               'IDA only', 'IDA total', 'Late-demographic dividend', 'Latin America & Caribbean',
               'Latin America & Caribbean (excluding high income)', 'Latin America & the Caribbean (IDA & IBRD countries)',
               'Least developed countries: UN classification', 'Low & middle income', 'Low income', 'Lower middle income',
               'Middle East & North Africa', 'Middle East & North Africa (excluding high income)',
               'Middle East & North Africa (IDA & IBRD countries)', 'Middle income', 'North America', 'Not classified',
               'OECD members', 'Other small states', 'Pacific island small states', 'Post-demographic dividend',
               'Pre-demographic dividend', 'Small states', 'South Asia', 'South Asia (IDA & IBRD)',
               'Sub-Saharan Africa', 'Sub-Saharan Africa (excluding high income)', 'Sub-Saharan Africa (IDA & IBRD countries)',
               'Upper middle income', 'World']


def get_wb_series(wb_name, colname, wb_raw_data_path, download, start=2000):
    """
    Retrieves a World Bank series and renames its column.

    Args:
        wb_name (str): World Bank indicator name.
        colname (str or float): Column name for the data.

    Returns:
        pd.DataFrame: DataFrame containing the World Bank series.
    """
    return get_wb_df(wb_name, colname, wb_raw_data_path, download, start)[colname]


def get_wb_mrv(wb_name, colname, wb_raw_data_path, download):
    """
    Retrieves the most recent value of a World Bank series.

    Args:
        wb_name (str): World Bank indicator name.
        colname (str): Column name for the data.

    Returns:
        pd.Series: Series containing the most recent values of the World Bank series.
    """
    # if year is not in index, it is already the most recent value
    return get_most_recent_value(get_wb_df(wb_name, colname, wb_raw_data_path, download), drop_year=True)


def get_most_recent_value(data, drop_year=True, dropna=True):
    """
    Extracts the most recent value for each group in the data.

    Args:
        data (pd.DataFrame or pd.Series): Input data.
        drop_year (bool): Whether to drop the year column. Defaults to True.

    Returns:
        pd.Series: Data with the most recent values for each group.
    """
    if 'year' in data.index.names:
        levels_new = data.index.droplevel('year').names
        if dropna:
            res = data.dropna().reset_index()
        else:
            res = data.reset_index()
        if drop_year:
            res = res.loc[res.groupby(levels_new)['year'].idxmax()].drop(columns='year').set_index(levels_new).squeeze()
        else:
            res = res.loc[res.groupby(levels_new)['year'].idxmax()].set_index(levels_new).squeeze()
        return res
    else:
        print("Warning: No 'year' in index names, returning data as is.")
    return data


def get_wb_df(wb_name, colname, wb_raw_data_path, download, start=2000):
    """
    Downloads a World Bank dataset and renames its column.

    Args:
        wb_name (str): World Bank indicator name.
        colname (str): Column name for the data.

    Returns:
        pd.DataFrame: DataFrame containing the World Bank dataset.
    """
    wb_raw_path = os.path.join(wb_raw_data_path, f"{wb_name}.csv")
    metadata_path = os.path.join(wb_raw_data_path, "__metadata.csv")
    if download or not os.path.exists(wb_raw_path):
        # return all values
        wb_raw = wb.download(indicator=wb_name, start=1900, end=YEAR_TODAY, country="all")
        wb_raw.to_csv(wb_raw_path)
        if os.path.exists(metadata_path):
            metadata = pd.read_csv(metadata_path, index_col=0)
            metadata.loc[wb_name, 'last_updated'] = pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
        else:
            metadata = pd.DataFrame(index=[wb_name], columns=['last_updated'])
            metadata.loc[wb_name, 'last_updated'] = pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
        metadata.to_csv(metadata_path)
    else:
        wb_raw = pd.read_csv(wb_raw_path)
        wb_raw = wb_raw.set_index(list(np.intersect1d(wb_raw.columns, ['country', 'year'])))
    if 'year' in wb_raw.index.names:
        # make year index an integer
        wb_raw = wb_raw[~wb_raw.index.get_level_values('year').astype(str).str.contains('W|S')]
        wb_raw.rename(index={y: int(y) for y in wb_raw.index.get_level_values('year').unique()}, level='year', inplace=True)
    if start is not None:
        # filter for start year
        wb_raw = wb_raw[wb_raw.index.get_level_values('year') >= start]
    wb_raw.rename(columns={wb_raw.columns[0]: colname}, inplace=True)
    return wb_raw


def broadcast_to_population_resolution(data, resolution):
    """
    Scales data to a specified population resolution.

    Args:
        data (pd.DataFrame or pd.Series): Input data to be scaled.
        resolution (float): Resolution to scale the data to.

    Returns:
        pd.DataFrame or pd.Series: Data scaled to the specified resolution.

    Raises:
        ValueError: If the input data is not a DataFrame or Series.
    """
    # scale to resolution
    if type(data) == pd.DataFrame:
        return pd.concat([broadcast_to_population_resolution(data[col], resolution) for col in data.columns], axis=1)
    elif type(data) == pd.Series:
        series_name = data.name
        data_ = data.unstack('income_cat').copy()
        data_ = pd.concat([data_] + [data_.rename(columns={c: np.round(c - (i + 1) * resolution, len(str(resolution).split('.')[1])) for c in data_.columns}) for i in range(int(.2 / resolution) - 1)], axis=1)
        data_ = data_.stack().sort_index()
        data_.name = series_name
    else:
        raise ValueError("Data must be a DataFrame or Series")
    return data_


def lorenz_lognormal(p_, sigma_):
    """Lorenz curve for lognormal distribution"""
    # Lorenz curve for lognormal distribution: L(p) = Φ(Φ^(-1)(p) - σ)
    # Irwin, R. J., & Hautus, M. J. (2015). Lognormal Lorenz and normal receiver operating characteristic curves as mirror images. Royal Society Open Science, 2(2), 140280.
    return norm.cdf(norm.ppf(p_) - sigma_)


def lorenz_derivative(p_, sigma_):
    z = norm.ppf(p_)
    return np.exp(sigma_ * z - 0.5 * sigma_ * sigma_)


def upscale_income_resolution(income_shares_, num_quantiles=100):
    def fit_lorenz_sigma(p_, cum_income_):
        """Fit sigma for one country given cumulative points"""

        def objective(sigma):
            pred = lorenz_lognormal(p_[:-1], sigma)  # exclude p=1
            return np.sum((cum_income_[:-1] - pred) ** 2)

        res = minimize(objective, x0=np.array([1]), bounds=[(1e-3, 10)])
        return res.x[0]

    def upscale_country(shares_, p_target_):
        """Upscale from quintiles to arbitrary resolution using Lorenz lognormal"""
        # cumulative income shares
        cum_income = shares_.cumsum().values
        p_data = shares_.index.values.astype(float)

        # fit sigma
        sigma_hat = fit_lorenz_sigma(p_data, cum_income)

        # predict Lorenz curve on target grid
        cum_income_pred = lorenz_lognormal(p_target_, sigma_hat)
        income_pred = np.diff(np.insert(cum_income_pred, 0, 0))  # non-cumulative shares

        return pd.Series(income_pred, index=np.round(p_target, 3)), sigma_hat

    p_target = np.linspace(1 / num_quantiles, 1, num_quantiles)
    res = []
    sigma_values = []
    for iso3, country_data in income_shares_.groupby(level=0):
        country_data = country_data.droplevel('iso3')
        shares_pred, sigma_pred = upscale_country(country_data, p_target)
        shares_pred = pd.DataFrame({"iso3": iso3, "income_cat": p_target.round(3), "income_share": shares_pred})
        sigma_pred = pd.DataFrame({"iso3": [iso3], "sigma": [sigma_pred]})
        res.append(shares_pred)
        sigma_values.append(sigma_pred)
    res = pd.concat(res, ignore_index=True).set_index(['iso3', 'income_cat']).squeeze()
    sigma_values = pd.concat(sigma_values, ignore_index=True).set_index(['iso3']).squeeze()
    return res, sigma_values


def estimate_missing_transfer_shares(transfers_, regression_params_, root_dir_, any_to_wb, wb_raw_data_path, verbose=True,
                                     reg_data_outpath=None, tables_outpath=None, download=False):
    """
    Predicts missing transfer shares using regression models. Regression specification is hard-coded.

    Args:
        transfers_ (pd.DataFrame): DataFrame containing category information with transfer shares.
        root_dir_ (str): Root directory of the project.
        any_to_wb (dict): Mapping of country names to World Bank ISO3 codes.
        verbose (bool): Whether to print verbose output. Defaults to True.
        reg_data_outpath (str, optional): Path to save regression data. Defaults to None.

    Returns:
        pd.DataFrame: Updated category information with predicted transfer shares.
    """

    if regression_params_ is None:
        print("No regression specification provided, skipping transfer share estimation.")
        return transfers_

    load_indices = {
        'REM': 'BX.TRF.PWKR.DT.GD.ZS',
        'UNE': 'SL.UEM.TOTL.ZS',
        'GDP_{pc}': 'NY.GDP.PCAP.PP.KD',
        'gini_index': 'SI.POV.GINI',
    }

    wb_countries = get_world_bank_countries(wb_raw_data_path, download)

    wb_datasets = []
    for key, wb_id in load_indices.items():
        ds = get_wb_series(wb_id, key, wb_raw_data_path, download)
        ds = df_to_iso3(ds.reset_index(), 'country', any_to_wb, verbose).dropna(subset='iso3')
        ds = ds.set_index(list(np.intersect1d(['iso3', 'year'], ds.columns))).drop('country', axis=1)
        if key == 'gdp_pc_pp':
            ds /= 1000  # scale to 1000 USD
        wb_datasets.append(ds)
    wb_datasets = pd.concat([get_most_recent_value(wb_ds) if 'year' in wb_ds.index.names else wb_ds for wb_ds in wb_datasets], axis=1)

    ilo_sp_exp = pd.read_csv(os.path.join(root_dir_, 'data/raw/social_share_regression/ILO_WSPR_SP_exp.csv'),
                             index_col='iso3', na_values=['...', '…']).drop('country', axis=1).rename(columns={'exp_SP_GDP': 'SOC'})
    x = pd.concat([wb_datasets, ilo_sp_exp, wb_countries], axis=1)

    x['FSY'] = False
    fsy_countries = pd.read_csv(os.path.join(root_dir_, 'data/raw/social_share_regression/fsy_countries.csv'),
                                header=None)
    fsy_countries = df_to_iso3(fsy_countries, 0, any_to_wb, verbose).iso3.values
    x.loc[fsy_countries, 'FSY'] = True
    y = transfers_.transfers.unstack('income_cat') * 100
    regression_data = pd.concat([x, y], axis=1).dropna(how='all')

    if reg_data_outpath is not None:
        regression_data.to_csv(reg_data_outpath)

    column_labels = {
        0.2: r'$\gamma_{q=1}^{sp,pt}$',
        0.4: r'$\gamma_{q=2}^{sp,pt}$',
        0.6: r'$\gamma_{q=3}^{sp,pt}$',
        0.8: r'$\gamma_{q=4}^{sp,pt}$',
        1.0: r'$\gamma_{q=5}^{sp,pt}$',
    }

    transfers_predicted = None

    if regression_params_['type'] == 'OLS':
        regression_data = pd.concat(
            [regression_data.drop(columns=['region', 'income_group']),
             pd.get_dummies(regression_data['region'].apply(lambda x_: "I_{" + f"{x_}" + "}")),
             pd.get_dummies(regression_data['income_group'].apply(lambda x_: "I_{" + f"{x_}" + "}"))],
            axis=1
        )

        if 'GDP_{pc}' in regression_data.columns:
            regression_data['GDP_{pc}'] /= 1000  # scale to 1000 USD

        ols_spec = regression_params_['specification']

        var_order = ['GDP_{pc}', 'SOC', 'REM', 'UNE', 'I_{FSY}', 'I_{HICs}', 'I_{UMICs}', 'I_{LMICs}', 'I_{LICs}',
                     'I_{ECA}', 'I_{LAC}', 'I_{NMA}', 'I_{EAP}', 'I_{SAR}', 'I_{SSA}', 'I_{MNA}', 'const']

        model_results = {}

        for income_cat, spec in ols_spec.items():
            variables = spec.split('~')[1].strip().split(' + ')
            regression_data_ = regression_data[variables + [income_cat]].dropna().copy()
            model = sm.OLS(
                endog=regression_data_[[income_cat]].astype(float),
                exog=sm.add_constant(regression_data_.drop(columns=income_cat)).astype(float)
            ).fit()

            # Store model results
            model_results[income_cat] = model

            # Print model summary
            print(f"############ Regression for {income_cat} ############")
            print(model.summary())

            # Predict missing values
            prediction_data_ = regression_data[variables].dropna()
            predicted = model.predict(sm.add_constant(prediction_data_).astype(float)).rename(income_cat)
            if transfers_predicted is None:
                transfers_predicted = predicted
            else:
                transfers_predicted = pd.concat([transfers_predicted, predicted], axis=1)

        # store LaTeX table
        if tables_outpath is not None:
            all_vars = set()
            for model in model_results.values():
                all_vars.update(model.params.index)
            # Sort all_vars according to the order in var_labels
            all_vars = [var for var in var_order if var in all_vars]

            latex_lines = [
                r'\begin{table}[htb]',
                r'\centering',
                r'\small',
                r'  \caption{\textbf{Regression results for imputed income share from social protection and transfers.} Values are in percent.}',
                r'  \label[suptable]{suptab:suptable_7_transfers_regression}',
                r'\smallskip',
                r'\begin{tabular}{l ' + ' '.join(['c'] * len(ols_spec)) + '}',
                r'\toprule',
            ]

            # Column headers
            header_cols = ' & '.join([column_labels[k] for k in sorted(ols_spec.keys())])
            latex_lines.append(f'& {header_cols} \\\\')
            latex_lines.append(r'\midrule')

            # Rows
            for var in all_vars:
                # First row: coefficients with stars
                coef_row = [f"${var}$"]
                se_row = ['']  # row for standard errors
                for key in sorted(ols_spec.keys()):
                    model = model_results.get(key)
                    if model is not None and var in model.params.index:
                        coef = model.params[var]
                        se = model.bse[var]
                        p = model.pvalues[var]
                        # significance stars
                        if p < 0.001:
                            stars = r'\textsuperscript{***}'
                        elif p < 0.01:
                            stars = r'\textsuperscript{**}'
                        elif p < 0.05:
                            stars = r'\textsuperscript{*}'
                        else:
                            stars = ''
                        coef_row.append(f'{coef:.4f}{stars}')
                        se_row.append(f'({se:.3f})')
                    else:
                        coef_row.append('')
                        se_row.append('')
                latex_lines.append(' & '.join(coef_row) + r'\\')
                latex_lines.append(' & '.join(se_row) + r'\\ [1ex]')

            # R^2 and N rows
            r2_row = ['$R^2$']
            n_row = ['$N$']
            for key in sorted(ols_spec.keys()):
                model = model_results.get(key)
                if model is not None:
                    r2_row.append(f'{model.rsquared:.3f}')
                    n_row.append(f'{int(model.nobs)}')
                else:
                    r2_row.append('')
                    n_row.append('')
            latex_lines += [
                r'\midrule',
                ' & '.join(r2_row) + r'\\',
                ' & '.join(n_row) + r'\\',
                r'\bottomrule',
                r'\multicolumn{3}{@{}l@{}}{\footnotesize Note: $^{*}\, p<0.05$; $^{**}\, p<0.01$; $^{***}\, p<0.001$}',
                r'\end{tabular}',
                r'\end{table}',
            ]
            if not os.path.exists(tables_outpath):
                os.makedirs(tables_outpath)
            with open(os.path.join(tables_outpath, 'transfers_regression_ols.tex'), 'w') as f:
                f.write('\n'.join(latex_lines))
    elif regression_params_['type'] == 'LassoCV':
        x_cols = regression_params_['features']
        var_order = ['GDP_{pc}', 'SOC', 'REM', 'UNE', 'FSY', 'HICs', 'UMICs', 'LMICs', 'LICs', 'ECA', 'LAC', 'NMA',
                     'EAP', 'SAR', 'SSA', 'MNA']

        categorical_features = np.intersect1d(x_cols, ['region', 'income_group'])
        numeric_features = np.setdiff1d(x_cols, categorical_features) # treat FSY (binary) as numerical

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numeric_features),
                ('cat', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), categorical_features)
            ]
        )

        models = {}  # stores fitted pipelines
        results = {}  # stores alpha & coefficient tables

        for target in y.columns:
            print(f"\nRunning LassoCV for target: {target}")
            regression_data_ = regression_data[[target] + x_cols].dropna().copy()
            x_ = regression_data_.drop(columns=target)
            y_ = regression_data_[target].astype(float)

            # LassoCV with 5-fold cross-validation
            lasso_cv = LassoCV(
                cv=5,
                random_state=0,
                max_iter=10000
            )

            # Pipeline: preprocessing + Lasso
            model = Pipeline(steps=[
                ('preprocessor', preprocessor),
                ('lasso_cv', lasso_cv)
            ])

            # Fit model
            model.fit(x_, y_)
            models[target] = model

            # Retrieve alpha and coefficients
            best_alpha = model.named_steps['lasso_cv'].alpha_
            coefs = model.named_steps['lasso_cv'].coef_

            # Feature names after preprocessing
            encoder = model.named_steps['preprocessor'].named_transformers_['cat']
            cat_feature_names = encoder.get_feature_names_out(categorical_features)
            feature_names = list(numeric_features) + list(cat_feature_names)

            coef_table = pd.DataFrame({
                'feature': feature_names,
                'coefficient': coefs
            })

            results[target] = {
                'alpha': best_alpha,
                'coef_table': coef_table
            }

            # Predict missing values
            prediction_data_ = regression_data[x_cols].dropna().copy()
            predicted = model.predict(prediction_data_)
            predicted = pd.Series(predicted, index=prediction_data_.index, name=target)
            if transfers_predicted is None:
                transfers_predicted = predicted
            else:
                transfers_predicted = pd.concat([transfers_predicted, predicted], axis=1)

        # store LaTeX table
        if tables_outpath is not None:
            latex_lines = [
                r'\begin{table}[htb]',
                r'\centering',
                r'\small',
                r'  \caption{\textbf{Lasso regression results for imputed income share from social protection and'
                r' transfers.} 5-fold cross-validation was used to select $\alpha$. Numerical predictors were'
                r' standardized (mean=0, SD=1) and coefficients represent change in the outcome per 1 standard deviation'
                r' increase in the predictor. Zero coefficients are omitted. Variables with all-zero coefficients are'
                r' omitted entirely.}',
                r'  \label[suptable]{suptab:suptable_7_transfers_regression}',
                r'\smallskip',
                r'\begin{tabular}{l ' + ' '.join(['c'] * len(results)) + '}',
                r'\toprule',
            ]

            # Column headers
            header_cols = ' & '.join([column_labels[k] for k in sorted(results.keys())])
            latex_lines.append(f'& {header_cols} \\\\')
            latex_lines.append(r'\midrule')

            # Collect all features that have non-zero coefficients in any model
            all_features = set()
            for res in results.values():
                non_zero_features = res['coef_table'][res['coef_table']['coefficient'] != 0]['feature']
                all_features.update(non_zero_features)
            all_features = sorted(all_features)

            # Rows
            for feature in all_features:
                coef_row = [f"${feature}$"]
                for key in sorted(results.keys()):
                    coef_table = results[key]['coef_table']
                    coef_value = coef_table.loc[coef_table['feature'] == feature, 'coefficient']
                    if not coef_value.empty and coef_value.values[0] != 0:
                        coef_row.append(f'{coef_value.values[0]:.4f}')
                    else:
                        coef_row.append('')
                latex_lines.append(' & '.join(coef_row) + r'\\')

            # Const. row
            const_row = [r'const.']
            for key in sorted(results.keys()):
                model = models[key]
                const_value = model.named_steps['lasso_cv'].intercept_
                const_row.append(f'{const_value:.4f}')

            # Alpha row
            alpha_row = [r'$\alpha$']
            for key in sorted(results.keys()):
                alpha_value = results[key]['alpha']
                alpha_row.append(f'{alpha_value:.4f}')

            in_sample_r2_row = [r'$R^2$ (full sample)']
            for key in sorted(results.keys()):
                model = models[key]
                in_sample_r2 = model.score(regression_data[x_cols + [key]].dropna()[x_cols], regression_data[x_cols + [key]].dropna()[key])
                in_sample_r2_row.append(f'{in_sample_r2:.3f}')

            cv_r2_row = [r'$R^2$ (cross-validated)']
            for key in sorted(results.keys()):
                model = models[key]
                cv_r2 = cross_val_score(model, regression_data[x_cols + [key]].dropna()[x_cols], regression_data[x_cols + [key]].dropna()[key], cv=5, scoring='r2').mean()
                cv_r2_row.append(f'{cv_r2:.3f}')

            n_row = [r'$N$']
            for key in sorted(results.keys()):
                n_row.append(f'{len(regression_data[x_cols + [key]].dropna())}')

            latex_lines += [
                ' & '.join(const_row) + r'\\',
                r'\midrule',
                ' & '.join(alpha_row) + r'\\',
                r'\midrule',
                ' & '.join(in_sample_r2_row) + r'\\',
                ' & '.join(cv_r2_row) + r'\\',
                ' & '.join(n_row) + r'\\',
                r'\bottomrule',
                r'\end{tabular}',
                r'\end{table}',
            ]
            if not os.path.exists(tables_outpath):
                os.makedirs(tables_outpath)
            with open(os.path.join(tables_outpath, 'transfers_regression_lassoCV.tex'), 'w') as f:
                f.write('\n'.join(latex_lines))

    transfers_predicted.columns.name = 'income_cat'
    transfers_predicted = transfers_predicted.stack().dropna().sort_index().rename('transfers_predicted')
    transfers_predicted = transfers_predicted.clip(0, 100) / 100

    transfers_ = pd.concat([transfers_, transfers_predicted], axis=1)
    imputed_countries = transfers_[transfers_.transfers.isna() & transfers_.transfers_predicted.notna()].index.get_level_values('iso3').unique()
    available_countries = np.setdiff1d(transfers_.index.get_level_values('iso3').unique(), imputed_countries)
    update_data_coverage(root_dir_, 'transfers', available_countries, imputed_countries)

    transfers_['transfers'] = transfers_['transfers'].fillna(transfers_['transfers_predicted'])
    transfers_.drop('transfers_predicted', axis=1, inplace=True)

    return transfers_


def download_quintile_data(name, id_q1, id_q2, id_q3, id_q4, id_q5, wb_raw_data_path, download, most_recent_value=True,
                           upper_bound=None, lower_bound=None):
    """
    Downloads World Bank quintile data and processes it.

    Args:
        name (str): Name of the data.
        id_q1 (str): World Bank indicator ID for the first quintile.
        id_q2 (str): World Bank indicator ID for the second quintile.
        id_q3 (str): World Bank indicator ID for the third quintile.
        id_q4 (str): World Bank indicator ID for the fourth quintile.
        id_q5 (str): World Bank indicator ID for the fifth quintile.
        most_recent_value (bool): Whether to use the most recent value. Defaults to True.
        upper_bound (float, optional): Upper bound for the data. Defaults to None.
        lower_bound (float, optional): Lower bound for the data. Defaults to None.

    Returns:
        pd.Series: Processed quintile data indexed by country, year, and income category.
    """
    data_q1 = get_wb_series(id_q1, .2, wb_raw_data_path, download)
    data_q2 = get_wb_series(id_q2,.4, wb_raw_data_path, download)
    data_q3 = get_wb_series(id_q3, .6, wb_raw_data_path, download)
    data_q4 = get_wb_series(id_q4, .8, wb_raw_data_path, download)
    data_q5 = get_wb_series(id_q5, 1, wb_raw_data_path, download)
    data = pd.concat([data_q1, data_q2, data_q3, data_q4, data_q5], axis=1).stack().rename(name)
    data.index.names = ['country', 'year', 'income_cat']
    # note: setting upper and lower bounds to nan s.th. the more recent available value is used
    if upper_bound is not None:
        data[data > upper_bound] = np.nan
    if lower_bound is not None:
        data[data < lower_bound] = np.nan
    if most_recent_value:
        data = get_most_recent_value(data)
    return data


def load_pip_data(wb_raw_data_path_, download_, poverty_line_, pip_reference_year_):
    pip_data_name =  f"pip_data_ppp{pip_reference_year_}_povline{poverty_line_}"
    pip_data_path = os.path.join(wb_raw_data_path_, f"{pip_data_name}.csv")
    if download_ or not os.path.exists(pip_data_path):
        pip_url = f"https://api.worldbank.org/pip/v1/pip?country=all&year=all&povline={poverty_line_}&fill_gaps=false&welfare_type=+++welfare_type+++&reporting_level=national&additional_ind=false&ppp_version={pip_reference_year_}&identity=PROD&format=csv"
        pip_data = pd.concat(
            [pd.read_csv(pip_url.replace("+++welfare_type+++", welfare_type)) for welfare_type in
             ['income', 'consumption']],
            ignore_index=True
        )
        pip_data.to_csv(pip_data_path, index=False)

        metadata_path = os.path.join(wb_raw_data_path_, "__metadata.csv")
        if os.path.exists(metadata_path):
            metadata = pd.read_csv(metadata_path, index_col=0)
        else:
            metadata = pd.DataFrame(index=[pip_data_name], columns=['last_updated'])
        metadata.loc[pip_data_name, 'last_updated'] = pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
        metadata.to_csv(metadata_path)
    else:
        pip_data = pd.read_csv(pip_data_path)
    pip_data = pip_data.rename(columns={'country_code': 'iso3', 'reporting_year': 'year'})
    pip_data = pip_data.set_index(['iso3', 'year', 'welfare_type'])
    return pip_data


def get_wb_data(root_dir, include_remittances=True, impute_missing_data=False, regression_params_=None,
                tables_outpath=None, match_years=False, drop_incomplete=True, recompute=True, verbose=True,
                include_poverty_data=False, resolution=.2, download=False, poverty_line=3.0, pip_reference_year=2021):
    """
    Downloads and processes World Bank socio-economic data, including macroeconomic and income-level data.

    Args:
        root_dir (str): Root directory of the project.
        pip_reference_year (int): Reference year for PPP data. Defaults to 2021.
        include_remittances (bool): Whether to include remittance data. Defaults to True.
        impute_missing_data (bool): Whether to impute missing data. Defaults to False.
        drop_incomplete (bool): Whether to drop countries with incomplete data. Defaults to True.
        recompute (bool): Whether to force recomputation of data. Defaults to True.
        verbose (bool): Whether to print verbose output. Defaults to True.
        include_poverty_data (bool): Whether to include shared prosperity line data. Defaults to False.
        resolution (float): Resolution for income shares. Defaults to 0.2.

    Returns:
        tuple: A tuple containing:
            - pd.DataFrame: Macroeconomic data indexed by ISO3 country codes.
            - pd.DataFrame: Category-level data indexed by ISO3 country codes and income categories.
    """
    macro_path = os.path.join(root_dir, "data/processed/wb_data_macro.csv")
    cat_info_path = os.path.join(root_dir, "data/processed/wb_data_cat_info.csv")
    rem_ade_path = os.path.join(root_dir, "data/processed/adequacy_remittances.csv")
    transfers_regr_data_path = os.path.join(root_dir, "data/processed/social_shares_regressors.csv")
    wb_raw_data_path = os.path.join(root_dir, "data/raw/WB_socio_economic_data/API")
    if not recompute and os.path.exists(macro_path) and os.path.exists(cat_info_path):
        print("Loading World Bank data from file...")
        macro_df = pd.read_csv(macro_path, index_col='iso3')
        cat_info_df = pd.read_csv(cat_info_path, index_col=['iso3', 'income_cat'])
        return macro_df, cat_info_df

    print(f"Recomputing World Bank data with{'out' if not download else ''} downloading data...")
    any_to_wb, iso3_to_wb, iso2_iso3 = get_country_name_dicts(root_dir)

    # Load income shares by quintile or decile, upscale depending on resolution
    if resolution == .2:
        # income shares (source: Poverty and Inequality Platform)
        income_shares_unscaled = download_quintile_data(name='income_share', id_q1='SI.DST.FRST.20', id_q2='SI.DST.02nd.20',
                                               id_q3='SI.DST.03rd.20', id_q4='SI.DST.04th.20', id_q5='SI.DST.05th.20',
                                               wb_raw_data_path=wb_raw_data_path, download=download,
                                               most_recent_value=False, upper_bound=100, lower_bound=0) / 100
        # make sure income shares add up to 1
        income_shares_unscaled /= income_shares_unscaled.unstack('income_cat').sum(axis=1)
        income_shares_unscaled = df_to_iso3(income_shares_unscaled.reset_index(), 'country', any_to_wb, verbose).dropna(
            subset='iso3').set_index(['iso3', 'year', 'income_cat']).drop('country', axis=1)
        income_shares_unscaled = get_most_recent_value(income_shares_unscaled.dropna())
        income_shares = income_shares_unscaled
    elif resolution < .2:
        pip_data = load_pip_data(wb_raw_data_path, download, poverty_line, pip_reference_year)

        income_shares_unscaled = pip_data[[f"decile{i}" for i in range(1, 11)]]
        income_shares_unscaled.columns = [round(i / 10, 1) for i in range(1, 11)]
        income_shares_unscaled = income_shares_unscaled.stack().rename('income_share')
        income_shares_unscaled.index.names = ['iso3', 'year', 'welfare_type', 'income_cat']
        income_shares_unscaled = pd.merge(
            income_shares_unscaled.xs('income', level='welfare_type'),
            income_shares_unscaled.xs('consumption', level='welfare_type'),
            left_index=True, right_index=True, how='outer'
        )
        income_shares_unscaled = income_shares_unscaled.income_share_x.fillna(income_shares_unscaled.income_share_y).rename(
            'income_share').to_frame()
        income_shares_unscaled.sort_values(by=['iso3', 'year', 'income_share'])
        income_shares_unscaled = get_most_recent_value(income_shares_unscaled.dropna())

        if resolution != .1:
            income_shares, _ = upscale_income_resolution(income_shares_unscaled, int(1 / resolution))
        else:
            income_shares = income_shares_unscaled
    else:
        raise ValueError("Resolution downscaling not supported")

    # World Development Indicators
    load_indices = {
        'gdp_pc_pp': 'NY.GDP.PCAP.PP.KD',  # Gdp per capita ppp (source: International Comparison Program)
        'gni_pc_pp': 'NY.GNP.PCAP.PP.KD',  # Gni per capita ppp (source: World Development Indicators)
        'pop': 'SP.POP.TOTL',  # population (source: World Development Indicators)
        'gini_index': 'SI.POV.GINI',  # Gini index (source: World Development Indicators)
    }

    wb_datasets = []
    for key, wb_id in load_indices.items():
        ds = get_wb_series(wb_id, key, wb_raw_data_path, download)
        ds = ds.drop(np.intersect1d(ds.index.get_level_values('country').unique(), AGG_REGIONS), level='country' if ds.index.nlevels > 1 else None)
        ds = df_to_iso3(ds.reset_index(), 'country', any_to_wb, verbose).dropna(subset='iso3')
        ds = ds.set_index(list(np.intersect1d(['iso3', 'year'], ds.columns))).drop('country', axis=1)
        wb_datasets.append(ds)

    if include_poverty_data:
        pip_data = load_pip_data(wb_raw_data_path, download, poverty_line, pip_reference_year)
        pip_cols = {
            'headcount': 'extr_pov_rate',
            'poverty_line': 'extr_pov_line',
            'spr': 'soc_pov_rate',
            'spl': 'soc_pov_line'
        }
        for col, var_name in pip_cols.items():
            pip_var_data = pd.merge(
                pip_data.xs('income', level='welfare_type')[col],
                pip_data.xs('consumption', level='welfare_type')[col],
                left_index=True, right_index=True, how='outer'
            )
            pip_var_data = pip_var_data[col+'_x'].fillna(pip_var_data[col+'_y']).rename(var_name).to_frame()
            wb_datasets.append(pip_var_data)

    if not match_years:
        macro_df = pd.concat([get_most_recent_value(wb_ds) if 'year' in wb_ds.index.names else wb_ds for wb_ds in wb_datasets], axis=1)
    else:
        macro_df = get_most_recent_value(pd.concat(wb_datasets, axis=1))

    country_classification = get_world_bank_countries(wb_raw_data_path, download)
    macro_df = pd.concat([macro_df, country_classification], axis=1)

    # calculate adjusted poverty lines that match reported poverty rates
    if include_poverty_data:
        _, sigma_vals = upscale_income_resolution(income_shares_unscaled, num_quantiles=100)
        common_countries = np.intersect1d(macro_df.index.get_level_values('iso3').unique(), sigma_vals.index)
        macro_df['extr_pov_line_adj'] = (lorenz_derivative(macro_df['extr_pov_rate'].loc[common_countries], sigma_vals.loc[common_countries]) * macro_df.loc[common_countries, 'gdp_pc_pp']).clip(lower=poverty_line) / 365
        macro_df['soc_pov_line_adj'] = (lorenz_derivative(macro_df['soc_pov_rate'].loc[common_countries], sigma_vals.loc[common_countries]) * macro_df.loc[common_countries, 'gdp_pc_pp']).clip(lower=macro_df['soc_pov_line']) / 365

    # ASPIRE
    # Adequacies
    # Total transfer amount received by all beneficiaries in a population group as a share of the total welfare of
    # beneficiaries in that group
    adequacy_remittances = download_quintile_data(name='adequacy_remittances', id_q1='per_pr_allpr.adq_q1_tot',
                                                  id_q2='per_pr_allpr.adq_q2_tot', id_q3='per_pr_allpr.adq_q3_tot',
                                                  id_q4='per_pr_allpr.adq_q4_tot', id_q5='per_pr_allpr.adq_q5_tot',
                                                  wb_raw_data_path=wb_raw_data_path, download=download,
                                                  most_recent_value=False, upper_bound=100, lower_bound=0) / 100

    # Total transfer amount received by all beneficiaries in a population group as a share of the total welfare of
    # beneficiaries in that group
    adequacy_all_prot_lab = download_quintile_data(name='adequacy_all_prot_lab', id_q1='per_allsp.adq_q1_tot',
                                                   id_q2='per_allsp.adq_q2_tot', id_q3='per_allsp.adq_q3_tot',
                                                   id_q4='per_allsp.adq_q4_tot', id_q5='per_allsp.adq_q5_tot',
                                                   wb_raw_data_path=wb_raw_data_path, download=download,
                                                   most_recent_value=False, upper_bound=100, lower_bound=0) / 100

    # Coverage
    coverage_remittances = download_quintile_data(name='coverage_remittances', id_q1='per_pr_allpr.cov_q1_tot',
                                                  id_q2='per_pr_allpr.cov_q2_tot', id_q3='per_pr_allpr.cov_q3_tot',
                                                  id_q4='per_pr_allpr.cov_q4_tot', id_q5='per_pr_allpr.cov_q5_tot',
                                                  wb_raw_data_path=wb_raw_data_path, download=download,
                                                  most_recent_value=False, upper_bound=100, lower_bound=0) / 100

    coverage_all_prot_lab = download_quintile_data(name='coverage_all_prot_lab', id_q1='per_allsp.cov_q1_tot',
                                                   id_q2='per_allsp.cov_q2_tot', id_q3='per_allsp.cov_q3_tot',
                                                   id_q4='per_allsp.cov_q4_tot', id_q5='per_allsp.cov_q5_tot',
                                                   wb_raw_data_path=wb_raw_data_path, download=download,
                                                   most_recent_value=False, upper_bound=100, lower_bound=0) / 100

    if include_remittances:
        # fraction of income that is from transfers
        transfers = (coverage_all_prot_lab * adequacy_all_prot_lab + coverage_remittances * adequacy_remittances).rename('transfers')
    else:
        # fraction of income that is from transfers
        transfers = (coverage_all_prot_lab * adequacy_all_prot_lab).rename('transfers')
    transfers = df_to_iso3(transfers.reset_index(), 'country', any_to_wb, verbose)
    transfers = transfers.dropna(subset='iso3').set_index(['iso3', 'year', 'income_cat']).transfers
    transfers = get_most_recent_value(transfers.dropna())

    # store data coverage
    update_data_coverage(root_dir, '__purge__', [], None)
    for v in macro_df.columns:
        update_data_coverage(root_dir, v, macro_df.dropna(subset=v).index.unique(), None)
    update_data_coverage(root_dir, 'income_share', income_shares.unstack('income_cat').dropna().index.unique(), None)
    update_data_coverage(root_dir, 'transfers', transfers.unstack('income_cat').dropna().index.unique(), None)

    if impute_missing_data:
        transfers = estimate_missing_transfer_shares(
            transfers_=transfers.to_frame(),
            regression_params_=regression_params_,
            root_dir_=root_dir,
            any_to_wb=any_to_wb,
            wb_raw_data_path=wb_raw_data_path,
            verbose=verbose,
            reg_data_outpath=transfers_regr_data_path,
            download=download,
            tables_outpath=tables_outpath,
        ).squeeze()

    transfers = broadcast_to_population_resolution(transfers, resolution)

    cat_info_df = pd.concat([income_shares, transfers], axis=1).sort_index()

    complete_macro = macro_df.dropna().index.get_level_values('iso3').unique()
    complete_cat_info = cat_info_df.isna().any(axis=1).replace(True, np.nan).unstack('income_cat').dropna(how='any').index.unique()
    complete_countries = np.intersect1d(complete_macro, complete_cat_info)
    if verbose:
        print(f"Full data for {len(complete_countries)} countries.")
    if drop_incomplete:
        dropped = list(set(list(macro_df.index.get_level_values('iso3').unique()) +
                           list(cat_info_df.index.get_level_values('iso3').unique())) - set(complete_countries))
        if verbose:
            print(f"Dropped {len(dropped)} countries with missing data: {dropped}")
        macro_df = macro_df.loc[complete_countries]
        cat_info_df = cat_info_df.loc[complete_countries]

    macro_df.to_csv(macro_path)
    cat_info_df.to_csv(cat_info_path)
    pd.concat([adequacy_remittances, adequacy_all_prot_lab, coverage_remittances, coverage_all_prot_lab], axis=1).to_csv(rem_ade_path)
    return macro_df, cat_info_df
