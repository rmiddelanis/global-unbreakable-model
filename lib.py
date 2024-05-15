import pycountry as pc
from pandas_helper import load_input_data


def get_country_name_dicts(root_dir):
    # Country dictionaries
    any_to_wb = load_input_data(root_dir, "country_name_mappings/any_name_to_wb_name.csv", index_col="any")
    any_to_wb = any_to_wb[~any_to_wb.index.duplicated(keep='first')]  # drop duplicates
    any_to_wb.dropna(inplace=True)  # drop rows with NaN values

    any_to_wb = any_to_wb.squeeze()

    # iso3 to wb country name table
    iso3_to_wb = load_input_data(root_dir, "country_name_mappings/iso3_to_wb_name.csv").set_index("iso3").squeeze()
    # iso2 to iso3 table
    iso2_iso3 = load_input_data(root_dir, "country_name_mappings/names_to_iso.csv",
                                usecols=["iso2", "iso3"]).drop_duplicates().set_index("iso2").squeeze()

    # rename PWT countries to WB names:
    any_to_wb.loc["Côte d'Ivoire"] = "Cote d'Ivoire"
    any_to_wb.loc["Côte d’Ivoire"] = "Cote d'Ivoire"
    any_to_wb.loc['D.R. of the Congo'] = 'Congo, Dem. Rep.'
    any_to_wb.loc['China, Hong Kong SAR'] = 'Hong Kong SAR, China'
    any_to_wb.loc["Lao People's DR"] = "Lao PDR"
    any_to_wb.loc['China, Macao SAR'] = 'Macao SAR, China'
    any_to_wb.loc["North Macedonia"] = "Macedonia, FYR"
    any_to_wb.loc["Eswatini"] = "Swaziland"
    any_to_wb.loc['U.R. of Tanzania: Mainland'] = 'Tanzania'
    any_to_wb.loc['Taiwan, China'] = 'Taiwan'
    any_to_wb.loc['Czechia'] = 'Czech Republic'
    any_to_wb.loc['Turkiye'] = 'Turkey'

    for _c in any_to_wb.index:
        _c_nospace = _c.replace(' ', '')
        _c_lower = str.lower(_c)
        _c_lower_nospace = str.lower(_c_nospace)
        for __c in [_c_nospace, _c_lower, _c_lower_nospace]:
            if __c != _c:
                any_to_wb.loc[__c] = any_to_wb.loc[_c]

    # add values to iso3_to_wb
    iso3_to_wb['KSV'] = 'Kosovo'
    iso3_to_wb['XKX'] = 'Kosovo'
    iso3_to_wb['ROM'] = 'Romania'
    iso3_to_wb['ZAR'] = 'Congo, Dem. Rep.'
    iso3_to_wb['WBG'] = 'West Bank and Gaza'

    return any_to_wb, iso3_to_wb, iso2_iso3


def df_to_iso3(df_, column_name_, any_to_wb_=None, verbose_=True):
    if 'iso3' in df_:
        raise Exception("iso3 column already exists")

    def get_iso3(name):
        # hard coded country names:
        hard_coded = {
            'congo, dem. rep.': 'COD',
            'congo, democratic republic': 'COD',
            'democratic republic of the congo': 'COD',
            'congo, rep.': 'COG',
            'congo brazzaville': 'COG',
            'congo, republic of the': 'COG',
            'cape verde': 'CPV',
            'côte d’ivoire': 'CIV',
            'côte d\'ivoire': 'CIV',
            'hong kong sar, china': 'HKG',
            'china, hong kong special administrative region': 'HKG',
            'hong kong, china': 'HKG',
            'macao sar, china': 'MAC',
            'china, macao special administrative region': 'MAC',
            'macau, china': 'MAC',
            'macao, china': 'MAC',
            'taiwan, china': 'TWN',
            'korea, rep.': 'KOR',
            'republic of korea': 'KOR',
            'korea, south': 'KOR',
            "korea, dem. people's rep.": 'PRK',
            'korea, dem. rep.': 'PRK',
            'korea (the democratic people\'s republic of)': 'PRK',
            'st. vincent and the grenadines': 'VCT',
            'st. vincent ': 'VCT',
            'swaziland': 'SWZ',
            'bolivia (plurinational state of)': 'BOL',
            'faeroe islands': 'FRO',
            'iran (islamic republic of)': 'IRN',
            'iran, islamic rep.': 'IRN',
            'iran': 'IRN',
            'micronesia (federated states of)': 'FSM',
            'micronesia, fed. sts': 'FSM',
            'micronesia, fed. sts.': 'FSM',
            'the former yugoslav republic of macedonia': 'MKD',
            'macedonia, fyr': 'MKD',
            'united states virgin islands': 'VIR',
            'virgin islands (u.s.)': 'VIR',
            'venezuela (bolivarian republic of)': 'VEN',
            'venezuela, rb': 'VEN',
            'netherlands antilles': 'ANT',
            'st. kitts and nevis': 'KNA',
            'st. lucia': 'LCA',
            'st. martin (french part)': 'MAF',
            'bahamas, the': 'BHS',
            'curacao': 'CUW',
            'egypt, arab rep.': 'EGY',
            'gambia, the': 'GMB',
            'lao pdr': 'LAO',
            'turkiye': 'TUR',
            'turkey (turkiye)': 'TUR',
            'türkiye': 'TUR',
            'west bank and gaza': 'PSE',
            'gaza strip': 'PSE',
            'west bank': 'PSE',
            'occupied palestinian territory': 'PSE',
            'yemen, rep.': 'YEM',
            'kosovo': 'XKX',
            'tanzania, united rep.': 'TZA',
            'bosnia herzegovina': 'BIH',
        }
        if str.lower(name) in hard_coded:
            return hard_coded[str.lower(name)]

        # try to find in pycountry
        if pc.countries.get(name=name) is not None:
            return pc.countries.get(name=name).alpha_3
        elif pc.countries.get(common_name=name) is not None:
            return pc.countries.get(common_name=name).alpha_3
        elif pc.countries.get(official_name=name) is not None:
            return pc.countries.get(official_name=name).alpha_3
        else:
            try:
                fuzzy_search_res = pc.countries.search_fuzzy(name)
            except LookupError:
                if any_to_wb_ is not None and name in any_to_wb_.index and any_to_wb_.loc[name] != name:
                    if verbose_:
                        print(f"Warning: {name} not found in pycountry, but found in any_to_wb. Retry with "
                              f"{any_to_wb_.loc[name]}")
                    return get_iso3(any_to_wb_.loc[name])
                else:
                    if verbose_:
                        print(f"Warning: {name} not found in pycountry, and fuzzy search failed")
                    return None
            if len(fuzzy_search_res) == 1:
                if verbose_:
                    print(f"Warning: {name} not found in pycountry, but fuzzy search found {fuzzy_search_res[0].name}")
                return fuzzy_search_res[0].alpha_3
            elif len(fuzzy_search_res) > 1:
                if verbose_:
                    print(f"Warning: {name} not found in pycountry, but fuzzy search found multiple matches: {fuzzy_search_res}")
                return None

    df = df_.copy()
    df['iso3'] = df[column_name_].apply(lambda x: get_iso3(x))
    if df.iso3.isna().any() and verbose_:
        print(f"Warning: ISO3 could not be found for {len(df[df.iso3.isna()].country.unique())} countries.")
    return df
