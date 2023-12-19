import os
import numpy as np
import pandas as pd
from functools import partial
from itertools import product
import geopandas as gpd
import matplotlib.pyplot as plt

import tqdm

# possible materials (level 1) for GEM taxonomy; from GEM Building Taxonomy Version 2.0
GEM_LAT_LOAD_MAT_NAMES = {'MAT99': 'Unknown material', 'C99': 'Concrete, unknown reinforcement',
                          'CU': 'Concrete, unreinforced', 'CR': 'Concrete, reinforced',
                          'SRC': 'Concrete, composite with steel section', 'S': 'Steel', 'ME': 'Metal (except steel)',
                          'M99': 'Masonry, unknown reinforcement', 'MUR': 'Masonry, unreinforced',
                          'MCF': 'Masonry, confined', 'MR': 'Masonry, reinforced',
                          'E99': 'Earth, unknown reinforcement',
                          'EU': 'Earth, unreinforced', 'ER': 'Earth, reinforced', 'W': 'Wood', 'MATO': 'Other material'}

GEM_LAT_LOAD_MAT_FIRST_LEVEL = ['MAT99', 'C99', 'CU', 'CR', 'SRC', 'S', 'ME', 'M99', 'MUR', 'MCF', 'MR', 'E99', 'EU',
                                'ER', 'W', 'MATO']
GEM_LAT_LOAD_MAT_SECOND_LEVEL = ['CT99', 'CIP', 'PC', 'CIPPS', 'PCPS', 'S99', 'SL', 'SR', 'SO', 'ME99', 'MEIR', 'MEO',
                                 'MUN99', 'ADO', 'ST99', 'STRUB', 'STDRE', 'CL99', 'CLBRS', 'CLBRH', 'CLBLH', 'CB99',
                                 'CBS', 'CBH', 'MO', 'MR99', 'RS', 'RW', 'RB', 'RCM', 'RBC', 'ET99', 'ETR', 'ETC',
                                 'ETO',
                                 'W99', 'WHE', 'WLI', 'WS', 'WWD', 'WBB', 'WO']
GEM_LAT_LOAD_MAT_THIRD_LEVEL = ['SC99', 'WEL', 'RIV', 'BOL', 'MO99', 'MON', 'MOM', 'MOL', 'MOC', 'MOCL', 'SP99', 'SPLI',
                                'SPSA', 'SPTU', 'SPSL', 'SPGR', 'SPBA', 'SPO']

GEM_LAT_LOAD_SYS_FIRST_LEVEL = ['L99', 'LN', 'LFM', 'LFINF', 'LFBR', 'LPB', 'LWAL', 'LDUAL', 'LFLS', 'LFLSINF', 'LH',
                                'LO']
GEM_LAT_LOAD_SYS_SECOND_LEVEL = ['DU99', 'DUC', 'DNO', 'DBD']

GEM_HEIGHT_FIRST_LEVEL = ['H99', 'H', 'HBET', 'HEX', 'HAPP']  # only hight above ground is necessary, ignore HB, HF

# manually add materials that are not in the GEM taxonomy but appear in the dataset:
GEM_LAT_LOAD_MAT_FIRST_LEVEL.append('M')  # Masonry (assumed due to 'M+ST' materials, occurs only in India)
GEM_LAT_LOAD_MAT_FIRST_LEVEL.append('ST')  # Stone, unknown technology (=ST99) (assumed)
GEM_LAT_LOAD_MAT_FIRST_LEVEL.append('CB')  # Concrete block, unknown type (=CB99)
GEM_LAT_LOAD_MAT_FIRST_LEVEL.append('INF')  # Informal (assumed)
GEM_LAT_LOAD_MAT_FIRST_LEVEL.append('MIX(C-S)')  # Mix of Concrete (assumed) and Steel
GEM_LAT_LOAD_MAT_FIRST_LEVEL.append('MIX(C-W)')  # Mix of Concrete and Wood
GEM_LAT_LOAD_MAT_FIRST_LEVEL.append('MIX(M-W)')  # Mix of Masonry and Wood
GEM_LAT_LOAD_MAT_FIRST_LEVEL.append('MIX(S-W)')  # Mix of Steel and Wood
GEM_LAT_LOAD_MAT_FIRST_LEVEL.append('MIX(CR-W)')  # Mix of Concrete, reinforced and Wood
GEM_LAT_LOAD_MAT_FIRST_LEVEL.append('MIX(M-ST)')  # Mix of Masonry and Stone, unknown technology (=ST99)
GEM_LAT_LOAD_MAT_FIRST_LEVEL.append('MIX(W-M)')  # Mix of Wood and Masonry
GEM_LAT_LOAD_MAT_FIRST_LEVEL.append('MIX(W-EU)')  # Mix of Wood and Earth, unknown reinforcement (=E99)
GEM_LAT_LOAD_MAT_FIRST_LEVEL.append('MIX(MUR-W)')  # Mix of Masonry, unreinforced and Wood
GEM_LAT_LOAD_MAT_FIRST_LEVEL.append('MIX(MUR-CR)')  # Mix of Masonry, unreinforced and Concrete, reinforced
GEM_LAT_LOAD_MAT_FIRST_LEVEL.append('MIX(MUR-STDRE-W)')  # Mix of Masonry, unreinforced + Dressed stone masonry and Wood
GEM_LAT_LOAD_MAT_FIRST_LEVEL.append('MIX(MUR-STRUB-W)')  # Mix of Masonry, unreinforced + Rubble (field stone) or semi-dressed stone and Wood
GEM_LAT_LOAD_MAT_FIRST_LEVEL.append('MIX(S-CR-PC)')  # Mix of Steel + Concrete, reinforced + Precast concrete
GEM_LAT_LOAD_MAT_FIRST_LEVEL.append('MIX(MR-W)')  # Mix of Masonry, reinforced and Wood
GEM_LAT_LOAD_MAT_FIRST_LEVEL.append('MIX(S-CR)')  # Mix of Steel and Concrete, reinforced
GEM_LAT_LOAD_MAT_FIRST_LEVEL.append('MIX')  # Mix of materials

# GEM_LAT_LOAD_SYS_FIRST_LEVEL.append('UNK')
GEM_LAT_LOAD_MAT_SECOND_LEVEL.append('ST')  # Stone, unknown technology (=ST99)
GEM_LAT_LOAD_MAT_SECOND_LEVEL.append('CB')  # Concrete blocks, unknown type (=CB99)

FIELD_VALUE_TO_TYPE_MAP = {
    **{v: 'lat_load_mat' for v in GEM_LAT_LOAD_MAT_FIRST_LEVEL},
    **{v: 'lat_load_mat' for v in GEM_LAT_LOAD_MAT_SECOND_LEVEL},
    **{v: 'lat_load_mat' for v in GEM_LAT_LOAD_MAT_THIRD_LEVEL},
    **{v: 'lat_load_sys' for v in GEM_LAT_LOAD_SYS_FIRST_LEVEL},
    **{v: 'lat_load_sys' for v in GEM_LAT_LOAD_SYS_SECOND_LEVEL},
    **{v: 'height' for v in GEM_HEIGHT_FIRST_LEVEL}
}

HAZUS_COUNTRIES = ['VIR', 'PRI', 'CAN', 'USA']

VULNERABILITY_MAPPING = {
    # setting any adobe block structure to fragile
    'MUR': 'fragile',  # Masonry, unreinforced
    'MUR+MO': 'fragile',  # Masonry, unreinforced (+ mortar unknown)
    'MUR+ADO': 'fragile',  # Masonry, unreinforced + adobe blocks
    'MUR+ADO+MOC': 'fragile',  # Masonry, unreinforced + adobe blocks + cement mortar
    'MUR+ADO+MOM': 'fragile',  # Masonry, unreinforced + adobe blocks + mud mortar
    'MUR+STRUB': 'fragile',  # Masonry, unreinforced + Rubble (field stone) or semi-dressed stone + mortar unknown
    'MUR+STRUB+MON': 'fragile',  # Masonry, unreinforced + Rubble (field stone) or semi-dressed stone + no mortar
    'MUR+STRUB+MOM': 'fragile',  # Masonry, unreinforced + Rubble (field stone) or semi-dressed stone + mud mortar
    'MUR+STRUB+MOL': 'fragile',  # Masonry, unreinforced + Rubble (field stone) or semi-dressed stone + lime mortar
    'MUR+STRUB+MOC': 'fragile',  # Masonry, unreinforced + Rubble (field stone) or semi-dressed stone + cement mortar
    'MUR+STDRE': 'fragile',  # Masonry, unreinforced + Dressed stone masonry + mortar unknown
    'MUR+STDRE+MOM': 'fragile',  # Masonry, unreinforced + Dressed stone masonry + mud mortar
    'MUR+STDRE+MOL': 'median',  # Masonry, unreinforced + Dressed stone masonry + lime mortar
    'MUR+STDRE+MOC': 'median',  # Masonry, unreinforced + Dressed stone masonry + cement mortar
    'MUR+CL': 'fragile',  # Masonry, unreinforced + Fired clay unit, unknown type (+ mortar unknown)
    'MUR+CLBRS': 'fragile',  # Masonry, unreinforced + Fired clay solid bricks (+ mortar unknown)
    'MUR+CLBRS+MOM': 'fragile',  # Masonry, unreinforced + Fired clay solid bricks + mud mortar
    'MUR+CLBRS+MOL': 'median',  # Masonry, unreinforced + Fired clay solid bricks + lime mortar
    'MUR+CLBRS+MOC': 'median',  # Masonry, unreinforced + Fired clay solid bricks + cement mortar
    'MUR+CLBRH': 'fragile',  # Masonry, unreinforced + Fired clay hollow bricks (+ mortar unknown)
    'MUR+CB99+MOC': 'median',  # Masonry, unreinforced + Concrete block + cement mortar ==> PAGER UCB

    # here, assume as observed with previous mappings, that mud mortar is fragile, lime and cement mortar are median
    'MUR+ST': 'fragile',  # Masonry, unreinforced + Stone, unknown technology (assuming ST99) + mortar unknown
    'MUR+ST+MOM': 'fragile',  # Masonry, unreinforced + Stone, unknown technology (assuming ST99) + mud mortar
    'MUR+ST+MOL': 'median',  # Masonry, unreinforced + Stone, unknown technology (assuming ST99) + lime mortar
    'MUR+ST+MOC': 'median',  # Masonry, unreinforced + Stone, unknown technology (assuming ST99) + cement mortar

    'MUR+CLBLH': 'fragile',  # Masonry, unreinforced + Fired clay hollow blocks (+ mortar unknown)
    'MUR+CB': 'fragile',  # Masonry, unreinforced + Concrete block + (mortar unknown) # TODO: could also be median, depending on mortar type
    'MUR+CBS': 'fragile',  # Masonry, unreinforced + Concrete block, solid + (mortar unknown)
    'MUR+CBH': 'fragile',  # Masonry, unreinforced + Concrete block, hollow + (mortar unknown)
    'MUR+CBR': 'fragile',  # Masonry, unreinforced + (CBR not in spec, some type of concrete block) + (mortar unknown)

    'E+ETO': 'fragile',  # Earth, unknown reinforcement + Earth technology other (E not in spec, assume E99) # TODO fragile or median?
    'EU': 'fragile',  # Earth, unreinforced TODO fragile or median?
    'EU+ETC': 'fragile',  # Earth, unreinforced + cob or wet construction ==> PAGER M1
    'EU+ETR': 'median',  # Earth, unreinforced + rammed earth
    'ER+ETR': 'median',  # Earth, reinforced + rammed earth

    'MCF': 'robust',  # Masonry, confined
    'MCF+CB': 'robust',  # Masonry, confined + Concrete block, unknown type (assuming CB99)
    'MCF+CBH': 'robust',  # Masonry, confined + Concrete block, hollow
    'MCF+CBS': 'robust',  # Masonry, confined + Concrete block, solid
    'MCF+CBR': 'robust',  # Masonry, confined + (CBR not in spec, some type of concrete block)
    'MCF+CL': 'robust',  # Masonry, confined + Fired clay unit, unknown type (assuming CL99)
    'MCF+CF': 'robust',  # Masonry, confined + (CF not in spec)
    'MCF+CLBRS': 'robust',  # Masonry, confined + Fired clay solid bricks
    'MCF+CLBRH': 'robust',  # Masonry, confined + Fired clay hollow bricks
    'MCF+CLBLH': 'robust',  # Masonry, confined + Fired clay hollow blocks or tiles
    'MCF+S': 'robust',  # Masonry, confined + Steel

    'MR': 'robust',  # Masonry, reinforced
    'MR+CB': 'robust',  # Masonry, reinforced + Concrete block, unknown type (assuming CB99)
    'MR+CBR': 'robust',  # Masonry, reinforced + (CBR not in spec, some type of concrete block)
    'MR+CBH': 'robust',  # Masonry, reinforced + Concrete block, hollow
    'MR+CL': 'robust',  # Masonry, reinforced + Fired clay unit, unknown type (assuming CL99)
    'MR+STRUB+RCB+MOC': 'median',  # Masonry, reinforced + Rubble (field stone) or semi-dressed stone +
                                   # reinforced concrete block + cement mortar

    'CR': 'robust',  # Concrete, reinforced ==> PAGER 'C'
    'CR+CIP': {  # Concrete, reinforced + Cast in place
        'LDUAL': {'HBET:3,1': 'median', 'default': 'robust'},
        'LFINF': {'HBET:3,1': 'median', 'default': 'robust'},
        'LFM': {'HBET:3,1': 'median', 'default': 'robust'},
        'default': 'robust',  # other lat load sys values: None, LWAL
    },
    'CR+PCPS': {  # Concrete, reinforced + Precast prestressed concrete
        'LFM': {'HBET:3,1': 'median', 'default': 'robust'},  # define exception as for CR+CIP
        'default': 'robust',  # only other lat load system is 'LWAL'
    },
    'CR+PC': {  # Concrete, reinforced + Precast concrete
        'LDUAL': {'HBET:3,1': 'median', 'default': 'robust'},
        'LFINF': {'HBET:3,1': 'median', 'default': 'robust'},
        'LFM': {'HBET:3,1': 'median', 'default': 'robust'},
        'default': 'robust',  # TODO: What about LFM, LPB which also occur in GEM data but not in PAGER vulnerabilities?
    },

    'S': 'robust',  # Steel
    'S+SL': 'robust',  # Steel + light-weight steel members
    'SL': 'robust',  # Steel, light-weight steel members
    'S+SR': 'robust',  # Steel + regular-weight steel members
    'SR': 'robust',  # Steel, regular-weight steel members
    'S+SO': 'robust',  # Steel + other steel members
    'SRC': 'robust',  # Concrete, composite with steel section

    'W': 'median',  # Wood ==> PAGER 'W'
    'W+WWD': 'fragile',  # Wood + Wattle and daub
    'W+WBB': 'fragile',  # Wood + Bamboo
    'W+WO': 'fragile',  # Wood + Wood other
    'W+WS': 'fragile',  # Wood + Solid Wood ==> PAGER W4
    'W+WLI': {  # Wood + Light wood members
        'LPB': 'fragile',
        'LWAL': 'median',  # PAGER MH (mobile homes) maps to W+WLI/LWAL/HBET:1,2; but it does not seem useful to distinguish here
        'LFBR': 'fragile',  # TODO discuss
        'LFM': 'fragile',  # according to the documentation, this is not a valid system for 'Wood'; TODO discuss vulnerability
        'default': 'fragile',  # TODO discuss
    },
    'W+WHE': {  # Wood + Heavy wood
        'LPB': 'median',  # PAGER 'W2'
        'LWAL': 'fragile',  # PAGER 'W6'
        'default': 'fragile',  # TODO discuss
    },
    'ME': 'fragile',  # Metal (except steel) TODO: should metal always be fragile?
    'ME+MEO': 'fragile',  # Metal (except steel) + other metal TODO: discuss
    'ME+MEIR': 'fragile',  # Metal (except steel) + iron TODO: discuss

    'M+ADO': 'fragile',  # Masonry (unknown reinforcement) + adobe (M not in spec, assume M99)
    'M+ST': 'fragile',  # Masonry + Stone, unknown technology (=ST99); occurs only in India for Insutrial and Commercial use
    'M+CB': 'median',  # Masonry + Concrete block, unknown type (=SB99); occurs only in Japan TODO: discuss

    # here follow all the mixed types. TODO: discuss
    'W+S': 'median',  # Wood + Steel; occurs only in the Netherlands and for Industrial use
    'MIX(MUR-STRUB-W)': 'fragile',  # Mix of Masonry, unreinforced + Rubble (field stone) or semi-dressed stone and Wood
    'MIX(MUR-STDRE-W)': 'fragile',  # Mix of Masonry, unreinforced + Dressed stone masonry and Wood
    'MIX(MUR-W)': 'fragile',  # Mix of Masonry, unreinforced and Wood
    'MIX(MUR-CR)': 'median',  # Mix of Masonry, unreinforced and Concrete, reinforced; occurs only in Switzerland and Germany TODO: discuss
    'MIX(S-CR-PC)': 'robust',  # Mix of Steel + Concrete, reinforced + Precast concrete; occurs only in Australia
                               # and for Industrial use
    'MIX(S-W)': 'median',  # Mix of Steel and Wood, occurs only in India and with lateral load system 'LO' (other)
    'MIX(S-CR)': 'robust',  # Mix of Steel and Concrete, reinforced; occurs only in Australia and for industrial use
    'MIX(C-S)': 'robust',  # Mix of Concrete and Steel; occurs only in India and with lateral load system 'LO' (other)
    'MIX(C-W)': 'median',  # Mix of Concrete and Wood; occurs only in India and with lateral load system 'LO' (other)
    'MIX(M-W)': 'fragile',  # Mix of Masonry and Wood; occurs only with lateral load system 'LO' (other) or NaN
    'MIX(M-ST)': 'fragile',  # Mix of Masonry and Stone, unknown technology (=ST99); occurs only in India and with
                             # lateral load system 'LO' (other)
    'MIX(W-M)': 'fragile',  # Mix of Wood and Masonry; occurs only in India and with lateral load system 'LO' (other)
    'MIX(W-EU)': 'fragile',  # Mix of Wood and Earth, unknown reinforcement (=E99); occurs only in Pakistan
    'MIX(CR-W)': 'median',  # Mix of Concrete, reinforced and Wood; only entries with lateral load system NaN; use
                            # 'median', as concrete, reinforced is typically robust
    'MIX(MR-W)': 'median',  # Mix of Masonry, reinforced and Wood; occurs only in Russia and only for residential use
    'MIX': 'median',  # Mix of materials; occurs only in Kosovo, Slovenia, Bulgaria, Austria, and with hybrid lateral
                      # load resistance system
    'MATO': 'fragile',  # Other material
    'INF': 'fragile',  # Informal; occurs only on Fiji, Solomon Islands, Vanatu, lat_load_sys always NaN TODO: discuss
    'UNK': 'median',  # material unknown / not specified
}

# add mapping for occurring full material strings
VULNERABILITY_MAPPING['S+S99+SC99'] = VULNERABILITY_MAPPING['S']
VULNERABILITY_MAPPING['S+SL+SC99'] = VULNERABILITY_MAPPING['S+SL']
VULNERABILITY_MAPPING['MR+MUN99+MR99+MO99'] = VULNERABILITY_MAPPING['MR']
VULNERABILITY_MAPPING['MUR+MUN99+MO99'] = VULNERABILITY_MAPPING['MUR']


def assign_vulnerability(material, resistance_system, height, mapping):
    """
    This function assigns a vulnerability to a given GEM taxonomy.

    Parameters:
    gem_taxonomy (str): The GEM taxonomy.
    mapping (dict): The mapping between GEM taxonomy and vulnerability.

    Returns:
    str: The vulnerability.
    """
    if material in mapping.keys():
        material_vulnerability = mapping[material]
        if type(material_vulnerability) is str:
            return material_vulnerability
        if type(resistance_system) is str and len(resistance_system) > 0:
            resistance_system = resistance_system.split('+')[0]
            if resistance_system in material_vulnerability.keys():
                resistance_system_vulnerability = material_vulnerability[resistance_system]
                if type(resistance_system_vulnerability) is str:
                    return resistance_system_vulnerability
                else:
                    if type(height) is str and len(height) > 0:
                        height = height.split(':')[1].split('+')[0].split('-' if '-' in height else ',')
                        if len(height) > 1 and len(height[1]) == 0 or len(height) == 1:
                            height = [height[0], height[0]]
                        try:
                            height = [int(h) for h in height]
                        except ValueError as e:
                            print(f"Warning: could not parse height value {height} to integer. Using default value.")
                            return resistance_system_vulnerability['default']
                        for key in resistance_system_vulnerability.keys():
                            if key != 'default':
                                h_range = sorted([int(h) for h in key.split(':')[1].split(',')])
                                if h_range[0] <= height[0] <= h_range[1] or h_range[0] <= height[1] <= h_range[1]:
                                    return resistance_system_vulnerability[key]
                return resistance_system_vulnerability['default']
        return material_vulnerability['default']
    # raise ValueError(f"Unknown material {material} for taxonomy {gem_taxonomy}.")
    print(f"Could not assign vulnerability for unknown material {material}.")
    return 'unknown'


def gather_gem_data(gem_repo_root_dir, hazus_gem_mapping_path):
    # Initialize an empty DataFrame
    gem_data = pd.DataFrame()

    vars_to_keep = {'ID_0': 'iso3', 'NAME_0': 'country', 'OCCUPANCY': 'building_type',
                    'MACRO_TAXO': 'macro_taxonomy', 'TAXONOMY': 'taxonomy', 'BUILDINGS': 'n_buildings',
                    'DWELLINGS': 'n_dwellings', 'OCCUPANTS_PER_ASSET': 'occupants_per_asset',
                    'TOTAL_AREA_SQM': 'total_area_sqm'}
    index_vars = ['ID_0', 'NAME_0', 'OCCUPANCY', 'MACRO_TAXO', 'TAXONOMY']

    # Walk through root_dir
    for dirpath, dirnames, filenames in os.walk(gem_repo_root_dir):
        for filename in filenames:
            # Check if the file is 'Exposure_Summary_Taxonomy.csv'
            if filename == 'Exposure_Summary_Taxonomy.csv':
                # Construct the full file path
                file_path = os.path.join(dirpath, filename)

                # Read the file into a DataFrame
                df = pd.read_csv(file_path)

                vars_diff = np.setdiff1d(list(vars_to_keep.keys()), df.columns)
                if len(vars_diff) > 0:
                    print(f"Warning: the following variables aren't available for country {os.path.basename(dirpath)}: "
                          + ", ".join(list(vars_diff)))
                df = df[list(set(vars_to_keep.keys()) - set(vars_diff))].groupby(index_vars).sum().reset_index()

                # Append df to gem_data
                gem_data = pd.concat([gem_data, df])
    gem_data.rename(vars_to_keep, axis=1, inplace=True)
    gem_data.reset_index(inplace=True, drop=True)

    replace_strings = {s: s.replace('+', '-') for s in
                       ['MIX(MUR+W)', 'MIX(MR+W)', 'MIX(S+CR)', 'MIX(MUR+CR)', 'MIX(W+EU)', 'MIX(MUR+STRUB+W)',
                        'MIX(MUR+STDRE+W)', 'MIX(S+CR+PC)']}
    for s, r in replace_strings.items():
        gem_data.taxonomy = gem_data.taxonomy.apply(lambda x: x.replace(s, r))

    # handle countries with HAZUS taxonomy:
    # countries that use HAZUS have a taxonomy of the form "{occupancy}-{HAZUS id/[height]}-{Design Code}"
    # use hazus id to replace taxonomy with respective GEM taxonomy string
    # some entries come with an additional height value. This is mostly consistent with the GEM taxonomy strings as
    # per the GEM Building Taxonomy Version 2.0 table D-2, except for some W1 Hazus IDs, which have height > 2 stories
    # information on Design Codes can be found in the following document (page 2-4, section 2.3):
    # https://www.fema.gov/sites/default/files/2020-09/fema_hazus_advanced-engineering-building-module_user-manual.pdf
    hazus_gem_mapping = pd.read_csv(hazus_gem_mapping_path, index_col=0).astype(str)
    # set mobile homes to informal, s.th. vulnerability will be fragile
    hazus_gem_mapping.loc['MH', 'gem_str'] = 'INF/'
    # W3 and W4 are not allowed as per Hazus documentation, but occur in the dataset; setting to general 'Wood'
    hazus_gem_mapping.loc['W3', 'gem_str'] = 'W/'
    hazus_gem_mapping.loc['W4', 'gem_str'] = 'W/'
    gem_data.loc[gem_data.iso3.isin(HAZUS_COUNTRIES), 'taxonomy'] = (
        gem_data.loc[gem_data.iso3.isin(HAZUS_COUNTRIES), 'taxonomy'].apply(
            lambda x: hazus_gem_mapping.loc[x.split('-')[1].split('/')[0], 'gem_str']
        )
    )

    unique_tax_strings = gem_data.taxonomy.unique()
    decoded_tax_strings = pd.concat(
        [decode_taxonomy(t, keep_unknown=False, verbose=False)
         for t in tqdm.tqdm(unique_tax_strings, desc="decoding taxonomy strings")]
    )
    res = pd.merge(gem_data, decoded_tax_strings, how='left', on='taxonomy')

    # set material to 'UNK' if Lateral load resisting system value = 'LN' (No lateral load-resisting system)
    res.lat_load_mat[(res.lat_load_mat.isna()) & (res.lat_load_sys.apply(lambda x: 'LN' in x if type(x) is str else False))] = 'UNK'
    # if taxonomy starts with 'UNK', assume this is the material code and set material to 'UNK'
    res.lat_load_mat[(res.lat_load_mat.isna()) & (res.taxonomy.apply(lambda x: x.startswith('UNK')))] = 'UNK'

    res['vulnerability'] = res.apply(lambda x: assign_vulnerability(x.lat_load_mat, x.lat_load_sys, x.height,
                                                                    VULNERABILITY_MAPPING), axis=1)

    return res


def get_material_from_taxonomy(taxonomy, keep_subcategories=False):
    # HAZUS taxonomy
    if '-' in taxonomy and '/' not in taxonomy:
        return taxonomy

    # GEM taxonomy
    # see also GEM taxonomy tool at https://platform.openquake.org/taxtweb/
    elif '/' in taxonomy:
        attributes = taxonomy.split('/')
        if len(attributes) > 0:
            if 'DX' in attributes[0] and len(attributes) > 1:
                material_string = attributes[1]
            else:
                material_string = attributes[0]
            if not keep_subcategories:
                material_string = material_string.split('+')[0] if 'MIX' not in material_string else material_string
            if sum([m == material_string.split('+')[0] for m in GEM_LAT_LOAD_MAT_NAMES]) != 1:
                print(f"Warning: Unknown material {material_string} for taxonomy {taxonomy}.")
        else:
            print(f"Warning: no attributes found in taxonomy {taxonomy}.")
            material_string = ''
        return material_string


def decode_taxonomy(taxonomy, keep_unknown=False, verbose=True, compute_vulnerability=False):
    res = pd.DataFrame({col: [[]] for col in ['lat_load_mat', 'lat_load_sys', 'height', 'unknown', 'vulnerability']},
                       index=[taxonomy])
    res.index.name = 'taxonomy'
    # # HAZUS taxonomy
    # if '-' in taxonomy and taxonomy[:3] in ['COM', 'RES', 'IND']:
    #     res.loc[taxonomy, 'hazus_id'] = [taxonomy.split('/')[0].split('-')[1]]
    # # GEM taxonomy
    # elif '/' in taxonomy:
    attribute_types = {attribute: identify_gem_attribute_type(attribute, verbose) for attribute in taxonomy.split('/')}
    for attribute, attribute_type in attribute_types.items():
        res.loc[[taxonomy], attribute_type] = (
                res.loc[taxonomy, attribute_type] +
                pd.DataFrame(index=[taxonomy], columns=attribute_type, data=[[[attribute]] * len(attribute_type)])
        )
    for col in res.columns:
        if len(res.loc[taxonomy, col]) == 0:
            res.loc[taxonomy, col] = np.nan
        elif len(res.loc[taxonomy, col]) == 1:
            res.loc[taxonomy, col] = res.loc[taxonomy, col][0]
        elif len(res.loc[taxonomy, col]) > 1:
            if res.loc[taxonomy, col][0] in ['MATO', 'UNK'] and 'UNK' in res.loc[taxonomy, col][0]:
                res.loc[taxonomy, col] = res.loc[taxonomy, col][0]
            elif verbose:
                print(f"Warning: Multiple attributes have been mapped to the same type for taxonomy {taxonomy}.")
    material, resistance_system, height = res.loc[taxonomy, ['lat_load_mat', 'lat_load_sys', 'height']]
    if compute_vulnerability:
        res.loc[taxonomy, 'vulnerability'] = assign_vulnerability(material, resistance_system, height,
                                                                  VULNERABILITY_MAPPING)
    else:
        res.drop('vulnerability', axis=1, inplace=True)
    if keep_unknown:
        return res
    else:
        return res.drop('unknown', axis=1)


def identify_gem_attribute_type(attribute, verbose=True):
    if len(attribute) == 0 and verbose:
        print("Warning: Empty attribute.")
    types = np.unique([FIELD_VALUE_TO_TYPE_MAP.get(field.split(':')[0], 'unknown') for field in attribute.split('+')])
    if len(types) == 1 and 'unknown' in types and verbose:
        print(f"Warning: Unknown type for attribute {attribute}.")
    elif len(types) == 2 and 'unknown' in types:
        types = types[types != 'unknown']
    elif len(types) > 1 and verbose:
        print(f"Warning: Multiple types {types} for attribute {attribute}.")
    return types



def compute_taxonomy_correspondences(gem_data):
    """
    This function computes the correspondence between macro taxonomy and derived material.

    Parameters:
    gem_data (DataFrame): The DataFrame containing the GEM data.

    Returns:
    DataFrame: A DataFrame containing the correspondence between macro taxonomy and derived material.
    """

    # Filter out the rows where 'iso3' is not in HAZUS_COUNTRIES and
    # keep only 'macro_taxonomy', 'taxonomy', 'n_buildings' columns
    matching = gem_data[~gem_data['iso3'].isin(HAZUS_COUNTRIES)][['macro_taxonomy', 'taxonomy', 'n_buildings']]

    # Apply the function 'get_material_from_taxonomy' to the 'taxonomy' column and store the result in a new column
    # 'derived_material'
    matching['derived_material'] = matching.taxonomy.apply(partial(get_material_from_taxonomy, keep_subcategories=True))

    # Create result DataFrame with unique 'macro_taxonomy' as index and unique 'derived_material' as columns
    res = pd.DataFrame(index=gem_data.macro_taxonomy.unique(), columns=matching.derived_material.unique())
    res.index.name = 'macro_taxonomy'
    res.columns.name = 'derived_material'

    # For each 'macro_taxonomy', compute the sum of 'n_buildings' grouped by 'derived_material' and normalize it
    for macro_taxo in res.index:
        n_buildings = matching[matching.macro_taxonomy == macro_taxo].groupby('derived_material').n_buildings.sum()
        n_buildings = n_buildings / n_buildings.sum()
        res.loc[macro_taxo, n_buildings.index] = n_buildings.values

    # shape the result DataFrame
    res = res.stack().reset_index().sort_values(by=['macro_taxonomy', 0], ascending=False).reset_index(drop=True)

    # Rename
    res.rename({0: 'share'}, axis=1, inplace=True)

    # Create a new column 'derived_material_l1' by splitting 'derived_material' on '+' and keeping the first level
    res['derived_material_l1'] = res.derived_material.apply(lambda s: s.split('+')[0] if '(' not in s else s)

    # Set 'macro_taxonomy', 'derived_material_l1', 'derived_material' as the index of the DataFrame
    res.set_index(['macro_taxonomy', 'derived_material_l1', 'derived_material'], inplace=True)

    return res


def get_taxonomy_mappings(outfile=None, pager_cat_mapping_path='./pager_vulnerability_mapping.csv',
                          pager_gem_mapping_path='./pager-gem_mapping.csv',
                          hazus_gem_mapping_path='./hazus-gem_mapping.csv'):
    # PAGER id to category mapping
    pager_category_mapping = pd.read_csv(pager_cat_mapping_path, index_col=0).astype(str)
    # note: PAGER id pairs of (C2, C2L, C2M, C2H) and (S4, S4L, S4M, S4H) are mapped to the same taxonomy string as
    # per the GEM taxonomy documentation appendix, table D-1; however, the pager-vulnerability mapping assigns all of
    # them, except C2L, to the same vulnerability 'robust'. Therefore, the vulnerability of C2L is likely wrong, and is
    # set to 'robust' as well.
    pager_category_mapping.loc['C2L'] = 'robust'

    # PAGER to GEM
    pager_gem_mapping = pd.read_csv(pager_gem_mapping_path, index_col=0).astype(str)
    pager_gem_mapping.rename({'gem_str': 'taxonomy'}, axis=1, inplace=True)

    # HAZUS to GEM
    hazus_gem_mapping = pd.read_csv(hazus_gem_mapping_path, index_col=0).astype(str)
    hazus_gem_mapping.rename({'gem_str': 'taxonomy'}, axis=1, inplace=True)

    decoded_gem_strings = pd.concat([decode_taxonomy(t, keep_unknown=False, verbose=False) for t in pager_gem_mapping.taxonomy])

    gem_to_vulnerability_mapping = pager_gem_mapping.reset_index().set_index('taxonomy').join(decoded_gem_strings)
    gem_to_vulnerability_mapping['building_category'] = gem_to_vulnerability_mapping.pager_id.apply(lambda x: pager_category_mapping.loc[x, 'aggregate_category'])
    gem_to_vulnerability_mapping.reset_index(inplace=True)
    gem_to_vulnerability_mapping.drop(['pager_id', 'taxonomy', 'pager_description'], axis=1, inplace=True)
    gem_to_vulnerability_mapping.drop_duplicates(inplace=True)
    gem_to_vulnerability_mapping.set_index(['lat_load_mat', 'lat_load_sys', 'height'], inplace=True)
    if gem_to_vulnerability_mapping.index.duplicated().any():
        duplicated_index = gem_to_vulnerability_mapping.index[gem_to_vulnerability_mapping.index.duplicated()]
        duplicated_index = duplicated_index.values[0]
        print("Warning: duplicated index in gem_to_vulnerability_mapping for index {}.".format(duplicated_index))
        print("Duplicated index yields vulnerability classes {}.".format(gem_to_vulnerability_mapping.loc[duplicated_index, 'building_category'].values))

    # add all possible heights
    for i in gem_to_vulnerability_mapping.index:
        if 'HBET:' in i[2]:
            h_max, h_min = [int(h) for h in i[2].split('HBET:')[1].split('+')[0].split(',')]
            building_cat = gem_to_vulnerability_mapping.loc[i, 'building_category']
            for h in range(h_min, h_max + 1):
                i_new = (i[0], i[1], i[2].replace(f"HBET:{h_max},{h_min}", f"H:{h}"))
                gem_to_vulnerability_mapping.loc[i_new] = building_cat

    # add variants with 99 IDs removed
    for i in gem_to_vulnerability_mapping.index:
        if np.any(['99' in a for a in i]):
            index_permutations = [[]] * len(i)
            for level_idx, level_value in enumerate(i):
                if '99' in i[level_idx]:
                    level_permutations = []
                    attributes = i[level_idx].split('+')
                    if len(attributes) > 1:
                        for attribute in attributes:
                            if '99' in attribute:
                                level_permutations = level_permutations + [[attribute, '']]
                            else:
                                level_permutations.append([attribute])
                        # level_permutations = [p if '' not in p else list(set(p) - {''}) for p in level_permutations]
                        level_permutations = list(product(*level_permutations))
                        level_permutations = ['+'.join(p) if len(p) > 1 else p for p in level_permutations]
                        level_permutations = [p.replace('++', '+') for p in level_permutations]
                        level_permutations = [p if len(p) > 0 else 'None' for p in level_permutations]
                        level_permutations = [p[1 if p[0] == '+' else 0:-1 if p[-1] == '+' else len(p)] for p in level_permutations]
                    else:
                        level_permutations = [attributes[0], 'None']
                    index_permutations[level_idx] = level_permutations
                else:
                    index_permutations[level_idx] = [level_value]
            for index_permutation in product(*index_permutations):
                if index_permutation not in gem_to_vulnerability_mapping.index:
                    gem_to_vulnerability_mapping.loc[index_permutation] = gem_to_vulnerability_mapping.loc[i]
    if outfile is None:
        return gem_to_vulnerability_mapping
    gem_to_vulnerability_mapping.to_csv(outfile)


def get_vulnerability_for_taxonomy(material, system, height, gem_to_vulnerability_mapping):
    """
    This function returns the vulnerability class for a given material, system and height.

    Parameters:
    material (str): The material.
    system (str): The system.
    height (str): The height.
    gem_to_vulnerability_mapping (DataFrame): The DataFrame containing the mapping between GEM and PAGER.

    Returns:
    str: The vulnerability class.
    """
    if (material, system, height) in gem_to_vulnerability_mapping.index:
        return gem_to_vulnerability_mapping.loc[(material, system, height), 'building_category']
    else:
        print("Warning: no direct vulnerability mapping found for material {}, system {} and height {}. Looking for "
              "closest vulnerability".format(material, system, height))
        value_counts = gem_to_vulnerability_mapping['building_category'].value_counts()
        if material in gem_to_vulnerability_mapping:
            value_counts = gem_to_vulnerability_mapping.loc[material, 'building_category'].value_counts()
            if len(value_counts) > 1:
                if system in gem_to_vulnerability_mapping.loc[material]:
                    value_counts = gem_to_vulnerability_mapping.loc[(material, system), 'building_category'].value_counts()
                    if len(value_counts) > 1:
                        print("Warning: could not approximate vulnerability for material {}, system {} and height {}."
                              "Using most frequent vulnerability class.".format(material, system, height))
        return value_counts[value_counts == value_counts.max()].index.item()


# TODO
def plot_average_vulnerability(gem_data, gadm_path, v_fragile=.7, v_median=.3, v_robust=.1):
    vulnerability = gem_data.groupby(['iso3', 'vulnerability']).n_buildings.sum() / gem_data.groupby('iso3').n_buildings.sum()
    vulnerability = vulnerability.unstack()
    vulnerability.fillna(0, inplace=True)
    vulnerability['fragile'] = vulnerability['fragile'] * v_fragile
    vulnerability['median'] = vulnerability['median'] * v_median
    vulnerability['robust'] = vulnerability['robust'] * v_robust
    vulnerability = vulnerability.sum(axis=1)

    # Load GADM shapefile data
    world = gpd.read_file(gadm_path)

    # Merge GADM data with gem_data
    merged = world.set_index('iso3').join(vulnerability)

    # Plot the merged DataFrame
    fig, ax = plt.subplots(1, 1)
    merged.plot(column='vulnerability', ax=ax, legend=True, cmap='coolwarm')
    plt.show()
