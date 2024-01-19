import geopandas as gpd
import numpy as np
import pandas as pd
import rioxarray as rxr
import tqdm
import xarray as xr


def process_flopros_data(flopros_path, flopros_update_path, flopros_update_shapes_path, population_path, wb_shapes_path,
                         chn_shape_path, twn_shape_path, outpath=None):
    """
        This function processes flood protection data (flopros) and aggregates it to the country level.

        Parameters:
        flopros_path (str): Path to the original flopros data shapefile. Can be downloaded from https://doi.org/10.5194/nhess-16-1049-2016
        flopros_update_path (str): Path to the updated flopros data file (Protection_constant_*.xlsx). Can be downloaded from https://doi.org/10.5281/zenodo.4275517
        flopros_update_shapes_path (str): Path to the shapefile for the updated flopros data.
        population_path (str): Path to the population data file.
        wb_shapes_path (str): Path to the World Bank shapefiles.
        chn_shape_path (str): Path to the China shapefile.
        twn_shape_path (str): Path to the Taiwan shapefile.
        outpath (str, optional): Path to store the output files. If None, the output files are not stored.

        Returns:
        None
        """
    # Read in flopros data
    flopros = gpd.read_file(flopros_path)
    flopros = flopros[flopros.OBJECTID != 0]

    # Original data has no merged layer due to lack of the modeled layer. Merge available design and policy layers
    # need to replace 0 with nan first, otherwise inland regions would decrease a country's ntl. coastal protection
    flopros[['DL_Max_Co', 'DL_Min_Co', 'PL_Max_Co', 'PL_Min_Co']] = (
        flopros[['DL_Max_Co', 'DL_Min_Co', 'PL_Max_Co', 'PL_Min_Co']].replace({0: np.nan}))
    flopros['DL_PL_Co'] = flopros.DL_Max_Co.fillna(flopros.DL_Min_Co).fillna(flopros.PL_Max_Co).fillna(flopros.PL_Min_Co)

    # Read in flopros modeled layer data for coastal protection
    flopros_mod_co = pd.read_excel(flopros_update_path, index_col=0)
    flopros_update_shapes = gpd.read_file(flopros_update_shapes_path, index_col=0)
    flopros_mod_co = pd.merge(flopros_update_shapes[['FID_Aque', 'geometry']], flopros_mod_co, on='FID_Aque')
    flopros_mod_co = flopros_mod_co[['Protection standards', 'geometry']]
    flopros_mod_co.rename({'Protection standards': 'ModL_Co'}, axis=1, inplace=True)
    flopros_mod_co.dropna(inplace=True)

    # Read in population data for the year 2020 (raster = 5)
    pop = rxr.open_rasterio(population_path, masked=True).sel(raster=5)
    pop.rio.set_crs(pop.attrs['proj4'], inplace=True)

    # Read in world bank shapefiles
    world_shapes = gpd.read_file(wb_shapes_path)[['ISO_A3', 'WB_NAME', 'geometry']]
    world_shapes.rename({'ISO_A3': 'iso3', 'WB_NAME': 'country'}, axis=1, inplace=True)
    world_shapes.loc[world_shapes.country == 'France', 'iso3'] = 'FRA'
    world_shapes.loc[world_shapes.country == 'Kosovo', 'iso3'] = 'XKX'
    world_shapes.loc[world_shapes.country == 'Norway', 'iso3'] = 'NOR'
    # drop overseas territories for now TODO: discuss
    world_shapes = world_shapes[~(world_shapes.iso3 == '-99') & ~(world_shapes.country.isin(
        ['Bonaire (Neth.)', 'Sint Eustatius (Neth.)', 'Saba (Neth.)', 'Johnston Atoll (US)', 'Jarvis Island (US)',
         'Baker Island (US)', 'Howland Island (US)', 'Wake Island (US)', 'Midway Islands (US)', 'Navassa Island (US)',
         'Palmyra Atoll (US)', 'Kingman Reef (US)', 'Tokelau (NZ)']))]
    world_shapes = world_shapes.set_index(['iso3', 'country']).squeeze()

    chn_shape = gpd.read_file(chn_shape_path)
    world_shapes.loc['CHN', 'China'] = chn_shape[~chn_shape.NAME_1.isin(['Hong Kong', 'Macau'])].geometry.unary_union
    world_shapes.loc['HKG', 'Hong Kong (SAR, China)'] = chn_shape[chn_shape.NAME_1 == 'Hong Kong'].geometry.item()
    world_shapes.loc['MAC', 'Macau (SAR, China)'] = chn_shape[chn_shape.NAME_1 == 'Macau'].geometry.item()

    world_shapes.loc['TWN', 'Taiwan'] = gpd.read_file(twn_shape_path).geometry.item()

    prot_ntl_aggregated = gpd.GeoDataFrame(index=world_shapes.index, columns=['MerL_Riv', 'MerL_Co'])

    protection_gridded = xr.Dataset(
        data_vars={'MerL_Riv': (('x', 'y'), np.full((len(pop.x), len(pop.y)), np.nan)),
                   'MerL_Co': (('x', 'y'), np.full((len(pop.x), len(pop.y)), np.nan))},
        coords={d: pop.coords[d].values for d in pop.dims})
    protection_gridded.rio.set_crs(pop.attrs['proj4'], inplace=True)

    # Assign protection levels to grid
    for i, row in tqdm.tqdm(flopros.iterrows(), desc='assigning flopros protection levels to grid',
                            total=len(flopros)):
        mask = pop.rio.clip([row.geometry], flopros.crs, drop=False).isnull()
        protection_gridded['MerL_Riv'] = protection_gridded['MerL_Riv'].where(mask, row['MerL_Riv'])
        if not np.isnan(row['DL_PL_Co']):
            protection_gridded['MerL_Co'] = protection_gridded['MerL_Co'].where(mask, row['DL_PL_Co'])

    # Assign modeled coastal protection levels to grid
    for i, row in tqdm.tqdm(flopros_mod_co.iterrows(), desc='assigning modeled coastal flood protection levels to grid',
                            total=len(flopros_mod_co)):
        mask = pop.rio.clip([row.geometry], flopros_mod_co.crs, drop=False).isnull()
        protection_gridded['MerL_Co'] = protection_gridded['MerL_Co'].where(mask, row['ModL_Co'])

    # Calculate population weighted protection levels
    protection_weighted = protection_gridded * pop

    # Aggregate gridded protection to country level
    for iso3, country in tqdm.tqdm(prot_ntl_aggregated.index, desc='aggregating to country level',
                                   total=len(prot_ntl_aggregated)):
        mask = ~pop.rio.clip([world_shapes.loc[(iso3, country)]], world_shapes.crs, drop=False).isnull()
        country_protection = protection_weighted.where(mask).sum() / pop.where(mask).sum()
        prot_ntl_aggregated.loc[(iso3, country), 'MerL_Riv'] = country_protection.MerL_Riv.item()
        prot_ntl_aggregated.loc[(iso3, country), 'MerL_Co'] = country_protection.MerL_Co.item()
        prot_ntl_aggregated.loc[(iso3, country), 'geometry'] = world_shapes.loc[(iso3, country)]

    if outpath is not None:
        if '.' in outpath:
            outpath = ''.join(outpath.split('.')[:-1])
        print(f'Storing protection levels to {outpath + ".shp"} and {outpath + ".csv"}.')
        prot_ntl_aggregated.to_file(outpath + ".shp", driver='ESRI Shapefile')
        prot_ntl_aggregated.drop('geometry', axis=1).to_csv(outpath + ".csv")