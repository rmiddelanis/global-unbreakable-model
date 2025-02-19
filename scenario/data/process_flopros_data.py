import os

import geopandas as gpd
import numpy as np
import pandas as pd
import rioxarray as rxr
import tqdm
import xarray as xr


def process_flopros_data(flopros_path, flopros_update_path, flopros_update_shapes_path, population_path, gadm_path, outpath=None):#, wb_shapes_path, chn_shape_path, twn_shape_path):
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
    # Load flopros data
    flopros = gpd.read_file(flopros_path)
    flopros = flopros[flopros.OBJECTID != 0]

    # Original data has no merged layer due to lack of the modeled layer. Merge available design and policy layers
    # need to replace 0 with nan first, otherwise inland regions would decrease a country's ntl. coastal protection
    flopros[['DL_Max_Co', 'DL_Min_Co', 'PL_Max_Co', 'PL_Min_Co']] = (
        flopros[['DL_Max_Co', 'DL_Min_Co', 'PL_Max_Co', 'PL_Min_Co']].replace({0: np.nan}))
    # flopros['DL_PL_Co'] = flopros.DL_Max_Co.fillna(flopros.DL_Min_Co).fillna(flopros.PL_Max_Co).fillna(flopros.PL_Min_Co)
    flopros['DL_PL_Co'] = flopros[['DL_Max_Co', 'DL_Min_Co']].mean(axis=1).fillna(flopros[['PL_Max_Co', 'PL_Min_Co']].mean(axis=1))

    # Load flopros modeled layer data for coastal protection
    flopros_mod_co = pd.read_excel(flopros_update_path, index_col=0)
    flopros_update_shapes = gpd.read_file(flopros_update_shapes_path)
    flopros_mod_co = pd.merge(flopros_update_shapes[['FID_Aque', 'geometry']], flopros_mod_co, on='FID_Aque')
    flopros_mod_co = flopros_mod_co[['Protection standards', 'geometry']]
    flopros_mod_co.rename({'Protection standards': 'ModL_Co'}, axis=1, inplace=True)
    flopros_mod_co.dropna(inplace=True)

    # Load population data for the year 2020 (raster = 5)
    pop = rxr.open_rasterio(population_path, masked=True).sel(raster=5)
    pop.rio.set_crs(pop.attrs['proj4'], inplace=True)

    world_shapes = gpd.read_file(gadm_path, layer='ADM_0').rename(columns={'GID_0': 'iso3', 'COUNTRY': 'country'})
    world_shapes = world_shapes.set_index(['iso3', 'country']).geometry

    prot_ntl_aggregated = gpd.GeoDataFrame(index=world_shapes.index, columns=['MerL_Riv', 'MerL_Co'])

    protection_gridded = xr.Dataset(
        data_vars={'MerL_Riv': (('x', 'y'), np.full((len(pop.x), len(pop.y)), np.nan)),
                   'MerL_Co': (('x', 'y'), np.full((len(pop.x), len(pop.y)), np.nan))},
        coords={d: pop.coords[d].values for d in pop.dims})
    protection_gridded.rio.set_crs(pop.attrs['proj4'], inplace=True)

    # Assign modeled coastal protection levels to grid
    for i, row in tqdm.tqdm(flopros_mod_co.iterrows(), desc='assigning modeled coastal flood protection levels to grid',
                            total=len(flopros_mod_co)):
        mask = pop.rio.clip([row.geometry], flopros_mod_co.crs, drop=False).isnull()
        protection_gridded['MerL_Co'] = protection_gridded['MerL_Co'].where(mask, row['ModL_Co'])

    # Assign protection levels to grid
    for i, row in tqdm.tqdm(flopros.iterrows(), desc='assigning flopros protection levels to grid',
                            total=len(flopros)):
        mask = pop.rio.clip([row.geometry], flopros.crs, drop=False).isnull()
        protection_gridded['MerL_Riv'] = protection_gridded['MerL_Riv'].where(mask, row['MerL_Riv'])
        if not np.isnan(row['DL_PL_Co']):
            protection_gridded['MerL_Co'] = protection_gridded['MerL_Co'].where(mask, row['DL_PL_Co'])

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
        # prot_ntl_aggregated.to_file(outpath + ".shp", driver='ESRI Shapefile')
        prot_ntl_aggregated.drop('geometry', axis=1).to_csv(os.path.join(outpath, "flopros_protection_processed.csv"))

    return prot_ntl_aggregated


if __name__ == '__main__':
    result = process_flopros_data(
        flopros_path="./data/raw/FLOPROS/Scussolini_et_al_FLOPROS_shp_V1/",
        flopros_update_path="./data/raw/FLOPROS/Tiggeloven_et_al_2020_data/Results_adaptation_objectives/Protection_constant_Adaptation_objective_RCP4P5_SSP1.xlsx",
        flopros_update_shapes_path="./data/raw/FLOPROS/Tiggeloven_et_al_2020_data/Results_adaptation_objectives/Countries_States_simplified.shp",
        population_path="/Users/robin/data/NASA_SEDAC/GPW_gridded_population_of_the_world/gpw-v4-population-density-adjusted-to-2015-unwpp-country-totals-rev11_totpop_2pt5_min_nc/gpw_v4_population_density_adjusted_rev11_2pt5_min.nc",
        gadm_path='/Users/robin/data/GADM/gadm_410-levels.gpkg',
        outpath="./data/processed/"
    )

