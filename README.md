## Code and data for the global Unbreakable model

This repository provides the model and processing code used for R. Middelanis et al. "Global Socio-economic Resilience to Natural Disasters."


To reproduce the manuscript results, please follow these steps:
1. Install the required dependencies using the provided environment.yml file. 
   * The easiest way to do this is to create a new conda environment with the command:
   ```bash
   conda env create -f environment.yml
   ```
   * Activate the environment with:
   ```bash
   conda activate <environment_name>
   ```
2. run the model by executing the script 
   ```bash
   python model/run_model.py <path_to_settings_file>
   ```
   * A settings.yml file for each simulation is located in the respective subdirectories of the "01_simulation_data" folder. 
   * The "force_recompute" flag in the settings files should be set to "False" to use pre-processed input data from 
   "scenario/data/processed". This will exactly reproduce the simulation outputs used in the manuscript. Setting the flag 
   to "True" will recompute all input data from the raw data files in the 
   "scenario/data/raw" directory, as well as from dynamically downloaded data sets, which may have been updated.
   * The script will generate the simulation output data in the output directory specified in the settings file.
   * The simulation outputs used for the manuscript are provided in the "01_simulation_data" folder.
3. Generate figures using the script 
   ```bash
   python misc/plotting.py <path_to_simulation_output_directory> <path_to_figure_output_directory> <path_to_GADM_data>
   ```
   * The script will generate all figures in the specified output directory.
   * The GADM global map shapes data must be provided in the GeoPackage format.
   * Note figures in the manuscript containing maps were further adjusted by the World Bank's cartography division.