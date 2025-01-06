import os
from functools import partial

import pandas as pd
 
# from sorted_nicely import sorted_nicely

 
def get_list_of_index_names(df):
    """returns name of index in a data frame as a list. (single element list if the dataframe as a single index)""
    """
    
    if df.index.name is None:
        return list(df.index.names)
    else:
        return [df.index.name] #do not use list( ) as list breaks strings into list of chars

 
def concat_categories(p,np, index):
    """works like pd.concat with keys but swaps the index so that the new index is innermost instead of outermost
    http://pandas.pydata.org/pandas-docs/stable/merging.html#concatenating-objects
    """
    
    if index.name is None:
        raise Exception("index should be named")
        
    
    y= pd.concat([p, np], 
        keys = index, 
        names=[index.name]+get_list_of_index_names(p)
            )#.sort_index()
    
    #puts new index at the end            
    y=y.reset_index(index.name).set_index(index.name, append=True).sort_index()
    
    #makes sure a series is returned when possible
    return y.squeeze()
#
#
# # only for updating purposes; load data from inputs or __legacy_structure/inputs, depending on whether it has
# # already been updated
# def load_input_data(root_dir, filename, **kwargs):
#     if filename.endswith('.csv') or filename.endswith('.csv.zip'):
#         load_func = partial(pd.read_csv, **kwargs)
#     elif filename.endswith('.xlsx'):
#         load_func = partial(pd.read_excel, **kwargs)
#     elif filename.endswith('.txt'):
#         def load_txt(x):
#             return open(x, 'r').read().strip().split('\n')
#         load_func = load_txt
#     else:
#         raise ValueError('Unknown file type for file {}'.format(filename))
#     if os.path.exists(os.path.join(root_dir, 'inputs', filename)):
#         return load_func(os.path.join(root_dir, 'inputs', filename))
#     raise ValueError('File {} not found.'.format(filename))
