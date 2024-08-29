import numpy as np
import pooch
import h5py

BRIAN = pooch.create(
    # sensible default for the cache
    path = pooch.os_cache('gemmm-model-data'),
    base_url='https://github.com/jontycarruthersphe/od_test/raw/main/model_data/', # needs the raw, CHANGE TO CORRECT URL
    # The registry specifies the files that can be fetched
    registry = {
        'fourier_data_weekday.hdf5', # need the hash for each of these
        'fourier_data_weekend.hdf5',
        'radiation_data_weekday.hdf5'
        'radiation_data_weekend.hdf5'
        }
    )


def find_indices(x1, x2):
        ''' 
        For each entry in x2, find the index of it in x1.
        Does not require x1 to be sorted.
        '''
        sorted_indices = np.argsort(x1)
        x1_sorted = x1[sorted_indices]
        idx_sorted = np.searchsorted(x1_sorted, x2)
        return sorted_indices[idx_sorted]


def fetch_model_data(day_type, msoas):
    '''
    Load the model data for a given day type.
    day_type (str): either "weekday" or "weekend"
    msoas (list): the list of MSOAs to generate numbers of journeys between
    '''
    # The file will be downloaded automatically the first time this is run
    # returns the file path to the downloaded file. Afterwards, Pooch finds
    # it in the local cache and doesn't repeat the download.
    try:
        fourier_file = BRIAN.fetch(f'fourier_data_{day_type}.hdf5')
        radiation_file = BRIAN.fetch(f'radiation_data_{day_type}.hdf5')
    except:
        # should only be for development
        fourier_file = f'C:/Users/Jonathan.Carruthers/Documents/telecoms/Gemmm/model_data/fourier_data_{day_type}.hdf5'
        radiation_file = f'C:/Users/Jonathan.Carruthers/Documents/telecoms/Gemmm/model_data/radiation_data_{day_type}.hdf5'

    
    # The "fetch" method returns the full path to the downloaded data file.
    # All we need to do now is load it with our standard Python tools.
    with h5py.File(fourier_file, 'r') as fourier_loader:
        # get the MSOA order used to create the fourier series/migration files.
        # this is needed to get the correct entries from the sparse matrices
        msoa_order = fourier_loader['msoa_order'][:].astype('U')
        if len(set(msoas) - set(msoa_order)) > 0:
            raise ValueError('MSOAs have been provided that are not included in the telecoms model')
        
        msoa_indices = find_indices(msoa_order, msoas)
        assert all(msoa_order[msoa_indices] == msoas)
        
        # get the hourly means for the fourier series model
        row = fourier_loader['row_idx'][:]
        col = fourier_loader['col_idx'][:]
        idx_mask = np.isin(row, msoa_indices) & np.isin(col, msoa_indices)
        new_row = find_indices(msoa_indices, row[idx_mask])        
        new_col = find_indices(msoa_indices, col[idx_mask])
        fourier_mean = fourier_loader['fourier'][:, idx_mask]
        
        # load the overdispersion parameters used to sample from the negative binomial distribution
        fourier_overdispersion = fourier_loader['overdispersion'][:]
        
    with h5py.File(radiation_file, 'r') as radiation_loader:
        # get the radiation model parameters
        radiation_mean = radiation_loader['radiation'][:,:][np.ix_(msoa_indices, msoa_indices)]
        radiation_theta = radiation_loader['theta'][:]
        
    return fourier_mean, new_row, new_col, fourier_overdispersion, radiation_mean, radiation_theta
        
        

'''
Model data cache locations are:
    Windows: C:\\Users\\<user>\\AppData\\Local\\<AppAuthor>\\gemmm-model-data\\Cache
'''
