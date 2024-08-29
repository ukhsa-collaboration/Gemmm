import numpy as np
import scipy.sparse as sp
from dask import delayed, compute
import pathlib 
from datetime import datetime
import netCDF4
import ast

from .fetch_data import fetch_model_data


class OriginDestination:
    def __new__(cls, msoas=None, day_type=None, path=None):
        if path is not None:
            #print('Loading existing sample')
            return super().__new__(ODLoader)
            
        elif (msoas is not None) & (day_type is not None):
            #print('Generating samples')
            return super().__new__(ODSampler)
            
        else:
            raise ValueError('Either provide a path to existing samples or a list of MSOAs and day_type to generate new samples.')

            
            
class ODSampler(OriginDestination):
    ''' Generate sampled numbers of journeys between a given list of MSOAs '''
    
    def __init__(self, msoas, day_type):
        '''
        msoas (list): A list of MSOAs to generate numbers of journeys between
        day_type (str): either 'weekday' or 'weekend'
        '''
        if not isinstance(day_type, str):
            raise TypeError('day_type must be a str')
            
        if not day_type in ('weekday', 'weekend'):
            raise ValueError("day_type should be either 'weekday' or 'weekend'")
        
        self.day_type = day_type
        
        if not (isinstance(msoas, list) | isinstance(msoas, np.ndarray)):
            raise TypeError('MSOAs should be provided in a list or numpy array')

        if len(msoas) <= 1:
            raise ValueError('Please provide multiple MSOAs')
        msoas = np.array(msoas, dtype='U')
        self.msoas = msoas
        
        # fetch the model data
        (self.fourier_mean, 
        self.new_row, 
        self.new_col, 
        self.fourier_overdispersion, 
        self.radiation_mean, 
        self.radiation_theta) = fetch_model_data(self.day_type, self.msoas)
        
        
    @staticmethod
    def check_sample_inputs(hours, n_realizations):
        '''
        Check the inputs for the sampling methods are appropriate
        '''
        if not hasattr(hours, '__len__'):
            hours = [hours]
        if any(not isinstance(hour, int) for hour in hours):
            raise TypeError('The hours must all be integers')
        if (min(hours) < 0) |  (max(hours) > 23):
            raise ValueError('The hours must be between 0 and 23')
        if not isinstance(n_realizations, int):
            raise TypeError('The number of realizations must be an integer')
        if n_realizations <= 0:
            raise ValueError('The number of realizations must be greater than 0')
        return hours, n_realizations
    
    
    @staticmethod
    def sparse_poisson_sample(mean, theta):
        ''' 
        Sample from a poisson distribution with mean equal to mean*theta
        mean (array-like)
        theta (float)
        '''
        S = np.random.poisson(theta*mean).astype(np.int16)
        S_sparse = sp.coo_matrix(S)
        return S_sparse
        
    
    def radiation_sample(self, hours, n_realizations=1, client=None, check_inputs=True):
        '''
        Sample numbers of journeys for pairs modelled by a radiation model
        hours (int, list of int, 0-23): the hour(s) for which to obtain samples
        n_realizations (int): the number of samples to obtain for each hour
        client (): used for parallel sampling of the radiation model
        '''
        if check_inputs:
            hours, n_realizations = self.check_sample_inputs(hours, n_realizations)
            
        if client is None:
            # serial
            sample = [self.sparse_poisson_sample(self.radiation_mean, theta) for theta in self.radiation_theta[hours] for _ in range(n_realizations)]
        else:
            # parallel
            delayed_mean = delayed(self.radiation_mean)
            delayed_sample = [delayed(self.sparse_poisson_sample)(delayed_mean, theta) \
                              for theta in self.radiation_theta[hours] \
                              for _ in range(n_realizations)]
            sample = compute(*delayed_sample, client=client)
        return sample
    
    
    def sparse_nbinom_sample(self, hour):
        '''
        Sample from a negative-binomial distribution with variance equal to mean + k*mean^2
        hour (int, 0-23): the hour for which to obtain a sample
        '''
        mean = self.fourier_mean[hour]
        k = self.fourier_overdispersion[hour]
        n = 1 / k
        p = 1 / (1 + k*mean)
        S = np.random.negative_binomial(n=n, p=p).astype(np.int16)
        S_sparse = sp.coo_matrix((S, (self.new_row, self.new_col)), shape=(len(self.msoas), len(self.msoas)))
        return S_sparse
    
    
    def fourier_sample(self, hours, n_realizations=1, check_inputs=True):
        '''
        Sample numbers of journeys for pairs modelled by a Fourier series
        hours (int, list of int, 0-23): the hour(s) for which to obtain samples
        n_realizations (int): the number of samples to obtain for each hour
        '''
        if check_inputs: 
            hours, n_realizations = self.check_sample_inputs(hours, n_realizations)
        
        sample = [self.sparse_nbinom_sample(hour) for hour in hours for _ in range(n_realizations)]
        return sample
        

    def generate_sample(self, hours, n_realizations=1, client=None, save_sample=False): 
        '''
        Sample numbers of journeys between pairs of MSOAs
        hours (int, list of int, 0-23): the hour(s) for which to obtain samples
        n_realizations (int): the number of samples to obtain for each hour
        client (dask client): used for parallel sampling of the radiation model
        save_sample (bool, path): if False, samples are not saved
                                  if True, samples are saved in the current working directory
                                  if str/pathlib.Path, samples are saved in this directory
        '''
        # checks on the path for saving 
        if save_sample:
            timestamp = datetime.today().strftime('%Y-%m-%d--%H-%M-%S')
            filename = f'{self.day_type}_samples_{timestamp}.nc'
            save_directory = pathlib.Path('')
            
            if save_sample is not True:
                if isinstance(save_sample, str):
                    save_directory = pathlib.Path(save_sample)
                elif isinstance(save_sample, pathlib.Path):
                    save_directory = save_sample
                else:
                    raise TypeError('the path to the save directory should either be a str or pathlib.Path')
                    
                # check the directory exists
                if not save_directory.is_dir():
                    raise FileNotFoundError(f'No such directory: {save_directory}')
        
            save_file = save_directory / filename
            print(save_file)

        hours, n_realizations = self.check_sample_inputs(hours, n_realizations)    
        radiation_sample = self.radiation_sample(hours, n_realizations, client=client, check_inputs=False)
        fourier_sample = self.fourier_sample(hours, n_realizations, check_inputs=False)
        assert len(fourier_sample) == len(radiation_sample)
        sample = [(radiation_sample[ii] + fourier_sample[ii]).tocoo() for ii in range(len(radiation_sample))]
        
        # save the sample to a netcdf file with the following variables
        #   data (total_nonzero_flows x 3): for all samples, stack the index of the start/end MSOA
        #                                   and the number of journeys. Pairs with 0 journeys are omitted
        #   index_lookup (n_hours x n_realizations x 2): The entries at index [i,j,:] contain the start
        #                                                and end row of data, corresponding to the sample
        #                                                for the ith hour and jth realization.
        #
        # Since the hours provided may not match the indices, a dictionary is provided in the metadata
        # e.g. if hours = [0, 6, 12], the dictionary is {'hr0':0, 'hr6':1, 'hr12':3}
        if save_sample:
            end_index = np.cumsum([len(s.data) for s in sample])
            start_index = np.insert(end_index[:-1], 0, 0)
            indices = np.array(list(zip(start_index, end_index)))
            indices = indices.reshape((len(hours), n_realizations, 2))
            total_nonzero = end_index[-1]
            
            journeys = np.concatenate([s.data for s in sample]).reshape(-1,1)
            row = np.concatenate([s.row.astype(np.int16) for s in sample]).reshape(-1,1)
            col = np.concatenate([s.col.astype(np.int16) for s in sample]).reshape(-1,1)
            data = np.concatenate((row, col, journeys), axis=1)
            
            # todo: should also add the msoas as a separate variable
            
            with netCDF4.Dataset(save_file, 'w', format='NETCDF4') as f:
                # specify the mapping between the hour and the row index of index_lookup
                hour_to_index = {f'hr{hour}': hours.index(hour) for hour in hours}
                f.hour_to_index_mapping = str(hour_to_index)
                
                # add the number of realizations to the metadata
                f.n_realizations = n_realizations
                
                # turn of filling
                f.set_fill_off()
                
                # set the dimensions of variables
                dim_hour = f.createDimension('n_hours', len(hours))
                dim_rlz = f.createDimension('n_realizations', n_realizations)
                dim_nz = f.createDimension('n_nonzero', total_nonzero)
                dim_rcd = f.createDimension('row_col_data', 3)
                dim_se = f.createDimension('start_end', 2)
                dim_msoa = f.createDimension('n_msoas', len(self.msoas))
                dim_char = f.createDimension('n_chars', 9)  # no. of characters in each MSOA code
                
                # define a variable for the index lookup
                lookup_dtype = np.uint32
                if total_nonzero > np.iinfo(np.uint32).max:
                    lookup_dtype = np.uint64
                index_lookup = f.createVariable('index_lookup', lookup_dtype, ('n_hours', 'n_realizations', 'start_end'),
                                               compression='zlib', complevel=9, shuffle=True)            
                index_lookup[...] = indices            
                
                # define a variable for the data
                sample_data = f.createVariable('data', np.int16, ('n_nonzero', 'row_col_data'),
                                               compression='zlib', complevel=9, shuffle=True)
                sample_data[...] = data
                
                # define a variable for the msoas
                sample_msoas = f.createVariable('msoas', 'S1', ('n_msoas', 'n_chars'))
                sample_msoas._Encoding = 'ascii' # this enables automatic conversion
                sample_msoas[...] = self.msoas.astype('S')
            
        return sample
    
    
    
class ODLoader(OriginDestination):
    ''' Load sampled numbers of journeys from an existing file '''
    
    def __init__(self, file):
        '''
        file (str or pathlib.Path): netcdf file where the samples are saved
        '''
        if isinstance(file, str):
            file = pathlib.Path(file)
        elif not isinstance(file, pathlib.Path):
            raise TypeError('The path should either be a str or pathlib.Path')
        
        # check that the file exists
        if not file.is_file():
            raise FileNotFoundError(f'No such file: {file}')
        
        self.file = file
    
        # get the available hours and number of realizations
        with netCDF4.Dataset(self.file, 'r', format='NETCDF4') as f:
            self.hour_to_index = ast.literal_eval(f.hour_to_index_mapping)
            available_hours = [key.split('hr')[-1] for key in self.hour_to_index.keys()]
            self.hours = [int(x) for x in available_hours]
            self.n_realizations = int(f.n_realizations)
            self.msoas = f['msoas'][:]
        
        print(f'Available hours: {(", ").join(available_hours)}\nNumber of realizations: {self.n_realizations}')
        
        
    def load_sample(self, hour, realization=None):
        '''
        hour (int, 0-23): the hour to load samples for
        realization (int): the number of a specific realization to load, 
                           if None the realization is selected at random
        '''
        # checks on inputs
        if not isinstance(hour, int):
            raise TypeError('The hour must be an integer')
            
        if not hour in self.hours:
            raise ValueError(f'Choose from the available hours: {self.hours}')
            
        if realization is not None:
            # it should be an integer and less than n_realizations
            if not isinstance(realization, int):
                raise TypeError('The realization must be an integer')
            if not 0 <= realization < self.n_realizations:
                raise ValueError('The realization must be between 0 and {self.n_realizations-1}')
        else:
            # randomly choose a realization
            realization = np.random.randint(low=0, high=self.n_realizations)
            
        with netCDF4.Dataset(self.file, 'r', format='NETCDF4') as f:
            hour_index = self.hour_to_index[f'hr{hour}']
            start_index, end_index = f['index_lookup'][hour_index, realization, :]
            sample = f['data'][start_index:end_index,:].data
            
        return sample
            
