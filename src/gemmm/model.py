'''
Generate sampled flows between pairs of MSOAs or load samples from an existing file.
'''
import pathlib
import ast
from datetime import datetime
import netCDF4
import pandas as pd
import numpy as np
import scipy.sparse as sp
from dask import delayed, compute

from .fetch_data import FetchData


class OriginDestination:
    '''
    Class for either generating or loading samples depending on the arguments provided.
    Sampling is performed if msoas and day_type are provided, otherwise a file containing
    existing samples must be provided.
    '''
    def __new__(cls, msoas=None, day_type=None, file=None):
        if file is not None:
            return super().__new__(ODLoader)

        if (msoas is not None) and (day_type is not None):
            return super().__new__(ODSampler)

        raise ValueError(('Either provide a path to existing samples or a list of MSOAs '
                          'and day_type to generate new samples.'))



class ODSampler(OriginDestination):
    '''
    Class for generating flows between pairs of MSOAs.

    Parameters
    ----------
    msoas: list or np.ndarray
        MSOAs to generate flows between.
    day_type: string
        The day type samples are being generated for, either 'weekday' or 'weekend'.
        
    Attributes
    ----------
    msoas : np.ndarray
        MSOAs to generate flows between.
    day_type : string
        Either 'weekday' or 'weekend'.
    fourier_data : FourierData
        Hourly means and overdispersion parameters for the Fourier series model.
    radiation_data : RadiationData
        Means and hourly scale factors for the radiation model.
    samples : None or dict
        A dictionary containing the generated samples. The keys are tuples of (hour, realization)
        and the values are scipy.sparse COO matrices containing the sampled values. None, until
        the generate_sample method has been called.
    '''

    def __init__(self, msoas, day_type):
        '''
        Parameters
        ----------
        msoas : list or np.ndarray
            MSOAs to generate flows between
        day_type : string
            Either "weekday" or "weekend"

        Raises
        ------
        TypeError
            If day_type is not a string, or the MSOAs are not provided in a list or np.ndarray.
        ValueError
            If day_type is not one of the accepted values, or only a single MSOA is provided.

        Returns
        -------
        None.
        '''

        if not isinstance(day_type, str):
            raise TypeError('day_type must be a str')

        if not day_type in ('weekday', 'weekend'):
            raise ValueError("day_type should be either 'weekday' or 'weekend'")

        self.day_type = day_type

        if not isinstance(msoas, list) | isinstance(msoas, np.ndarray):
            raise TypeError('MSOAs should be provided in a list or numpy array')

        if len(msoas) <= 1:
            raise ValueError('Please provide multiple MSOAs')
        msoas = np.array(msoas, dtype='U')

        self.msoas = msoas

        # Load the data required to run the Fourier series and radiation models
        fetcher = FetchData(msoas, day_type)
        self.fourier_data = fetcher.fetch_fourier()
        self.radiation_data = fetcher.fetch_radiation()

        self.samples = None


    @staticmethod
    def check_sample_inputs(hours, n_realizations):
        '''
        Check that the arguments of the sampling methods are appropriate.

        Parameters
        ----------
        hours : int or list of int
            The hours for which to obtain samples.
        n_realizations : int, > 0
            The number of realizations to generate for each hour.

        Raises
        ------
        TypeError
            If the hours or number of realizations are not integers.
        ValueError
            If the hours are not between 0 and 23, or the number of realizations is not positive.

        Returns
        -------
        hours : list of int
            The hours in a list.
        n_realizations : int
            The number of realizations.
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
    def sparse_poisson_sample(unscaled_mean, theta):
        '''
        Sample from a Poisson distribution with mean given by theta * unscaled_mean.

        Parameters
        ----------
        unscaled_mean : np.ndarray
            The unscaled means of the Poisson distribution.
        theta : numeric
            Scale factor.

        Returns
        -------
        sparse_samples : scipy.sparse matrix in COOrdinate format.
            The Poisson sampled values.
        '''

        sample = np.random.poisson(theta*unscaled_mean).astype(np.int16)
        sample_sparse = sp.coo_matrix(sample)
        return sample_sparse


    def radiation_sample(self, hours, n_realizations=1, client=None, check_inputs=True):
        '''
        Generate realizations using the radiation model.

        Parameters
        ----------
        hours : int or list of int
            The hours for which to generate samples.
        n_realizations : int, optional
            The number of realizations to generate for each hour. Default is 1.
        client : dask.distributed Client, optional
            Used to perform the radiation model sampling in parallel.
        check_inputs : boolean, optional
            Whether to check the hours and n_realizations arguments before performing the sampling.
            Default is True.

        Returns
        -------
        sample : list
            A list of scipy.sparse COO matrices containing the sampled values from the radiation
            model. The first n_realization matrices correspond to the first hour of hours,
            the next n_realization matrices correspond to the second hour etc.
        '''

        if check_inputs:
            hours, n_realizations = self.check_sample_inputs(hours, n_realizations)

        if client is None:
            # serial
            sample = [self.sparse_poisson_sample(self.radiation_data.mean, theta)
                      for theta in self.radiation_data.theta[hours]
                      for _ in range(n_realizations)]
        else:
            # parallel
            #delayed_mean = delayed(self.radiation_data.mean)
            scattered_mean = client.scatter(self.radiation_data.mean, broadcast=True)
            delayed_sample = [delayed(self.sparse_poisson_sample)(scattered_mean, theta)
                              for theta in self.radiation_data.theta[hours]
                              for _ in range(n_realizations)]
            sample = compute(*delayed_sample, client=client)
        return sample


    def sparse_nbinom_sample(self, hour):
        '''
        Sample from a negative-binomial distribution using the Fourier series mean (mean) and
        overdispersion parameter (k) for a given hour. The variance of the negative-binomial
        distribution is equal to mean + k*mean^2.

        Parameters
        ----------
        hour : int
            The hour for which to generate samples.

        Returns
        -------
        sparse_samples : scipy.sparse matrix in COOrdinate format.
            The negative-binomial sampled values.
        '''

        mean = self.fourier_data.mean[hour]
        k = self.fourier_data.overdispersion[hour]
        nb_n = 1 / k
        nb_p = 1 / (1 + k*mean)
        sample = np.random.negative_binomial(n=nb_n, p=nb_p).astype(np.int16)
        sample_sparse = sp.coo_matrix((sample, (self.fourier_data.row, self.fourier_data.col)),
                                      shape=(len(self.msoas), len(self.msoas)))
        return sample_sparse


    def fourier_sample(self, hours, n_realizations=1, check_inputs=True):
        '''
        Generate realizations using the Fourier series model.

        Parameters
        ----------
        hours : int or list of int
            The hours for which to generate samples.
        n_realizations : int, optional
            The number of realizations to generate for each hour. Default is 1.
        check_inputs : boolean, optional
            Whether to check the hours and n_realizations arguments before performing the sampling.
            Default is True.

        Returns
        -------
        sample : list
            A list of scipy.sparse COO matrices containing the sampled values from the Fourier
            series model. The first n_realization matrices correspond to the first hour of hours,
            the next n_realization matrices correspond to the second hour etc.
        '''

        if check_inputs:
            hours, n_realizations = self.check_sample_inputs(hours, n_realizations)

        sample = [self.sparse_nbinom_sample(hour) for hour in hours for _ in range(n_realizations)]
        return sample


    def generate_sample(self, hours, n_realizations=1, client=None, save_sample=False):
        '''
        Generate realizations using the combined Fourier series and radiation model.

        Parameters
        ----------
        hours : int or list of int
            The hours for which to generate samples.
        n_realizations : int, optional
            The number of realizations to generate for each hour. Default is 1.
        client : dask.distributed Client, optional
            Used to perform the radiation model sampling in parallel.
        save_sample : boolean or string/pathlib.Path, optional
            If False, samples are not saved.
            If True, samples are saved in the current working directory.
            If a string/pathlib.Path is provided, samples are saved to that directory.

        Raises
        ------
        TypeError
            If save_sample is not a boolean or a suitable path.
        FileNotFoundError
            If the directory specified by save_sample does not exist.

        Returns
        -------
        save_path : pathlib.Path
            Returns the file path if save_sample is True, otherwise returns None.
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
                    raise TypeError(('the path to the save directory should either '
                                     'be a str or pathlib.Path'))

                # check the directory exists
                if not save_directory.is_dir():
                    raise FileNotFoundError(f'No such directory: {save_directory}')

            save_file = save_directory / filename
            print(f'Saving samples to {save_file}')

        hours, n_realizations = self.check_sample_inputs(hours, n_realizations)
        radiation_sample = self.radiation_sample(hours, n_realizations, client=client,
                                                 check_inputs=False)
        fourier_sample = self.fourier_sample(hours, n_realizations, check_inputs=False)
        assert len(fourier_sample) == len(radiation_sample)
        sample = [(radiation_sample[ii] + fourier_sample[ii]).tocoo()
                  for ii in range(len(radiation_sample))]

        # convert to dict with (hour, realization) key for each sample
        # makes it easier to retrieve the sample for a specific hour, realization
        keys = [(hour, realization) for hour in hours for realization in range(n_realizations)]

        # setting the sample as an attribute make it easier to ensure that to_pandas is using
        # a sample that comes from the object that generated it.
        self.samples = dict(zip(keys, sample))

        if save_sample:
            self._save_netcdf(save_file, sample, hours, n_realizations, self.msoas)
            return save_file

        return None


    def to_pandas(self, hour, realization, wide=False):
        '''
        Convert a sample from a sparse matrix to a pandas DataFrame.

        Parameters
        ----------
        hour : int
            The hour of the sample to convert.
        realization : int
            The realization of the sample to convert.
        wide : boolean, optional
            Whether to return the pandas DataFrame in wide format. Default is False

        Raises
        ------
        TypeError
            If the hour or realization are not integers.
        KeyError
            If (hour, realization) is not a key in the dictionary of samples.

        Returns
        -------
        samples : pd.DataFrame
            If wide is False, the dataframe contains 3 columns for the start MSOA, end MSOA and number
            of journeys. If wide is True, the same dataframe is returned, but in wide format.
        '''

        if not isinstance(hour, int):
            raise TypeError('The hour must be an integer')
        if not isinstance(realization, int):
            raise TypeError('The realization must be an integer')

        key_hours, key_realizations = zip(*self.samples.keys())
        sample = self.samples.get((hour, realization))

        if sample is None:
            raise KeyError(('Key does not exist. '
                            f'The hour should be in {[int(i) for i in np.unique(key_hours)]} and '
                            f'the realziation should be between 0 and {max(key_realizations)}.'
                            ))

        sample_df = pd.DataFrame({'start_msoa': self.msoas[sample.row],
                                  'end_msoa': self.msoas[sample.col],
                                  'journeys': sample.data})
        if wide:
            # convert to wide format using pivot_table
            return pd.pivot_table(sample_df, columns='end_msoa', index='start_msoa', fill_value=0)

        return sample_df


    @staticmethod
    def _save_netcdf(file, samples, hours, n_realizations, msoas):
        '''
        Save samples to a netCD4 file.

        Parameters
        ----------
        file : pathlib.Path
            The file in which to save the samples.
        samples : list
            Contains scipy.sparse coo matrices of the sampled values
        hours : list
            Contains the hours for which the samples were obtained
        n_realizations : int
            The number of samples obtained for each hour
        msoas : np.ndarray
            Contains the MSOA codes used to obtain the samples

        File structure
        --------------
        Variables
            data (total_nonzero, 3) : A single array containing the stacked row indices, column
                                      indices and non-zero values from the scipy.sparse coo matrix
                                      representation of each sample
            index_lookup (n_hours, n_realizations, 2) : A lookup table to find the rows of 'data'
                                                        that correspond to a specific hour and
                                                        realization. The entry at index [i, j, :]
                                                        contains the start and end row for the ith
                                                        hour and jth realization.
            msoas (n_msoas, 9) : An array containing the codes of the MSOAs for which the sample
                                 was generated. Each MSOA code contains 9 characters
        Metadata
            n_realizations : The number of realizations that were obtained for each hour

            hour_to_index_mapping : A dictionary that provides a mapping between the hour value and
                                    its index in 'hours'. E.g. if hours = [0, 6, 12], the dictionary
                                    is {'hr0':0, 'hr6':1, 'hr12:2'}

        Returns
        -------
        None.
        '''

        end_index = np.cumsum([len(s.data) for s in samples])
        start_index = np.insert(end_index[:-1], 0, 0)
        total_nonzero = end_index[-1]
        indices = np.array(list(zip(start_index, end_index)))
        indices = indices.reshape((len(hours), n_realizations, 2))

        journeys = np.concatenate([s.data for s in samples]).reshape(-1,1)
        row = np.concatenate([s.row.astype(np.int16) for s in samples]).reshape(-1,1)
        col = np.concatenate([s.col.astype(np.int16) for s in samples]).reshape(-1,1)
        data = np.concatenate((row, col, journeys), axis=1)

        with netCDF4.Dataset(file, 'w', format='NETCDF4') as write_file:
            # specify the mapping between the hour and the row index of index_lookup
            hour_to_index = {f'hr{hour}': hours.index(hour) for hour in hours}
            write_file.hour_to_index_mapping = str(hour_to_index)

            # add the number of realizations to the metadata
            write_file.n_realizations = n_realizations

            # turn off filling
            write_file.set_fill_off()

            # set the dimensions of variables
            dim_dict = {'n_hours': len(hours),
                        'n_realizations': n_realizations,
                        'n_nonzero': total_nonzero,
                        'row_col_data': 3,
                        'start_end': 2,
                        'n_msoas': len(msoas),
                        'n_chars': 9}

            for dim_name, dim_size in dim_dict.items():
                write_file.createDimension(dim_name, dim_size)

            lookup_dtype = np.uint32
            if total_nonzero > np.iinfo(np.uint32).max:
                lookup_dtype = np.uint64
            index_lookup = write_file.createVariable('index_lookup', lookup_dtype,
                                                     ('n_hours', 'n_realizations', 'start_end'),
                                                     compression='zlib', complevel=9, shuffle=True)
            index_lookup[...] = indices

            # define a variable for the data
            sample_data = write_file.createVariable('data', np.int16, ('n_nonzero', 'row_col_data'),
                                                    compression='zlib', complevel=9, shuffle=True)
            sample_data[...] = data

            # define a variable for the msoas
            sample_msoas = write_file.createVariable('msoas', 'S1', ('n_msoas', 'n_chars'))
            # accessing protected member enables automatic conversion of strings
            sample_msoas._Encoding = 'ascii'
            sample_msoas[...] = msoas.astype('S')



class ODLoader(OriginDestination):
    '''
    Class for loading existing samples from a netCDF4 file.

    Parameters
    ----------
    file : string or pathlib.Path
        The full file path of the netCDF4 file containing the existing samples.

    Attributes
    ----------
    file : pathlib.Path
        The full file path of the netCDF4 file containing the existing samples.
    hour_to_index : dict
        A mapping between hour and the index of the start and end indices
        required to extract the correct journey numbers.
    hours : list
        The hours for which samples are available.
    n_realizations : int
        The number of realizations available for each hour.
    msoas : np.ndarray
        The MSOAs used to generate the samples
    '''

    def __init__(self, file):
        '''
        Parameters
        ----------
        file : string or pathlib.Path
            The netCDF4 file containing the existing samples.

        Raises
        ------
        TypeError
            If file is not a string of pathlib.Path
        FileNotFoundError
            If the file does not exist.
        ValueError
            If the file provided is not a netCDF file (does not have .nc extension)

        Returns
        -------
        None.
        '''

        if isinstance(file, str):
            file = pathlib.Path(file)
        elif not isinstance(file, pathlib.Path):
            raise TypeError('The file path should either be a str or pathlib.Path')

        # check that the file exists
        if not file.is_file():
            raise FileNotFoundError(f'No such file: {file}')

        if file.suffix != '.nc':
            raise ValueError('The file should be a netCDF file')

        self.file = file

        # get the available hours and number of realizations
        with netCDF4.Dataset(self.file, 'r', format='NETCDF4') as read_file:
            self.hour_to_index = ast.literal_eval(read_file.hour_to_index_mapping)
            available_hours = [key.split('hr')[-1] for key in self.hour_to_index.keys()]
            self.hours = [int(x) for x in available_hours]
            self.n_realizations = int(read_file.n_realizations)
            self.msoas = read_file['msoas'][:]

        print((f'Available hours: {(", ").join(available_hours)}'
              f'\nNumber of realizations: {self.n_realizations}'))


    def load_sample(self, hour, realization=None, as_pandas=True, wide=False):
        '''
        Loads a single sample for a specific hour.

        Parameters
        ----------
        hour : int
            The hour for which to load the sample, must be between 0 and 23.
        realization : int, optional
            The sample number, must be greater than or equal to 0. If None, a realization is selected at random.
        as_pandas : boolean, optional
            Whether to return the sample as a pandas DataFrame with the row/column indices replaced
            with the corresponding start/end MSOA codes. Default is True.
        wide : boolean, optional
            Whether to return the pandas DataFrame in wide format. Default is False.

        Raises
        ------
        TypeError
            If the hour or realization number is not an integer.
        ValueError
            If the hour or realization number is not available in the given file.

        Returns
        -------
        samples : pd.DataFrame or np.ndarray.
            If as_pandas is True, returns the default dataframe containing 3 columns for the start MSOA, end MSOA 
            and number of journeys. If wide is also True, returns the same dataframe in wide format. If as_pandas 
            is False, returns an array with 3 columns - the first 2 contain the indices of the start and end MSOAs 
            and the final column contains the number of journeys between them.
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
                raise ValueError(f'The realization must be between 0 and {self.n_realizations-1}')
        else:
            # randomly choose a realization
            realization = np.random.randint(low=0, high=self.n_realizations)

        with netCDF4.Dataset(self.file, 'r', format='NETCDF4') as read_file:
            hour_index = self.hour_to_index[f'hr{hour}']
            start_index, end_index = read_file['index_lookup'][hour_index, realization, :]
            sample = read_file['data'][start_index:end_index,:].data

        if as_pandas:
            # convert to a pandas dataframe with the row and column index replaced by the code of
            # the start and end MSOA
            sample_df = pd.DataFrame({'start_msoa': self.msoas[sample[:,0]],
                                      'end_msoa': self.msoas[sample[:,1]],
                                      'journeys': sample[:,2]})
            if wide:
                # convert to wide format using pivot_table
                return pd.pivot_table(sample_df, columns='end_msoa', index='start_msoa',
                                      fill_value=0)

            return sample_df

        return sample
