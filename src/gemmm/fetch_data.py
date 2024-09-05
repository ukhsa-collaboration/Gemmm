'''
Load the necessary data required to generate sample flows for a given set of MSOAs
'''

import numpy as np
import pooch
import h5py


class FetchData():
    '''
    Class for loading Fourier series and radiation model data required to
    generate sample flows between pairs of MSOAs.

    Attributes
    ----------
    fourier_file : string
        path to HDF5 file containing the Fourier series model data.
    radiation_fie : string
        path to HDF5 file containing the radiation model data.
    msoa_indices : np.ndarray
        Contains the indices of the specified MSOAs within the original array of
        MSOAs used to fit the model.
    new_row / new_col : np.ndarray
        For a matrix of means with rows and columns equal to msoas, contains
        the row / column indices of the non-zero values.
    fourier_mean : np.ndarray (24, n_nonzero)
        Contains the non-zero means for the Fourier series model.
        The ith row corresponds to the ith hour of the day.
    overdispersion : np.ndarray (24, )
        Contains the overdispersion parameters for each hour of the day that
        are needed to sample from the negative binomial distribution for pairs
        modelled using a Fourier series.
    radiation_mean : np.ndarray (n_msoas, n_msoas)
        Contains the means for the radiation model.
    theta : np.ndarray (24, )
        Contains scale factors for each hour of the day that multiply the
        radiation_mean when sampling from a Poisson distribution.
    '''

    def __init__(self):
        data_url = 'https://api.github.com/repos/ukhsa-collaboration/Gemmm/contents/model_data/'
        cache_dir = 'gemmm-model-data'
        self.github_token = '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' # valid until 12/9/2024

        self.goodboy = pooch.create(
            path = pooch.os_cache(cache_dir),
            base_url = data_url,

            # The registry specifies the files that can be fetched
            registry = {'fourier_data_weekday.hdf5':
                        'sha256:8537b3f3efdd38cd757198256f5143b4936a724e87724ea879f5d7ebb7ab6924',

                        'fourier_data_weekend.hdf5':
                        'sha256:2b30087bf55f454c0b4d9e0bf49335103f88c3ea2fdf06a791cf87a61d6fbc17',

                        'radiation_data_weekday.hdf5':
                        'sha256:3ea52ea1b0883a3334c5ee7b48b9cffbd68251722b753f7e2361fe9c49a9e0ba',

                        'radiation_data_weekend.hdf5':
                        'sha256:f61de789b2436609abd44035377f7fc8a509db68e54526a7ca33b4e962a78d43'
                        }
            )

        self.fourier_file = None
        self.radiation_file = None
        self.msoa_indices = None
        self.new_row = None
        self.new_col = None
        self.fourier_mean = None
        self.overdispersion = None
        self.radiation_mean = None
        self.theta = None


    @staticmethod
    def find_indices(x_1, x_2):
        '''
        Parameters
        ----------
        x_1 : np.ndarray
        x_2 : np.ndarray

        Returns
        -------
        np.ndarray
            An array containing the index in x_1 of each element in x_2.
            Does not require x_1 to be sorted.

        '''
        sorted_indices = np.argsort(x_1)
        x_1_sorted = x_1[sorted_indices]
        idx_sorted = np.searchsorted(x_1_sorted, x_2)
        return sorted_indices[idx_sorted]


    def fetch_model_data(self, msoas, day_type):
        '''
        Parameters
        ----------
        msoas : np.ndarray
            An array of MSOA codes that we require model data for
        day_type : string
                Either "weekday" or "weekend"

        Raises
        ------
        ValueError
            If msoas includes codes that were not including when initially
            fitting the model.

        Returns
        -------
        None.
        '''
        # The file will be downloaded automatically the first time this is run
        # returns the file path to the downloaded file. Afterwards, Pooch finds
        # it in the local cache and doesn't repeat the download.

        headers = {'Authorization': f'token {self.github_token}',
                   'Accept': 'application/vnd.github.v4.raw'
                   }

        downloader_auth = pooch.HTTPDownloader(headers=headers, progressbar=True)

        self.fourier_file = self.goodboy.fetch(f'fourier_data_{day_type}.hdf5',
                                               downloader=downloader_auth)

        self.radiation_file = self.goodboy.fetch(f'radiation_data_{day_type}.hdf5',
                                                 downloader=downloader_auth)

        '''
        # for development use only
        self.fourier_file = ('C:/Users/Jonathan.Carruthers/Documents/telecoms/Gemmm/model_data/'
                             f'fourier_data_{day_type}.hdf5')
        self.radiation_file = ('C:/Users/Jonathan.Carruthers/Documents/telecoms/Gemmm/model_data/'
                               f'radiation_data_{day_type}.hdf5')
        '''

        with h5py.File(self.fourier_file, 'r') as fourier_loader:
            # get the MSOA order used to create the fourier series/migration files.
            # this is needed to get the correct entries from the sparse matrices
            msoa_order = fourier_loader['msoa_order'][:]

        msoa_order = np.astype(msoa_order, 'U')

        if len(set(msoas) - set(msoa_order)) > 0:
            raise ValueError(('MSOAs have been provided that are '
                              'not included in the telecoms model'))

        self.msoa_indices = self.find_indices(msoa_order, msoas)
        assert all(msoa_order[self.msoa_indices] == msoas)

        self.fetch_fourier()
        self.fetch_radiation()


    def fetch_fourier(self):
        '''
        Load the data needed to run the Fourier series model.

        The mean is saved in sparse-style, with non-zero entries and their
        corresponding row/column indices. Only extract values relevant to the
        array of MSOA codes.

        Overdispersion parameters are needed to sample from a negative-binomial
        distribution. Each hour of the day has a different parameter, so the
        whole array is required.
        '''
        with h5py.File(self.fourier_file, 'r') as fourier_loader:
            # get the hourly means for the fourier series model
            row = fourier_loader['row_idx'][:]
            col = fourier_loader['col_idx'][:]
            idx_mask = np.isin(row, self.msoa_indices) & np.isin(col, self.msoa_indices)
            self.new_row = self.find_indices(self.msoa_indices, row[idx_mask])
            self.new_col = self.find_indices(self.msoa_indices, col[idx_mask])
            self.fourier_mean = fourier_loader['fourier'][:, idx_mask]

            # load the overdispersion parameters
            self.overdispersion = fourier_loader['overdispersion'][:]


    def fetch_radiation(self):
        '''
        Load the data needed to run the radiation model

        Only extract the values of the mean that are relevant to the array of
        MSOA codes.

        The parameters theta scale the mean depending on the hour of the day,
        so the whole array is required.
        '''
        with h5py.File(self.radiation_file, 'r') as radiation_loader:
            # get the radiation model parameters
            mesh_ind = np.ix_(self.msoa_indices, self.msoa_indices)
            self.radiation_mean = radiation_loader['radiation'][:,:][mesh_ind]
            self.theta = radiation_loader['theta'][:]
