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
    '''

    def __init__(self, msoas, day_type):
        data_url = 'https://github.com/ukhsa-collaboration/gemmm/raw/refs/heads/main/model_data/'
        cache_dir = 'gemmm-model-data'

        goodboy = pooch.create(
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

        #downloader_auth = pooch.HTTPDownloader(progressbar=True)
        self.fourier_file = goodboy.fetch(f'fourier_data_{day_type}.hdf5')
                                          #downloader=downloader_auth)

        self.radiation_file = goodboy.fetch(f'radiation_data_{day_type}.hdf5')
                                            #downloader=downloader_auth)

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
            new_row = self.find_indices(self.msoa_indices, row[idx_mask])
            new_col = self.find_indices(self.msoa_indices, col[idx_mask])

            # faster to first load the whole array when the number of MSOAs is large,
            # rather than fourier_loader['fourier'][:, idx_mask]
            fourier_mean = fourier_loader['fourier'][...]
            fourier_mean = fourier_mean[:, idx_mask]

            # load the overdispersion parameters
            overdispersion = fourier_loader['overdispersion'][:]

        return FourierData(new_row, new_col, fourier_mean, overdispersion)


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
            radiation_mean = radiation_loader['radiation'][:,:][mesh_ind]
            theta = radiation_loader['theta'][:]

        return RadiationData(radiation_mean, theta)



class RadiationData():
    ''' Class for storing the data needed to run the radiation model '''

    def __init__(self, mean, theta):
        mean.setflags(write=False)
        theta.setflags(write=False)
        self._mean = mean
        self._theta = theta

    @property
    def mean(self):
        '''
        Returns
        -------
        np.ndarray (n_msoas, n_msoas)
            Contains the unscaled means for the radiation model
        '''
        return self._mean

    @mean.setter
    def mean(self, value):
        raise AttributeError('Radiation mean is read-only')

    @property
    def theta(self):
        '''
        Returns
        -------
        np.ndarray (24, )
            Contains the scale factors for each hour of the day that multiply the
            radiation_mean when sampling from a Poisson distribution.
        '''
        return self._theta

    @theta.setter
    def theta(self, value):
        raise AttributeError('Radiation theta is read-only')



class FourierData():
    ''' Class for storing the data needed to run the Fourier series model '''

    def __init__(self, row, col, mean, overdispersion):
        row.setflags(write=False)
        col.setflags(write=False)
        mean.setflags(write=False)
        overdispersion.setflags(write=False)
        self._row = row
        self._col = col
        self._mean = mean
        self._overdispersion = overdispersion

    @property
    def row(self):
        '''
        Returns
        -------
        np.ndarray (n_nonzero, )
            Contains the indices of the start MSOAs
        '''
        return self._row

    @row.setter
    def row(self, value):
        raise AttributeError('Fourier row is read-only')

    @property
    def col(self):
        '''
        Returns
        -------
        np.ndarray (n_nonzero, )
            Contains the indices of the end MSOAs
        '''
        return self._col

    @col.setter
    def col(self, value):
        raise AttributeError('Fourier col is read-only')

    @property
    def mean(self):
        '''
        Returns
        -------
        np.ndarray (24, n_nonzero)
            Contains the mean number of journeys between MSOAs
        '''
        return self._mean

    @mean.setter
    def mean(self, value):
        raise AttributeError('Fourier mean is read-only')

    @property
    def overdispersion(self):
        '''
        Returns
        -------
        np.ndarray (24, )
            Contains the overdispersion parameters for each hour of the day that
            are needed to sample from the negative binomial distribution
        '''
        return self._overdispersion

    @overdispersion.setter
    def overdispersion(self, value):
        raise AttributeError('Fourier overdispersion is read-only')
