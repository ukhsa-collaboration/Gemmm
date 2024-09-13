'''
A series of tests to check that model outputs are as expected and invalid arguments raise
appropriate exceptions.
'''

import unittest
import numpy as np
import pathlib
from gemmm import OriginDestination


class TestModel(unittest.TestCase):
    '''
    Test the arguments and outputs for the OriginDestination model
    '''
    def setUp(self):
        '''
        Prepare the OriginDestination model using a small number of MSOAs.
        Called immediately before each test.
        '''
        msoas = ['E02000001', # City of london 001
                 'E02000977', # Westminster 018
                 'E02000808', # Southwark 002
                 'E02006801', # Lambeth 036
                 'E02006292', # Suffolk Coastal 006
                 'E02006272', # Mid Suffolk 012
                 'E02006251', # Ipswich 007
                 'E02006254', # Ipswich 010
                ]
        self.n_msoas = len(msoas)
        self.od_model = OriginDestination(msoas=msoas, day_type='weekday')


    def test_mean(self):
        ''' Checking the Fourier series mean is as expected. '''
        # check the mean for fourier series pairs at a single hour
        expected_mean_7 = np.array([[984.    ,  97.3750,  42.    ,  21.  ,  0.0001,  0.0553,
                                       2.    ,   2.5645],
                                    [ 88.1250, 402.    ,  17.    ,  26.  ,  0.    ,  0.    ,
                                       0.    ,   0.0001],
                                    [108.8750,  31.5   , 175.5   ,  20.75,  0.    ,  0.    ,
                                       0.    ,   0.    ],
                                    [ 74.    ,  49.9688,  20.9219, 246.  ,  0.    ,  0.    ,
                                       0.    ,   0.    ],
                                    [  4.6719,   0.5449,   0.    ,   0.  , 90.5   , 11.5781,
                                      18.5   ,  24.3594],
                                    [  2.4395,   0.    ,   0.    ,   0.  , 10.    , 93.5   ,
                                      34.6875,  31.1094],
                                    [ 11.6094,   0.1638,   0.    ,   0.  ,  6.7266, 14.9375,
                                     307.    ,   5.4258],
                                    [ 16.5   ,   1.1162,   0.1943,   0.  ,  5.2930, 18.5469,
                                       3.9512, 210.375 ]])

        hour = 7
        mean_7 = np.zeros((self.n_msoas, self.n_msoas))
        mean_7[self.od_model.fourier_data.row,
               self.od_model.fourier_data.col] = self.od_model.fourier_data.mean[hour]

        np.testing.assert_allclose(mean_7, expected_mean_7, rtol=0.001)


    def test_exceptions(self):
        ''' Checking the correct excecptions are raised when invalid arguments are provided. '''

        # Check MSOAs
        # expects list or numpy array, not tuple
        self.assertRaises(TypeError, OriginDestination, msoas=('E02000001', 'E02000977'),
                          day_type='weekday')
        # INVALID01 not in the list of valid MSOAs
        self.assertRaises(ValueError, OriginDestination, msoas=['E02000001', 'INVALID01'],
                          day_type='weekday')
        # expects more than one MSOA
        self.assertRaises(ValueError, OriginDestination, msoas=['E02000001'], day_type='weekday')

        # Check day_type
        # expects day_type to be a string
        self.assertRaises(TypeError, OriginDestination, msoas=['E02000001', 'E02000977'],
                          day_type=1)
        # expects day_type to be 'weekend' or 'weekday'
        self.assertRaises(ValueError, OriginDestination, msoas=['E02000001', 'E02000977'],
                          day_type='monday')

        # Check the sampling arguments
        # expects hours to be integers
        self.assertRaises(TypeError, self.od_model.generate_sample, hours=[2.0, 3],
                          n_realizations=1)
        # expects hours to be between 0 and 23
        self.assertRaises(ValueError, self.od_model.generate_sample, hours=24, n_realizations=1)
        # expects n_realizations to be an integer
        self.assertRaises(TypeError, self.od_model.generate_sample, hours=7, n_realizations=1.0)
        # expects n_realizations > 0
        self.assertRaises(ValueError, self.od_model.generate_sample, hours=[7, 8, 9],
                          n_realizations=-5)

        # Check the loading arguments
        # expects file to be a string or pathlib.Path
        self.assertRaises(TypeError, OriginDestination, file=123)
        # expects an existing netcdf file
        self.assertRaises(FileNotFoundError, OriginDestination, file='a/path/to/no/file.nc')


    def test_sample(self):
        ''' Checking the sampled number of journeys is as expected. '''

        expected_sample = np.array([[918, 108,  46,  20,   0,   0,   1,   2],
                                    [ 76, 436,  17,  20,   0,   0,   0,   0],
                                    [127,  18, 162,  14,   0,   0,   0,   0],
                                    [ 62,  45,  22, 249,   0,   0,   0,   0],
                                    [  6,   2,   0,   0,  92,  10,  17,  26],
                                    [  6,   0,   0,   0,  11,  80,  34,  22],
                                    [ 20,   0,   0,   0,  10,  24, 324,   2],
                                    [ 14,   0,   0,   0,  13,  20,   4, 181]], dtype=np.int16)

        np.random.seed(1234)
        self.od_model.generate_sample(hours=7, n_realizations=1, save_sample=False)
        
        generated_sample = self.od_model.samples[(7, 0)].toarray()
        np.testing.assert_array_equal(generated_sample, expected_sample)


    def test_load(self):
        ''' Checking the netCDF4 loaded sample is as expected. '''

        # (the file it is reading from contains only one realization for hour 7)
        file_name = pathlib.Path(__file__).parent.joinpath('test_sample.nc')
        sample_loader = OriginDestination(file=file_name)
        # expects an hour included in the file
        self.assertRaises(ValueError, sample_loader.load_sample, hour=0)
        # expects a realization included in the file
        self.assertRaises(ValueError, sample_loader.load_sample, hour=7, realization=100)

        # check the number of MSOAs, hours and n_realizations in the loader is correct
        self.assertEqual(len(sample_loader.msoas), self.n_msoas)
        np.testing.assert_array_equal(sample_loader.hours, [7])
        self.assertEqual(sample_loader.n_realizations, 1)

        expected_sample = np.array([[  0,   0, 918],
                                    [  0,   1, 108],
                                    [  0,   2,  46],
                                    [  0,   3,  20],
                                    [  0,   6,   1],
                                    [  0,   7,   2],
                                    [  1,   0,  76],
                                    [  1,   1, 436],
                                    [  1,   2,  17],
                                    [  1,   3,  20],
                                    [  2,   0, 127],
                                    [  2,   1,  18],
                                    [  2,   2, 162],
                                    [  2,   3,  14],
                                    [  3,   0,  62],
                                    [  3,   1,  45],
                                    [  3,   2,  22],
                                    [  3,   3, 249],
                                    [  4,   0,   6],
                                    [  4,   1,   2],
                                    [  4,   4,  92],
                                    [  4,   5,  10],
                                    [  4,   6,  17],
                                    [  4,   7,  26],
                                    [  5,   0,   6],
                                    [  5,   4,  11],
                                    [  5,   5,  80],
                                    [  5,   6,  34],
                                    [  5,   7,  22],
                                    [  6,   0,  20],
                                    [  6,   4,  10],
                                    [  6,   5,  24],
                                    [  6,   6, 324],
                                    [  6,   7,   2],
                                    [  7,   0,  14],
                                    [  7,   4,  13],
                                    [  7,   5,  20],
                                    [  7,   6,   4],
                                    [  7,   7, 181]], dtype=np.int16)

        loaded_sample = sample_loader.load_sample(hour=7, realization=0, as_pandas=False)
        np.testing.assert_array_equal(loaded_sample, expected_sample)


# see pygom/.github/workflows/test_package.yml for setting up testing

if __name__ == '__main__':
    unittest.main()
