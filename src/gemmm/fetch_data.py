import numpy as np
import pooch
from .version import __version__

BRIAN = pooch.create(
    # sensible default for the cache
    path = pooch.os_cache('gemmm-model-data'),
    #base_url='https://github.com/jontycarruthersphe/od_test/tree/main/model_data/',
    base_url='https://github.com/jontycarruthersphe/od_test/raw/main/model_data/', # needs the raw
    # The registry specifies the files that can be fetched
    registry = {
        'manchester_msoas.npy': 'sha256:a1e3b2563287142297c53cb799f76b34990c76809e5b2ac51ba4ac7ef83a036d'
        }
    )


def fetch_manchester_msoas():
    '''
    Load the numpy array of Manchester MSOAs
    '''
    # The file will be downloaded automatically the first time this is run
    # returns the file path to the downloaded file. Afterwards, Pooch finds
    # it in the local cache and doesn't repeat the download.
    fname = BRIAN.fetch('manchester_msoas.npy')
    # The "fetch" method returns the full path to the downloaded data file.
    # All we need to do now is load it with our standard Python tools.
    array = np.load(fname, allow_pickle=True)
    return array


'''
Model data cache locations are:
    Windows: C:\\Users\\<user>\\AppData\\Local\\<AppAuthor>\\gemmm-model-data\\Cache
'''
