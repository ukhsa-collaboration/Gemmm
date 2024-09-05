#__import__("pkg_resources").declare_namespace(__name__) # (?) might not need this

from pkg_resources import get_distribution, DistributionNotFound
from .model import OriginDestination

__all__ = ['OriginDestination']

# looks for tag in git
try:
    __version__ = get_distribution(__name__).version
except DistributionNotFound:
    # package is not installed
    pass