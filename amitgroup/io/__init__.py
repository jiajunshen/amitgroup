from __future__ import absolute_import

from .mnist import load_mnist
from .norb import load_small_norb
from .casia import load_casia
from .casia import load_casia_resize
from .cifar import load_cifar10
from .book import *
from .norb import load_small_norb
from .examples import load_example

try:
    import tables
    _pytables_ok = True 
except ImportError:
    _pytables_ok = False

if _pytables_ok:
    from .hdf5io import load, save
else:
    def _f(*args, **kwargs):
        raise ImportError("You need PyTables for this function") 
    load = save = _f
