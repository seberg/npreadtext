
import numpy as np
from ._readers import read


def _loadtxt(*args, **kwds):
    delimiter = kwds.pop('delimiter', None)
    if delimiter is None:
        delimiter = ''

    dtype = kwds.pop('dtype', None)
    if dtype is None:
        dtype = np.float64

    ndmin = kwds.pop('ndmin', None)
    if ndmin is None:
        ndmin = 0
    if ndmin not in [0, 1, 2]:
        raise ValueError(f'Illegal value of ndmin keyword: {ndmin}')

    comment = kwds.pop('comments', None)
    # Type conversions for Py3 convenience
    if comment is None:
        comment = '#'
    else:
        if isinstance(comment, (str, bytes)):
            comment = [comment]
        comment = [x.decode('latin1') if type(x) is bytes else x for x in comment]

    try:
        arr = read(*args, delimiter=delimiter, dtype=dtype,
                   comment=comment, quote='', **kwds)
    except RuntimeError as exc:
        raise ValueError(exc.args) from None

    # For now, ignore 'ndmin' if 'unpack' was given.
    if 'unpack' not in kwds:
        if arr.shape == (0, 0):
            arr.resize((0, 1))
        if ndmin == 2:
            if arr.shape == (0, 0):
                arr = arr.reshape((0, 1))
        elif ndmin == 1:
            arr = np.atleast_1d(np.squeeze(arr))
        else:
            arr = np.squeeze(arr)

    return arr
