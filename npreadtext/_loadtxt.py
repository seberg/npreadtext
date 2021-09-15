
import numpy as np
from ._readers import read


def _loadtxt(fname, dtype=float, comments='#', delimiter=None,
             converters=None, skiprows=0, usecols=None, unpack=False,
             ndmin=0, encoding='bytes', max_rows=None, **kwargs):
    """
    Monkeypatched version of `np.loadtxt`.  Unlike loadtxt it allows some
    additional keyword arguments, such as `quote='"'`.
    Please check `npreadtxt.read` for details.

    """
    if delimiter is None:
        delimiter = ''
    elif isinstance(delimiter, bytes):
        delimiter.decode("latin1")

    if dtype is None:
        dtype = np.float64

    if ndmin is None:
        ndmin = 0
    if ndmin not in [0, 1, 2]:
        raise ValueError(f'Illegal value of ndmin keyword: {ndmin}')

    comment = comments
    # Type conversions for Py3 convenience
    if comment is None:
        comment = ''
    else:
        if isinstance(comment, (str, bytes)):
            comment = [comment]
        comment = [x.decode('latin1') if isinstance(x, bytes) else x for x in comment]

    # Disable quoting unless passed:
    quote = kwargs.pop('quote', '')

    arr = read(fname, dtype=dtype, comment=comment, delimiter=delimiter,
               converters=converters, skiprows=skiprows, usecols=usecols,
               unpack=unpack, ndmin=ndmin, encoding=encoding,
               max_rows=max_rows, quote=quote, **kwargs)

    return arr


try:
    # Try giving some reasonable docs, but __doc__ could be None.
    _loadtxt.__doc__ += np.loadtxt.__doc__
except:
    pass