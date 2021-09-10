
import os
import operator
import numpy as np
from . import _flatten_dtype
from ._readtextmodule import _readtext_from_file_object


def _check_nonneg_int(value, name="argument"):
    try:
        operator.index(value)
    except TypeError:
        raise TypeError(f"{name} must be an integer") from None
    if value < 0:
        raise ValueError(f"{name} must be nonnegative")


def _preprocess_comments(iterable, comments, encoding):
    """
    Generator that consumes a line iterated iterable and strips out the
    multiple (or multi-character) comments from lines.
    This is a pre-processing step to achieve feature parity with loadtxt
    (we assume that this feature is a nieche feature).
    """
    for line in iterable:
        if isinstance(line, bytes):
            # Need to handle conversion here, or the splitting would fail
            line = line.decode(encoding)

        for c in comments:
            line = line.split(c, 1)[0]

        yield line


def read(file, *, delimiter=',', comment='#', quote='"',
         decimal='.', sci='E', imaginary_unit='j',
         usecols=None, skiprows=0,
         max_rows=None, converters=None, ndmin=None, unpack=False,
         dtype=np.float64, encoding=None):
    r"""
    Read a NumPy array from a text file.

    Parameters
    ----------
    file : str or file object
        The filename or the file to be read.
    delimiter : str, optional
        Field delimiter of the fields in line of the file.
        Default is a comma, ','.
    comment : str or sequence of str, optional
        Character that begins a comment.  All text from the comment
        character to the end of the line is ignored.
        Multiple comments or multiple-character comment strings are supported,
        but may be slower and `quote` must be empty if used.
    quote : str, optional
        Character that is used to quote string fields. Default is '"'
        (a double quote).
    decimal : str, optional
        The decimal point character.  Default is '.'.
    sci : str, optional
        The character in front of the exponent when exponential notation
        is used for floating point values.  The default is 'E'.  The value
        is case-insensitive.
    imaginary_unit : str, optional
        Character that represent the imaginay unit `sqrt(-1)`.
        Default is 'j'.
    usecols : array_like, optional
        A one-dimensional array of integer column numbers.  These are the
        columns from the file to be included in the array.  If this value
        is not given, all the columns are used.
    skiprows : int, optional
        Number of lines to skip before interpreting the data in the file.
    max_rows : int, optional
        Maximum number of rows of data to read.  Default is to read the
        entire file.
    ndmin : int, optional
        Minimum dimension of the array returned.
        Allowed values are 0, 1 or 2.  Default is 0.
    unpack : bool, optional
        If True, the returned array is transposed, so that arguments may be
        unpacked using ``x, y, z = read(...)``.  When used with a structured
        data-type, arrays are returned for each field.  Default is False.
    dtype : numpy data type
        A NumPy dtype instance, can be a structured dtype to map to the
        columns of the file.
    encoding : str, optional
        Specifies the encoding of the input file.

    Returns
    -------
    ndarray
        NumPy array.

    Examples
    --------
    First we create a file for the example.

    >>> s1 = '1.0,2.0,3.0\n4.0,5.0,6.0\n'
    >>> with open('example1.csv', 'w') as f:
    ...     f.write(s1)
    >>> a1 = read_from_filename('example1.csv')
    >>> a1
    array([[1., 2., 3.],
           [4., 5., 6.]])

    The second example has columns with different data types, so a
    one-dimensional array with a structured data type is returned.
    The tab character is used as the field delimiter.

    >>> s2 = '1.0\t10\talpha\n2.3\t25\tbeta\n4.5\t16\tgamma\n'
    >>> with open('example2.tsv', 'w') as f:
    ...     f.write(s2)
    >>> a2 = read_from_filename('example2.tsv', delimiter='\t')
    >>> a2
    array([(1. , 10, b'alpha'), (2.3, 25, b'beta'), (4.5, 16, b'gamma')],
          dtype=[('f0', '<f8'), ('f1', 'u1'), ('f2', 'S5')])
    """
    # TODO: encoding needs a bit closer look still to compare with loadtxt.
    #       1. loadtxt tries to guess encoding from the object.
    #       2. Does loadtxt actually use the preferred encoding, or only
    #          indirectly through the mechanism in 1.?
    #          (This matters for an iterable of byte!)
    if encoding is None:
        import locale
        encoding = locale.getpreferredencoding()

    if dtype is None:
        raise TypeError("a dtype must be provided.")
    dtype = np.dtype(dtype)

    # FIXME: Temporary hack just to get some loadtxt test to pass!
    ##if dtype == np.dtype('U0'):
    ##    dtype = np.dtype('U32')

    if usecols is not None:
        # Allow usecols to be a single int or a sequence of ints
        try:
            usecols_as_list = list(usecols)
        except TypeError:
            usecols_as_list = [usecols]
        for col_idx in usecols_as_list:
            try:
                operator.index(col_idx)
            except TypeError:
                # Some unit tests for numpy.loadtxt require that the
                # error message matches this format.
                raise TypeError(
                    "usecols must be an int or a sequence of ints but "
                    "it contains at least one element of type %s" %
                    type(col_idx),
                    ) from None
        # Fall back to existing code
        usecols = np.array([operator.index(i) for i in usecols_as_list],
                           dtype=np.int32)

    if converters is not None:
        if not isinstance(converters, dict):
            raise TypeError('converters must be a dictionary')
        for key, func in converters.items():
            try:
                operator.index(key)
            except TypeError:
                raise TypeError('keys of the converters dictionary must '
                                f'be integers; got {key!r}') from None
            if not callable(func):
                raise TypeError('values of the converters dictionary must '
                                'be callable, but the value associated with '
                                f'the key {key!r} is not')

    if ndmin not in [None, 0, 1, 2]:
        raise ValueError(f'ndmin must be None, 0, 1, or 2; got {ndmin}')

    if not isinstance(comment, str):
        # assume comments are a sequence of strings
        comments = tuple(comment)
        comment = ''
        # If there is only one comment, and that comment has one character,
        # the normal parsing can deal with it just fine.
        if len(comments) == 1:
            if isinstance(comments[0], str) and len(comments[0]) == 1:
                comment = comments[0]
                comments = None
    elif len(comment) > 1:
        comments = (comment,)
        comment = ''
    else:
        comments = None

    # comment is now either a 1 or 0 character string or a tuple:
    if comments is not None:
        assert comment == ''
        # Note: An earlier version support two character comments (and could
        #       have been extended to multiple characters, we assume this is
        #       rare enough to not optimize for.
        if quote != "":
            raise ValueError(
                "when multiple comments or a multi-character comment is given, "
                "quotes are not supported.  In this case the quote character "
                "must be set to the empty string: `quote=''`.")
    else:
        # No preprocessing necessary
        assert comments is None

    if len(imaginary_unit) != 1:
        raise ValueError('len(imaginary_unit) must be 1.')

    _check_nonneg_int(skiprows)
    if max_rows is not None:
        _check_nonneg_int(max_rows)
    else:
        # Passing -1 to the C code means "read the entire file".
        max_rows = -1

    # Compute `codes` and `sizes`.  These are derived from `dtype`, and we
    # also pass `dtype` to the C function, so we're passing in redundant
    # information.  This is because it is easier to write the code that
    # creates `codes` and `sizes` using Python than C.
    if dtype is not None:
        dtypes = np.lib._iotools.flatten_dtype(dtype, flatten_base=True)
        if (len(dtypes) != 1 and usecols is not None and
                len(dtypes) != len(usecols)):
            raise ValueError(f"length of usecols ({len(usecols)}) and "
                             f"number of fields in dtype ({len(codes)}) "
                             "do not match.")
    else:
        dtypes = None

    if isinstance(file, os.PathLike):
        fname = os.fspath(file)

    # TODO: loadtxt actually uses `file + ''` to decide this
    if isinstance(file, str):
        # We open a file-like object (using datasource).  Since we own the file
        # we do not have to worry about how much we read.
        f = np.lib._datasource.open(file, 'rt', encoding=encoding)
        try:
            if comments is not None:
                f = _preprocess_comments(iter(f), comments, encoding)
            arr = _readtext_from_file_object(
                    f, delimiter=delimiter, comment=comment, quote=quote,
                    decimal=decimal, sci=sci, imaginary_unit=imaginary_unit,
                    usecols=usecols, skiprows=skiprows, max_rows=max_rows,
                    converters=converters, dtype=dtype, dtypes=dtypes,
                    encoding=encoding, filelike=comments is None)
        finally:
            f.close()
    else:
        # TODO: If the user passed an open file, loadtxt guarantees reading
        #       only as far as necessary (and guarantees it does that).
        #       However, that makes this path slower than the above one.
        file = iter(file)
        if comments is not None:
            file = _preprocess_comments(file, comments, encoding)

        arr = _readtext_from_file_object(
            file, delimiter=delimiter, comment=comment, quote=quote,
            decimal=decimal, sci=sci, imaginary_unit=imaginary_unit,
            usecols=usecols, skiprows=skiprows, max_rows=max_rows,
            converters=converters, dtype=dtype, dtypes=dtypes,
            encoding=encoding, filelike=False)

    if ndmin is not None:
        # Handle non-None ndmin like np.loadtxt.  Might change this eventually?
        # Tweak the size and shape of the arrays - remove extraneous dimensions
        if arr.ndim > ndmin:
            arr = np.squeeze(arr)
        # and ensure we have the minimum number of dimensions asked for
        # - has to be in this order for the odd case ndmin=1,
        # X.squeeze().ndim=0
        if arr.ndim < ndmin:
            if ndmin == 1:
                arr = np.atleast_1d(arr)
            elif ndmin == 2:
                arr = np.atleast_2d(arr).T

    if unpack:
        # Handle unpack like np.loadtxt.
        # XXX Check interaction with ndmin!
        dt = arr.dtype
        if dt.names is not None:
            # For structured arrays, return an array for each field.
            return [arr[field] for field in dt.names]
        else:
            return arr.T
    else:
        return arr
