from os import path
import pytest
import numpy as np
from numpy.testing import assert_array_equal, assert_
from readtextstream import read_from_filename


@pytest.mark.parametrize('basename,delim', [('test1.csv', ','),
                                            ('test1.tsv', '\t')])
def test1_read_from_filename1(basename, delim):
    filename = path.join(path.dirname(__file__), 'data', basename)
    a = read_from_filename(filename, delimiter=delim)
    al = np.loadtxt(filename, delimiter=delim)
    assert_array_equal(a, al)


@pytest.mark.parametrize('basename,sci', [('test1e.csv', 'E'),
                                          ('test1d.csv', 'D')])
def test1_read_from_filename_sci(basename, sci):
    filename = path.join(path.dirname(__file__), 'data', basename)
    a = read_from_filename(filename, sci=sci)
    filename_e = path.join(path.dirname(__file__), 'data', 'test1e.csv')
    al = np.loadtxt(filename_e, delimiter=',')
    assert_array_equal(a, al)


@pytest.mark.parametrize('basename,delim', [('test1.csv', ','),
                                            ('test1.tsv', '\t')])
def test1_read_from_filename_usecols(basename, delim):
    filename = path.join(path.dirname(__file__), 'data', basename)
    a = read_from_filename(filename, usecols=[0, 2], delimiter=delim)
    al = np.loadtxt(filename, delimiter=delim, usecols=[0, 2])
    assert_array_equal(a, al)


def test1_read_from_filename_with_comment():
    filename = path.join(path.dirname(__file__),
                         'data', 'test1_with_comments.csv')
    a = read_from_filename(filename, comment='#')
    al = np.loadtxt(filename, delimiter=',')
    assert_array_equal(a, al)


def test_decimal_is_comma():
    filename = path.join(path.dirname(__file__),
                         'data', 'decimal_is_comma.txt')
    a = read_from_filename(filename, decimal=',', delimiter=' ')
    expected = np.array([[1.5, 1.75, 2.0],
                         [2.5, 2.75, 3.0],
                         [3.5, 3.75, 4.0],
                         [4.5, 4.75, 5.0]])
    assert_array_equal(a, expected)


def test_quoted_field():
    filename = path.join(path.dirname(__file__),
                         'data', 'quoted_field.csv')
    a = read_from_filename(filename)
    expected_dtype = np.dtype([('f0', 'S8'), ('f1', np.float64)])
    assert_(a.dtype == expected_dtype)
    expected = np.array([('alpha, x', 2.5),
                         ('beta, x', 4.5),
                         ('gamma, x', 5.0)], dtype=expected_dtype)
    assert_array_equal(a, expected)