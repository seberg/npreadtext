//
//  Requires C99.
//

#include <stdio.h>
#include <stdbool.h>

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#define PY_ARRAY_UNIQUE_SYMBOL npreadtext_ARRAY_API
#include "numpy/arrayobject.h"

#include "parser_config.h"
#include "stream_file.h"
#include "stream_python_file_by_line.h"
#include "field_types.h"
#include "analyze.h"
#include "rows.h"
#include "error_types.h"

#define LOADTXT_COMPATIBILITY true


static void
raise_analyze_exception(int nrows, char *filename)
{
    if (nrows == ANALYZE_OUT_OF_MEMORY) {
        if (filename) {
            PyErr_Format(PyExc_MemoryError,
                         "Out of memory while analyzing '%s'", filename);
        } else {
            PyErr_Format(PyExc_MemoryError,
                         "Out of memory while analyzing file.");
        }
    }
    else if (nrows == ANALYZE_FILE_ERROR) {
        if (filename) {
            PyErr_Format(PyExc_RuntimeError,
                         "File error while analyzing '%s'", filename);
        } else {
            PyErr_Format(PyExc_RuntimeError,
                         "File error while analyzing file.");
        }
    }
    else {
        if (filename) {
            PyErr_Format(PyExc_RuntimeError,
                         "Unknown error when analyzing '%s'", filename);
        } else {
            PyErr_Format(PyExc_RuntimeError,
                         "Unknown error when analyzing file.");
        }
    }
}


//
// `usecols` must point to a Python object that is Py_None or a 1-d contiguous
// numpy array with data type int32.
//
// `dtype` must point to a Python object that is Py_None or a numpy dtype
// instance.  If the latter, code and sizes must be arrays of length
// num_dtype_fields, holding the flattened data field type codes and byte
// sizes. (num_dtype_fields, codes, and sizes can be inferred from dtype,
// but we do that in Python code.)
//
// If both `usecols` and `dtype` are not None, and the data type is compound,
// then len(usecols) must equal num_dtype_fields.
//
// If `dtype` is given and it is compound, and `usecols` is None, then the
// number of columns in the file must match the number of fields in `dtype`.
//
static PyObject *
_readtext_from_stream(stream *s, char *filename, parser_config *pc,
                      PyObject *usecols, Py_ssize_t skiprows, Py_ssize_t max_rows,
                      PyObject *converters,
                      PyObject *dtype, PyArray_Descr **dtypes,
                      int num_dtype_fields)
{
    PyArrayObject *arr = NULL;
    PyArray_Descr *out_dtype = NULL;
    int32_t *cols;
    int ncols;
    npy_intp nrows;
    int num_fields;
    field_type *ft = NULL;

    bool homogeneous;
    npy_intp shape[2];

    if (dtype == Py_None) {
        // Make the first pass of the file to analyze the data type
        // and count the number of rows.
        // analyze() will assign num_fields and create the ft array,
        // based on the types of the data that it finds in the file.
        // XXX Note that analyze() does not use the usecols data--it
        // analyzes (and fills in ft for) all the columns in the file.
        nrows = analyze(s, pc, skiprows, -1, &num_fields, &ft);
        if (nrows < 0) {
            raise_analyze_exception(nrows, filename);
            return NULL;
        }
        stream_seek(s, 0);

        homogeneous = field_types_is_homogeneous(num_fields, ft);
        if (field_types_init_descriptors(num_fields, ft) < 0) {
            goto finish;
        }
        if (homogeneous) {
            out_dtype = ft[0].descr;
            Py_INCREF(out_dtype);
        }
        else {
            out_dtype = field_types_to_descr(num_fields, ft);
            if (out_dtype == NULL) {
                goto finish;
            }
        }
    }
    else {
        /*
         * If dtypes[0] is dtype the input was not structured and the result
         * is considered "homogeneous" and we have to discover the number of
         * columns/
         */
        out_dtype = (PyArray_Descr *)dtype;
        Py_INCREF(out_dtype);

        /* TODO: Ridiculous, should just pass it in (or reuse num_fields) */
        homogeneous = num_dtype_fields == 1 && (out_dtype == dtypes[0]);

        // A dtype was given.
        num_fields = num_dtype_fields;
        ft = field_types_create(num_fields, dtypes);
        if (ft == NULL) {
            PyErr_Format(PyExc_MemoryError, "out of memory");
            return NULL;
        }
        nrows = max_rows;
    }
    if (usecols == Py_None) {
        ncols = num_fields;
        cols = NULL;
    }
    else {
        ncols = PyArray_SIZE(usecols);
        cols = PyArray_DATA(usecols);
    }

    // XXX In the one-pass case, we don't have nrows.
    shape[0] = nrows;
    if (homogeneous) {
        shape[1] = ncols;
    }

    if (dtype == Py_None) {
        int ndim = homogeneous ? 2 : 1;

        Py_INCREF(out_dtype);
        PyArrayObject *tmp_arr = (PyArrayObject *)PyArray_SimpleNewFromDescr(
                ndim, shape, out_dtype);
        if (tmp_arr == NULL) {
            goto finish;
        }
        Py_ssize_t num_rows = nrows;
        arr = read_rows(
                s, &num_rows, num_fields, ft, pc,
                ncols, cols, skiprows, converters,
                tmp_arr, out_dtype, homogeneous);
        Py_DECREF(tmp_arr);  /* should be identical to `arr`, but we own it */
        if (arr == NULL) {
            goto finish;
        }
    }
    else {
        Py_ssize_t num_rows = nrows;

        arr = read_rows(
                s, &num_rows, num_fields, ft, pc,
                ncols, cols, skiprows, converters,
                NULL, out_dtype, homogeneous);
        if (arr == NULL) {
            goto finish;
        }
    }

  finish:
    Py_XDECREF(out_dtype);
    if (ft != NULL) {
        field_types_clear(num_fields, ft);
        free(ft);
    }
    return (PyObject *)arr;
}


static int
parse_control_character(PyObject *obj, Py_UCS4 *character)
{
    if (!PyUnicode_Check(obj) || PyUnicode_GetLength(obj) > 1) {
        PyErr_Format(PyExc_TypeError,
                "Control character must be a single unicode character or "
                "empty unicode string; but got: %.100R", obj);
        return 0;
    }
    if (PyUnicode_GET_LENGTH(obj) == 0) {
        /* TODO: This sets it to a non-existing character, could use NUL */
        *character = (Py_UCS4)-1;
        return 1;
    }
    *character = PyUnicode_READ_CHAR(obj, 0);
    return 1;
}


static PyObject *
_readtext_from_file_object(PyObject *self, PyObject *args, PyObject *kwargs)
{
    static char *kwlist[] = {"file", "delimiter", "comment", "quote",
                             "decimal", "sci", "imaginary_unit",
                             "usecols", "skiprows",
                             "max_rows", "converters",
                             "dtype", "dtypes",
                             "encoding", NULL};
    PyObject *file;
    Py_ssize_t skiprows = 0;
    Py_ssize_t max_rows = -1;
    PyObject *usecols = Py_None;
    PyObject *converters = Py_None;

    PyObject *dtype = Py_None;
    PyObject *dtypes_obj = Py_None;
    PyObject *encoding = Py_None;

    PyArray_Descr **dtypes = NULL;

    parser_config pc = {
        .delimiter = ',',
        .comment = '#',
        .quote = '"',
        .decimal = '.',
        .sci = 'E',
        .imaginary_unit = 'j',
        .allow_float_for_int = true,
        .allow_embedded_newline = true,
        .delimiter_is_whitespace = false,
        .ignore_leading_whitespace = false,
        .ignore_trailing_spaces = false,
        .ignore_blank_lines = true,
        .strict_num_fields = false,
    };

    PyObject *arr = NULL;
    int num_dtype_fields;

    if (!PyArg_ParseTupleAndKeywords(
            args, kwargs, "O|$O&O&O&O&O&O&OnnOOOO", kwlist,
            &file,
            &parse_control_character, &pc.delimiter,
            &parse_control_character, &pc.comment,
            &parse_control_character, &pc.quote,
            &parse_control_character, &pc.decimal,
            &parse_control_character, &pc.sci,
            &parse_control_character, &pc.imaginary_unit,
            &usecols, &skiprows, &max_rows, &converters,
            &dtype, &dtypes_obj, &encoding)) {
        return NULL;
    }

    if (pc.delimiter == (Py_UCS4)-1) {
        /* TODO: We can allow a '\0' delimiter; need to refine argparsing */
        pc.delimiter_is_whitespace = true;
        /* Ignore leading whitespace to match `string.split(None)` */
        pc.ignore_leading_whitespace = true;
    }

    /*
     * TODO: This needs some hefty input validation!
     */
    if (dtypes_obj == Py_None) {
        assert(dtype == Py_None);
        num_dtype_fields = -1;
    }
    else {
        dtypes_obj = PySequence_Fast(dtypes_obj, "dtypes not a sequence :(");
        if (dtypes_obj == NULL) {
            return NULL;
        }
        num_dtype_fields = PySequence_Fast_GET_SIZE(dtypes_obj);
        dtypes = (PyArray_Descr **)PySequence_Fast_ITEMS(dtypes_obj);
    }

    stream *s = stream_python_file_by_line(file, encoding);
    if (s == NULL) {
        PyErr_Format(PyExc_RuntimeError, "Unable to access the file.");
        Py_DECREF(dtypes_obj);
        return NULL;
    }

    arr = _readtext_from_stream(s, NULL, &pc, usecols, skiprows, max_rows,
                                converters,
                                dtype, dtypes, num_dtype_fields);
    Py_DECREF(dtypes_obj);
    stream_close(s, RESTORE_NOT);
    return arr;
}


// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
// Python extension module definition.
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

PyMethodDef module_methods[] = {
    {"_readtext_from_file_object", (PyCFunction) _readtext_from_file_object,
         METH_VARARGS | METH_KEYWORDS, "testing"},
    {0} // sentinel
};

static struct PyModuleDef moduledef = {
    .m_base     = PyModuleDef_HEAD_INIT,
    .m_name     = "_readtextmodule",
    .m_size     = -1,
    .m_methods  = module_methods,
};


PyMODINIT_FUNC
PyInit__readtextmodule(void)
{
    PyObject* m = NULL;

    //
    // Initialize numpy.
    //
    import_array();
    if (PyErr_Occurred()) {
        return NULL;
    }

    // ----------------------------------------------------------------
    // Finish the extension module creation.
    // ----------------------------------------------------------------  

    // Create module
    m = PyModule_Create(&moduledef);

    return m;
}
