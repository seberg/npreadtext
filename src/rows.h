
#ifndef _ROWS_H_
#define _ROWS_H_

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <stdio.h>

#include "stream.h"
#include "field_types.h"
#include "parser_config.h"

//
// This structure holds information about errors arising
// in read_rows().
//
typedef struct _read_error {
    int error_type;
    int line_number;
    int field_number;
    int char_position;
    PyArray_Descr *descr;
    // int32_t itemsize;  // not sure this is needed.
    int32_t column_index; // for ERROR_INVALID_COLUMN_INDEX;
} read_error_type;


PyArrayObject *
read_rows(stream *s,
        Py_ssize_t *nrows, int num_field_types, field_type *field_types,
        parser_config *pconfig, int num_usecols, int *usecols,
        Py_ssize_t skiplines, PyObject *converters,
        PyArrayObject *data_array, PyArray_Descr *out_descr,
        bool homogeneous);

#endif
