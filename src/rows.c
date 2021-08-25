
#define _XOPEN_SOURCE 700

#define PY_SSIZE_T_CLEAN
#include <Python.h>

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#define NO_IMPORT_ARRAY
#define PY_ARRAY_UNIQUE_SYMBOL npreadtext_ARRAY_API
#include "numpy/arrayobject.h"

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <stdbool.h>
#include <complex.h>

#include "stream.h"
#include "tokenize.h"
#include "char32utils.h"
#include "conversions.h"
#include "field_types.h"
#include "rows.h"
#include "error_types.h"
#include "str_to.h"
#include "str_to_int.h"

/* Minimum number of rows to grow, must be a power of two */
#define ROWS_PER_BLOCK 512


//
// If num_field_types is not 1, actual_num_fields must equal num_field_types.
//
static size_t
compute_row_size(
        int actual_num_fields, int num_field_types, field_type *field_types)
{
    size_t row_size;

    // rowsize is the number of bytes in each "row" of the array
    // filled in by this function.
    if (num_field_types == 1) {
        row_size = actual_num_fields * field_types[0].itemsize;
    }
    else {
        row_size = 0;
        for (int k = 0; k < num_field_types; ++k) {
            row_size += field_types[k].itemsize;
        }
    }
    return row_size;
}


/*
 *  Find the length of the longest token.
 */

static size_t
max_token_len(
        field_info *fields, int num_tokens, int32_t *usecols, int num_usecols)
{
    size_t maxlen = 0;
    for (int i = 0; i < num_tokens; ++i) {
        size_t j;
        if (usecols == NULL) {
            j = i;
        }
        else {
            j = usecols[i];
        }
        size_t m = fields[j+1].offset - fields[j].offset - 1;
        if (m > maxlen) {
            maxlen = m;
        }
    }
    return maxlen;
}


// WIP...
static size_t
max_token_len_with_converters(
        char32_t **tokens, int num_tokens, int32_t *usecols,
        int num_usecols, PyObject **conv_funcs)
{
    size_t maxlen = 0;
    size_t m;

    for (int i = 0; i < num_tokens; ++i) {
        size_t j;
        if (usecols == NULL) {
            j = i;
        }
        else {
            j = usecols[i];
        }

        if (conv_funcs && conv_funcs[j]) {
            // TODO: Moved converter calling function out of here.
            PyObject *obj = NULL; //call_converter_function(conv_funcs[i], tokens[j]);
            if (obj == NULL) {
                fprintf(stderr, "CALL FAILED!\n");
            }
            // XXX check for obj == NULL!
            PyObject *s = PyObject_Str(obj);
            if (s == NULL) {
                fprintf(stderr, "STR FAILED!\n");
            }
            Py_DECREF(obj);
            // XXX check for s == NULL!
            Py_ssize_t len = PySequence_Length(s);
            if (len == -1) {
                fprintf(stderr, "LEN FAILED!\n");
            }
            // XXX check for len == -1
            Py_DECREF(s);
            m = (size_t) len;
        }
        else {
            m = strlen32(tokens[j]);
        }
        if (m > maxlen) {
            maxlen = m;
        }
    }
    return maxlen;
}


/*
 * When resizing strings, we need to allocate a new area and then copy
 * all strings with zero padding.  (Currently, this function uses calloc
 * rather than manual zero padding.)
 */
static int
expand_string_data_and_copy(
        size_t old_itemsize, size_t new_itemsize,
        size_t allocated_rows, size_t num_fields,
        char **data_array, char **data_ptr)
{
    size_t new_num_elements = allocated_rows * num_fields;
    char *new_data = calloc(new_num_elements, new_itemsize);
    if (new_data == NULL) {
        PyErr_NoMemory();
        return -1;
    }
    char *orig_ptr = *data_array;
    char *new_ptr = new_data;
    while (orig_ptr < *data_ptr) {
        memcpy(new_ptr, orig_ptr, old_itemsize);
        new_ptr += new_itemsize;
        orig_ptr += old_itemsize;
    }
    *data_array = new_data;
    *data_ptr = new_ptr;
    return 0;
}


/*
 *  Create the array of converter functions from the Python converters dict.
 */
PyObject **
create_conv_funcs(
        PyObject *converters, int32_t *usecols, int num_usecols,
        int current_num_fields, read_error_type *read_error)
{
    PyObject **conv_funcs = NULL;

    conv_funcs = calloc(num_usecols, sizeof(PyObject *));
    if (conv_funcs == NULL) {
        read_error->error_type = ERROR_OUT_OF_MEMORY;
        return NULL;
    }
    for (int j = 0; j < num_usecols; ++j) {
        PyObject *key;
        PyObject *func;
        // k is the column index of the field in the file.
        size_t k;
        if (usecols == NULL) {
            k = j;
        }
        else {
            k = usecols[j];
        }

        // XXX Check for failure of PyLong_FromSsize_t...
        key = PyLong_FromSsize_t((Py_ssize_t) k);
        func = PyDict_GetItem(converters, key);
        Py_DECREF(key);
        if (func == NULL) {
            key = PyLong_FromSsize_t((Py_ssize_t) k - current_num_fields);
            func = PyDict_GetItem(converters, key);
            Py_DECREF(key);
        }
        if (func != NULL) {
            Py_INCREF(func);
            conv_funcs[j] = func;
        }
    }
    return conv_funcs;
}

/*
 *  XXX Handle errors in any of the functions called by read_rows().
 *
 *  XXX Check handling of *nrows == 0.
 *
 *  Parameters
 *  ----------
 *  stream *s
 *  int *nrows
 *      On input, *nrows is the maximum number of rows to read.
 *      If *nrows is positive, `data_array` must point to the block of data
 *      where the data is to be stored.
 *      If *nrows is negative, all the rows in the file should be read, and
 *      the given value of `data_array` is ignored.  Data will be allocated
 *      dynamically in this function.
 *      On return, *nrows is the number of rows actually read from the file.
 *  int num_field_types
 *      Number of field types (i.e. the number of fields).  This is the
 *      length of the array pointed to by field_types.
 *  field_type *field_types
 *  parser_config *pconfig
 *  int32_t *usecols
 *      Pointer to array of column indices to use.
 *      If NULL, use all the columns (and ignore `num_usecols`).
 *  int num_usecols
 *      Length of the array `usecols`.  Ignored if `usecols` is NULL.
 *  int skiplines
 *  PyObject *converters
 *      dicitionary of converters
 *  void *data_array
 *  int *num_cols
 *      The actual number of columns (or fields) of the data being returned.
 *  read_error_type *read_error
 *      Information about errors detected in read_rows()
 */
char *
read_rows(stream *s,
        int *nrows, int num_field_types, field_type *field_types,
        parser_config *pconfig, int32_t *usecols, int num_usecols,
        int skiplines, PyObject *converters, char *data_array,
        int *num_cols, bool homogeneous, bool needs_init,
        read_error_type *read_error)
{
    char *data_ptr = NULL;
    int current_num_fields;
    size_t row_size;
    size_t size;
    PyObject **conv_funcs = NULL;

    bool track_string_size = false;

    bool data_array_allocated = data_array == NULL;
    size_t data_allocated_rows = 0;

    int ts_result = 0;
    tokenizer_state ts;
    if (tokenizer_init(&ts, pconfig) < 0) {
        return NULL;
    }

    int actual_num_fields = -1;

    read_error->error_type = 0;

    for (; skiplines > 0; skiplines--) {
        ts.state = TOKENIZE_GOTO_LINE_END;
        ts_result = tokenize(s, &ts, pconfig);
        if (ts_result < 0) {
            return NULL;
        }
        else if (ts_result != 0) {
            /* Fewer lines than skiplines is acceptable */
            break;
        }
    }

    // track_string_size will be true if the user passes in
    // dtype=np.dtype('S') or dtype=np.dtype('U').  That is, the
    // data type is string or unicode, but a length was not given.
    // In this case, we must track the maximum length of the fields
    // and update the actual length for the dtype dynamically as
    // the file is read.
    track_string_size = ((num_field_types == 1) &&
                         (field_types[0].itemsize == 0) &&
                         ((field_types[0].typecode == 'S') ||
                          (field_types[0].typecode == 'U')));

    size_t row_count = 0;  /* number of rows actually processed */
    while ((*nrows < 0 || row_count < *nrows) && ts_result == 0) {
        ts_result = tokenize(s, &ts, pconfig);
        if (ts_result < 0) {
            return NULL;
        }
        current_num_fields = ts.num_fields;
        field_info *fields = ts.fields;
        if (ts.num_fields == 0) {
            continue;  /* Ignore empty line */
        }

        if (NPY_UNLIKELY(data_ptr == NULL)) {
            // We've deferred some of the initialization tasks to here,
            // because we've now read the first line, and we definitively
            // know how many fields (i.e. columns) we will be processing.
            if (!homogeneous) {
                actual_num_fields = num_field_types;
            }
            else if (usecols != NULL) {
                actual_num_fields = num_usecols;
            }
            else {
                // num_field_types is 1.  (XXX Check that it can't be 0 or neg.)
                // Set actual_num_fields to the number of fields found in the
                // first line of data.
                actual_num_fields = current_num_fields;
            }

            if (usecols == NULL) {
                num_usecols = actual_num_fields;
            }

            if (converters != Py_None) {
                conv_funcs = create_conv_funcs(converters, usecols, num_usecols,
                                               current_num_fields, read_error);
            }
            else {
                conv_funcs = calloc(num_usecols, sizeof(PyObject *));
            }
            if (conv_funcs == NULL) {
                return NULL;
            }

            *num_cols = actual_num_fields;
            row_size = compute_row_size(actual_num_fields,
                                        num_field_types, field_types);

            if (*nrows < 0) {
                // Any negative value means "read the entire file".
                // In this case, it is assumed that *data_array is NULL
                // or not initialized. I.e. the value passed in is ignored,
                // and instead is initialized to the first block.
                data_allocated_rows = ROWS_PER_BLOCK;
                if (!needs_init) {
                    data_array = malloc(data_allocated_rows * row_size);
                }
                else {
                    data_array = calloc(data_allocated_rows * row_size, 1);
                }
                if (data_array == NULL) {
                    // XXX Check for other clean up that might be necessary.
                    read_error->error_type = ERROR_OUT_OF_MEMORY;
                    goto error;
                }
            }
            else {
                // *nrows >= 0
                // FIXME: Ensure that *nrows == 0 is handled correctly.
                if (data_array == NULL) {
                    // The number of rows to read was given, but a memory buffer
                    // was not, so allocate one here.
                    size = *nrows * row_size;
                    // TODO: this is wrong, it can never be freed, do we need this?
                    data_array = malloc(size);
                    if (data_array == NULL) {
                        read_error->error_type = ERROR_OUT_OF_MEMORY;
                        goto error;
                    }
                }
                data_allocated_rows = *nrows;
            }
            data_ptr = data_array;
        }

        if (track_string_size) {
            // typecode must be 'S' or 'U'.
            // Find the maximum field length in the current line.
            if (converters != Py_None) {
                // XXX Not handled yet.
            }
            size_t maxlen = max_token_len(fields, actual_num_fields,
                    usecols, num_usecols);
            size_t new_itemsize = (field_types[0].typecode == 'S') ? maxlen : 4*maxlen;

            if (new_itemsize > field_types[0].itemsize) {
                if (expand_string_data_and_copy(
                        field_types[0].itemsize, new_itemsize,
                        data_allocated_rows, actual_num_fields,
                        &data_array, &data_ptr) < 0) {
                    goto error;
                }
                row_size = new_itemsize * actual_num_fields;
                field_types[0].itemsize = new_itemsize;
                field_types[0].descr->elsize = new_itemsize;
            }
        }

        if (!usecols && (actual_num_fields != current_num_fields)) {
            read_error->error_type = ERROR_CHANGED_NUMBER_OF_FIELDS;
            read_error->line_number = row_count + 1;
            read_error->column_index = current_num_fields;
            goto error;
        }

        if (NPY_UNLIKELY(data_allocated_rows == row_count)) {
            /*
             * Grow by ~25% and rounded up to the next ROWS_PER_BLOCK
             * NOTE: This is based on very crude timings and could be refined!
             */
            size_t growth = (data_allocated_rows >> 2) + ROWS_PER_BLOCK;
            growth &= ~(size_t)(ROWS_PER_BLOCK-1);
            size_t new_rows = data_allocated_rows + growth;

            char *new_arr = realloc(data_array, new_rows * row_size);
            if (new_arr == NULL) {
                read_error->error_type = ERROR_OUT_OF_MEMORY;
                goto error;
            }
            if (new_arr != data_array) {
                data_array = new_arr;
                data_ptr = new_arr + data_allocated_rows * row_size;
            }
            data_allocated_rows = new_rows;
            if (needs_init) {
                memset(data_ptr, '\0', growth * row_size);
            }
        }

        for (int j = 0; j < num_usecols; ++j) {
            int f = homogeneous ? 0 : j;
            // k is the column index of the field in the file.
            int k;
            if (usecols == NULL) {
                k = j;
            }
            else {
                k = usecols[j];
                if (k < 0) {
                    // Python-like column indexing: k = -1 means the last column.
                    k += current_num_fields;
                }
                if (NPY_UNLIKELY((k < 0) || (k >= current_num_fields))) {
                    read_error->error_type = ERROR_INVALID_COLUMN_INDEX;
                    read_error->line_number = row_count - 1;
                    read_error->column_index = usecols[j];
                    goto error;
                }
            }

            if (NPY_UNLIKELY(k >= current_num_fields)) {
                PyErr_SetString(PyExc_NotImplementedError,
                        "internal error, k >= current_num_fields should not "
                        "be possible (and is note implemented)!");
                goto error;
            }

            int err = 0;
            char32_t *str = ts.field_buffer + fields[k].offset;
            char32_t *end = ts.field_buffer + fields[k + 1].offset - 1;
            if (conv_funcs[j] == NULL) {
                if (field_types[f].set_from_ucs4(field_types[f].descr,
                        str, end, data_ptr, pconfig) < 0) {
                    err = ERROR_BAD_FIELD;
                }
            }
            else {
                if (to_generic_with_converter(field_types[f].descr,
                        str, end, data_ptr, pconfig, conv_funcs[j]) < 0) {
                    err = ERROR_BAD_FIELD;
                }
            }
            data_ptr += field_types[f].itemsize;

            if (NPY_UNLIKELY(err)) {
                read_error->error_type = err;
                read_error->line_number = row_count - 1;
                read_error->field_number = k;
                read_error->char_position = -1; // FIXME
                read_error->descr = field_types[f].descr;
                goto error;
            }
        }

        ++row_count;
    }

    tokenizer_clear(&ts);
    free(conv_funcs);

    if (data_array_allocated && data_allocated_rows != row_count) {
        char *new_data = realloc(data_array, row_size * row_count);
        if (new_data == NULL) {
            free(data_array);
            PyErr_NoMemory();
            return NULL;
        }
        data_array = new_data;
    }

    //stream_close(s, RESTORE_FINAL);

    *nrows = row_count;

    if (read_error->error_type) {
        return NULL;
    }
    return data_array;

  error:
    free(conv_funcs);
    tokenizer_clear(&ts);
    if (data_array_allocated) {
        free(data_array);
    }
    return NULL;
}
