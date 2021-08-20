
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
#include "blocks.h"

#define INITIAL_BLOCKS_TABLE_LENGTH 200
#define ROWS_PER_BLOCK 500


/*
 * Defines liberated from NumPy's, only used for the PyArray_Pack hack!
 * TODO: Remove!
 */
#if PY_VERSION_HEX < 0x030900a4
    /* Introduced in https://github.com/python/cpython/commit/d2ec81a8c99796b51fb8c49b77a7fe369863226f */
    #define Py_SET_TYPE(obj, type) ((Py_TYPE(obj) = (type)), (void)0)
    /* Introduced in https://github.com/python/cpython/commit/c86a11221df7e37da389f9c6ce6e47ea22dc44ff */
    #define Py_SET_REFCNT(obj, refcnt) ((Py_REFCNT(obj) = (refcnt)), (void)0)
#endif


//
// If num_field_types is not 1, actual_num_fields must equal num_field_types.
//
size_t
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

size_t
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
size_t
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

void *
read_rows(stream *s,
        int *nrows, int num_field_types, field_type *field_types,
        parser_config *pconfig, int32_t *usecols, int num_usecols,
        int skiplines, PyObject *converters, void *data_array,
        int *num_cols, bool homogeneous, bool needs_init,
        read_error_type *read_error)
{
    char *data_ptr;
    int current_num_fields;
    size_t row_size;
    size_t size;
    PyObject **conv_funcs = NULL;

    bool track_string_size = false;

    bool use_blocks = false;
    blocks_data *blks = NULL;


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

    int row_count = 0;  /* number of rows actually processed */
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

        int j, k;

        if (NPY_UNLIKELY(actual_num_fields == -1)) {
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
            else {
                // Normalize the values in usecols.
                for (j = 0; j < num_usecols; ++j) {
                    if (usecols[j] < 0) {
                        usecols[j] += current_num_fields;
                    }
                    // XXX Check that the value is between 0 and current_num_fields.
                }
            }

            if (field_types_prepare_parsing(
                        num_field_types, field_types, pconfig) < 0) {
                return NULL;
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
                use_blocks = true;
                blks = blocks_init(row_size, ROWS_PER_BLOCK, INITIAL_BLOCKS_TABLE_LENGTH);
                if (blks == NULL) {
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
                data_ptr = data_array;
            }
        }

        if (track_string_size) {
            size_t new_itemsize;
            // typecode must be 'S' or 'U'.
            // Find the maximum field length in the current line.
            if (converters != Py_None) {
                // XXX Not handled yet.
            }
            size_t maxlen = max_token_len(fields, actual_num_fields,
                    usecols, num_usecols);
            new_itemsize = (field_types[0].typecode == 'S') ? maxlen : 4*maxlen;
            if (new_itemsize > field_types[0].itemsize) {
                // There is a field in this row whose length is
                // more than any previously seen length.
                if (use_blocks) {
                    int status = blocks_uniform_resize(blks, actual_num_fields, new_itemsize);
                    if (status != 0) {
                        // XXX Handle this--probably out of memory.
                    }
                }
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

        if (use_blocks) {
            data_ptr = blocks_get_row_ptr(blks, row_count, needs_init);
            if (data_ptr == NULL) {
                read_error->error_type = ERROR_OUT_OF_MEMORY;
                goto error;
            }
        }

        for (int j = 0; j < num_usecols; ++j) {
            int f = homogeneous ? 0 : j;
            // k is the column index of the field in the file.
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
                        str, end, data_ptr, field_types[f].userdata) < 0) {
                    err = ERROR_BAD_FIELD;
                }
            }
            else {
                /* TODO: This dual-use of to_generic is maybe not great */
                if (to_generic(field_types[f].descr,
                        str, end, data_ptr, conv_funcs[j]) < 0) {
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

    if (use_blocks) {
        if (read_error->error_type == 0) {
            // No error.
            // Copy the blocks into a newly allocated contiguous array.
            data_array = blocks_to_contiguous(blks, row_count);
        }
        blocks_destroy(blks);
    }

    //stream_close(s, RESTORE_FINAL);

    *nrows = row_count;

    /* TODO: Make sure no early return needs this cleanup (or use goto) */
    tokenizer_clear(&ts);

    if (read_error->error_type) {
        return NULL;
    }
    free(conv_funcs);
    return (void *) data_array;

  error:
    free(conv_funcs);
    tokenizer_clear(&ts);
    if (use_blocks) {
        blocks_destroy(blks);
    }
    return NULL;
}
