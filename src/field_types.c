
//
// Pure C -- no Python API.
// With TESTMAIN defined, this file has a main() that runs some "tests"
// (actually it just prints some stuff, to be verified by the user).
//

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include "field_types.h"
#include "conversions.h"
#include "str_to_int.h"
#include "parser_config.h"

#define NO_IMPORT_ARRAY
#define PY_ARRAY_UNIQUE_SYMBOL npreadtext_ARRAY_API
#include <numpy/arrayobject.h>


void
field_types_clear(int num_field_types, field_type *ft) {
    for (int i = 0; i < num_field_types; i++) {
        Py_XDECREF(ft[i].descr);
        ft[i].descr = NULL;
    }
}

field_type *
field_types_create(int num_field_types, PyArray_Descr *dtypes[])
{
    field_type *ft;

    ft = malloc(num_field_types * sizeof(field_type));
    if (ft == NULL) {
        return NULL;
    }
    for (int i = 0; i < num_field_types; ++i) {
        PyArray_Descr *descr = dtypes[i];
        if (PyDataType_ISSIGNED(descr)) {
            ft[i].typecode = 'i';
        }
        else if (PyDataType_ISUNSIGNED(descr)) {
            ft[i].typecode = 'u';
        }
        else if (PyDataType_ISFLOAT(descr) && descr->elsize <= 8) {
            ft[i].typecode = 'f';
        }
        else if (PyDataType_ISCOMPLEX(descr) && descr->elsize <= 16) {
            ft[i].typecode = 'c';
        }
        else if (descr->type_num == NPY_STRING) {
            ft[i].typecode = 'S';
        }
        else if (descr->type_num == NPY_UNICODE) {
            ft[i].typecode = 'U';
        }
        else {
            ft[i].typecode = 'x';
        }
        ft[i].itemsize = descr->elsize;
        Py_INCREF(descr);
        ft[i].descr = descr;
    }
    if (field_types_init_descriptors(num_field_types, ft) < 0) {
        free(ft);
        return NULL;
    }
    return ft;
}


/*
 * Fills in the parser config data (because we did not do that earlier)
 * TODO: Should likely be consolidated (was split for converters but that
 *       turned out not a great idea).
 */
int
field_types_prepare_parsing(int num, field_type *ft, parser_config *pconfig)
{
    for (int i = 0; i < num; ++i) {
        /* A converter replace the default parsing! */
        if (ft[i].typecode == 'f' || ft[i].typecode == 'c'
                 || ft[i].typecode == 'i' || ft[i].typecode == 'u') {
            /* also integers fall back to float (unclear if useful) */
            ft[i].userdata = pconfig;
        }
    }
    return 0;
}


void
field_types_fprintf(FILE *out, int num_field_types, const field_type *ft)
{
    for (int i = 0; i < num_field_types; ++i) {
        fprintf(out, "ft[%d].typecode = %c, .itemsize = %lu\n",
                     i, ft[i].typecode, (unsigned long)ft[i].itemsize);
    }
}


bool
field_types_is_homogeneous(int num_field_types, const field_type *ft)
{
    bool homogeneous = true;
    for (int k = 1; k < num_field_types; ++k) {
        if ((ft[k].typecode != ft[0].typecode) ||
                (ft[k].itemsize != ft[0].itemsize)) {
            homogeneous = false;
            break;
        }
    }
    return homogeneous;
}


int32_t
field_types_total_size(int num_field_types, const field_type *ft)
{
    int32_t size = 0;
    for (int k = 0; k < num_field_types; ++k) {
        size += ft[k].itemsize;
    }
    return size;
}


//
// *ft must be a pointer to field_type that is either NULL or
// previously malloc'ed.
// The function assumes that new_num_field_types > num_field_types.
//
int
field_types_grow(int new_num_field_types, int num_field_types, field_type **ft)
{
    size_t nbytes;
    field_type *new_ft;

    nbytes = new_num_field_types * sizeof(field_type);
    new_ft = (field_type *) realloc(*ft, nbytes);
    if (new_ft == NULL) {
        free(*ft);
        *ft = NULL;
        return -1;
    }
    for (int k = num_field_types; k < new_num_field_types; ++k) {
        new_ft[k].typecode = '*';
        new_ft[k].itemsize = 0;
        new_ft[k].descr = NULL;
    }
    *ft = new_ft;

    return 0;
}


/*
 * Finalize the field descriptors, if the descriptors are already set, does
 * not replace them, otherwise creates them (from the field typecode which
 * must be set).
 *
 * TODO: We are assuming here sane systems with 1, 2, 4, and 8 byte sized
 *       integers.  But NumPy relies more on the C levels.
 */
int
field_types_init_descriptors(int num_field_types, field_type *ft)
{
    for (int i = 0; i < num_field_types; i++) {
        int typenum = -1;
        ft[i].userdata = NULL;  // Most don't have any data.

        /* Strings have adaptable length, so handle it specifically. */
        if (ft[i].typecode == 'S' || ft[i].typecode == 'U') {
            typenum = ft[i].typecode == 'S' ? NPY_STRING : NPY_UNICODE;
            if (ft[i].typecode == 'S') {
                ft[i].set_from_ucs4 = (set_from_ucs4_function *)&to_string;
            }
            else {
                ft[i].set_from_ucs4 = (set_from_ucs4_function *)&to_unicode;
            }
            if (ft[i].descr == NULL) {
                ft[i].descr = PyArray_DescrNewFromType(typenum);
                if (ft[i].descr == NULL) {
                    field_types_clear(num_field_types, ft);
                    return -1;
                }
                ft[i].descr->elsize = ft[i].itemsize;
            }
            continue;
        }

        if (ft[i].typecode == 'i') {
            size_t itemsize = ft[i].itemsize;
            switch (itemsize) {
                case 1:
                    typenum = NPY_INT8;
                    ft[i].set_from_ucs4 = (set_from_ucs4_function *)&to_int8;
                    break;
                case 2:
                    ft[i].set_from_ucs4 = (set_from_ucs4_function *)&to_int16;
                    typenum = NPY_INT16;
                    break;
                case 4:
                    ft[i].set_from_ucs4 = (set_from_ucs4_function *)&to_int32;
                    typenum = NPY_INT32;
                    break;
                case 8:
                    ft[i].set_from_ucs4 = (set_from_ucs4_function *)&to_int64;
                    typenum = NPY_INT64;
                    break;
            }
        }
        else if (ft[i].typecode == 'u') {
            size_t itemsize = ft[i].itemsize;
            switch (itemsize) {
                case 1:
                    ft[i].set_from_ucs4 = (set_from_ucs4_function *)&to_uint8;
                    typenum = NPY_UINT8;
                    break;
                case 2:
                    ft[i].set_from_ucs4 = (set_from_ucs4_function *)&to_uint16;
                    typenum = NPY_UINT16;
                    break;
                case 4:
                    ft[i].set_from_ucs4 = (set_from_ucs4_function *)&to_uint32;
                    typenum = NPY_UINT32;
                    break;
                case 8:
                    ft[i].set_from_ucs4 = (set_from_ucs4_function *)&to_uint64;
                    typenum = NPY_UINT64;
                    break;
            }
        }
        else if (ft[i].typecode == 'f') {
            if (ft[i].itemsize == 8) {
                typenum = NPY_DOUBLE;
                ft[i].set_from_ucs4 = (set_from_ucs4_function *)&to_double;
            }
            else {
                assert(ft[i].itemsize == 4 && ft[i].descr != NULL);
                ft[i].set_from_ucs4 = (set_from_ucs4_function *)&to_float;
            }
        }
        else if (ft[i].typecode == 'c') {
            if (ft[i].itemsize == 16) {
                typenum = NPY_CDOUBLE;
                ft[i].set_from_ucs4 = (set_from_ucs4_function *)&to_cdouble;
            }
            else {
                assert(ft[i].itemsize == 8 && ft[i].descr != NULL);
                ft[i].set_from_ucs4 = (set_from_ucs4_function *)&to_cfloat;
            }
        }
        else if (ft[i].typecode == '*') {
            // TODO: does this happen?
            ft[i].set_from_ucs4 = (set_from_ucs4_function *)&to_double;
            typenum = NPY_DOUBLE;  /* use the default */
        }
        else {
            assert(ft[i].descr != NULL);
            ft[i].set_from_ucs4 = (set_from_ucs4_function *)&to_generic;
        }

        if (ft[i].descr == NULL) {
            ft[i].descr = PyArray_DescrFromType(typenum);
            if (ft[i].descr == NULL) {
                /* can't actually fail, but... */
                field_types_clear(num_field_types, ft);
                return -1;
            }
        }
    }
    return 0;
}


PyArray_Descr *
field_types_to_descr(int num_fields, field_type *ft)
{
    PyArray_Descr *result = NULL;
    PyObject *dtype_list = NULL, *empty_string = NULL;

    empty_string = PyUnicode_FromString("");
    if (empty_string == NULL) {
        goto finish;
    }
    dtype_list = PyList_New(0);
    if (dtype_list == NULL) {
        goto finish;
    }

    for (int i = 0; i < num_fields; i++) {
        PyObject *tup = PyTuple_Pack(2, empty_string, ft[i].descr);
        if (tup == NULL) {
            goto finish;
        }
        int res = PyList_Append(dtype_list, tup);
        Py_DECREF(tup);
        if (res < 0) {
            goto finish;
        }
    }
    PyArray_DescrConverter(dtype_list, &result);

  finish:
    Py_XDECREF(empty_string);
    Py_XDECREF(dtype_list);
    return result;
}

#ifdef TESTMAIN

void
show_field_types(int num_fields, field_type *ft)
{
    field_types_fprintf(stdout, num_fields, ft);

    bool homogeneous = field_types_is_homogeneous(num_fields, ft);
    printf("homogeneous = %s\n", homogeneous ? "true" : "false");

    int32_t total_size = field_types_total_size(num_fields, ft);
    printf("total_size = %lu\n", (unsigned long)total_size);

    char *dtypestr = field_types_build_str(num_fields, NULL, homogeneous, ft);
    printf("dtypestr = '%s'\n", dtypestr);

    free(dtypestr);
}


int
main(int argc, char *argv[])
{
    char *codes = "ffHHSU";
    int32_t sizes[] = {8, 8, 2, 2, 4, 48};
    int num_fields = sizeof(sizes) / sizeof(sizes[0]);

    field_type *ft = field_types_create(num_fields, codes, sizes);
    printf("ft:\n");
    show_field_types(num_fields, ft);

    int status = field_types_grow(num_fields + 2, num_fields, &ft);
    ft[num_fields].typecode = 'b';
    ft[num_fields].itemsize = 1;
    ++num_fields;
    ft[num_fields].typecode = 'f';
    ft[num_fields].itemsize = 4;
    ++num_fields;

    printf("\n");
    printf("ft, after grow:\n");
    show_field_types(num_fields, ft);

    free(ft);

    return 0;
}

#endif
