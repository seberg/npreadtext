
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


/*
 * Finalize the field descriptors, if the descriptors are already set, does
 * not replace them, otherwise creates them (from the field typecode which
 * must be set).
 *
 * TODO: We are assuming here sane systems with 1, 2, 4, and 8 byte sized
 *       integers.  But NumPy relies more on the C levels.
 */
static int
field_types_init_descriptors(int num_field_types, field_type *ft)
{
    for (int i = 0; i < num_field_types; i++) {
        int typenum = -1;
        /* Strings have adaptable length, so handle it specifically. */
        if (ft[i].typecode == 'S' || ft[i].typecode == 'U') {
            typenum = ft[i].typecode == 'S' ? NPY_STRING : NPY_UNICODE;
            if (ft[i].typecode == 'S') {
                ft[i].set_from_ucs4 = &to_string;
            }
            else {
                ft[i].set_from_ucs4 = &to_unicode;
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
                    ft[i].set_from_ucs4 = &to_int8;
                    break;
                case 2:
                    ft[i].set_from_ucs4 = &to_int16;
                    typenum = NPY_INT16;
                    break;
                case 4:
                    ft[i].set_from_ucs4 = &to_int32;
                    typenum = NPY_INT32;
                    break;
                case 8:
                    ft[i].set_from_ucs4 = &to_int64;
                    typenum = NPY_INT64;
                    break;
            }
        }
        else if (ft[i].typecode == 'u') {
            size_t itemsize = ft[i].itemsize;
            switch (itemsize) {
                case 1:
                    ft[i].set_from_ucs4 = &to_uint8;
                    typenum = NPY_UINT8;
                    break;
                case 2:
                    ft[i].set_from_ucs4 = &to_uint16;
                    typenum = NPY_UINT16;
                    break;
                case 4:
                    ft[i].set_from_ucs4 = &to_uint32;
                    typenum = NPY_UINT32;
                    break;
                case 8:
                    ft[i].set_from_ucs4 = &to_uint64;
                    typenum = NPY_UINT64;
                    break;
            }
        }
        else if (ft[i].typecode == 'f') {
            if (ft[i].itemsize == 8) {
                typenum = NPY_DOUBLE;
                ft[i].set_from_ucs4 = &to_double;
            }
            else {
                assert(ft[i].itemsize == 4 && ft[i].descr != NULL);
                ft[i].set_from_ucs4 = &to_float;
            }
        }
        else if (ft[i].typecode == 'c') {
            if (ft[i].itemsize == 16) {
                typenum = NPY_CDOUBLE;
                ft[i].set_from_ucs4 = &to_cdouble;
            }
            else {
                assert(ft[i].itemsize == 8 && ft[i].descr != NULL);
                ft[i].set_from_ucs4 = &to_cfloat;
            }
        }
        else if (ft[i].typecode == '*') {
            // TODO: does this happen?
            ft[i].set_from_ucs4 = &to_double;
            typenum = NPY_DOUBLE;  /* use the default */
        }
        else {
            assert(ft[i].descr != NULL);
            ft[i].set_from_ucs4 = &to_generic;
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
