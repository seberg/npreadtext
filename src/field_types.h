
#ifndef _FIELD_TYPES_H_
#define _FIELD_TYPES_H_

#include <stdint.h>
#include <stdbool.h>
#define PY_ARRAY_UNIQUE_SYMBOL npreadtext_ARRAY_API
#define NO_IMPORT_ARRAY
#include "numpy/ndarraytypes.h"

#include "typedefs.h"
#include "parser_config.h"

/*
 * The original code had some error details, but I assume that we don't need
 * it.  Printing the string from which we tried to modify it should be fine.
 * This should potentially be public NumPy API, although it is tricky, NumPy
 *
 * This function must support unaligned memory access.
 *
 * TODO: An earlier version of the code had unused default versions (pandas
 *       does this) when columns are missing.  We could define this either
 *       by passing `NULL` in, or by adding a default explicitly somewhere.
 *       (I think users should probably have to define the default, at which
 *       point it doesn't matter here.)
 *
 * NOTE: We are currently passing the parser config, this could be made public
 *       or could be set up to be dtype specific/private.  Always passing
 *       pconfig fully seems easier right now.
 */
typedef int (set_from_ucs4_function)(
        PyArray_Descr *descr, const char32_t *str, const char32_t *end,
        char *dataptr, parser_config *pconfig);

typedef struct _field_type {
    set_from_ucs4_function *set_from_ucs4;
    /* The original NumPy descriptor */
    PyArray_Descr *descr;

    // itemsize:
    //   Size of field, in bytes.  In theory this would only be
    //   needed for the 'S' or 'U' type codes, but it is expected to be
    //   correctly filled in for all the types.
    size_t itemsize;
    // typecode used currently during discovery:
    //     * : undefined field type (during auto-discovery)
    //     i : integer (byte to long long)
    //     u : unsigned integer (ubyte to unsigned long long)
    //     f : floating point (or double) -- not float128, as it can't be
    //         handled by the builtin parser (we ignore it here).
    //     c : complex (64bit or 128 bit).
    //     S : character string (1 character == 1 byte)
    //     U : Unicode string (32 bit codepoints)
    //     x : generic dtype.
    char typecode;
} field_type;


void
field_types_clear(int num_field_types, field_type *ft);

field_type *
field_types_create(int num_field_types, PyArray_Descr *dtypes[]);

void
field_types_fprintf(FILE *out, int num_field_types, const field_type *ft);

bool
field_types_is_homogeneous(int num_field_types, const field_type *ft);

int32_t
field_types_total_size(int num_field_types, const field_type *ft);

int
field_types_grow(int new_num_field_types, int num_field_types, field_type **ft);

int
field_types_init_descriptors(int num_field_types, field_type *ft);

PyArray_Descr *
field_types_to_descr(int num_fields, field_type *ft);

#endif
