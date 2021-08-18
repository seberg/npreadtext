
#define _XOPEN_SOURCE 700

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <stdint.h>

#include "typedefs.h"
#include "tokenize.h"
#include "field_types.h"
#include "analyze.h"
#include "type_inference.h"
#include "stream.h"
#include "char32utils.h"


typedef struct {

    // For integer types, the lower bound of the values.
    int64_t imin;

    // For integer types, the upper bound of the values.
    uint64_t umax;

} integer_range;


int
enlarge_ranges(int new_num_fields, int num_fields, integer_range **ranges)
{
    int nbytes;
    integer_range *new_ranges;

    nbytes = new_num_fields * sizeof(integer_range);
    new_ranges = (integer_range *) realloc(*ranges, nbytes);
    if (new_ranges == NULL) {
        free(*ranges);
        *ranges = NULL;
        return -1;
    }
    for (int k = num_fields; k < new_num_fields; ++k) {
        new_ranges[k].imin = 0;
        new_ranges[k].umax = 0;
    }
    *ranges = new_ranges;
    return 0;
}

int
enlarge_type_tracking_arrays(
        int new_num_fields, int num_fields,
        field_type **types, integer_range **ranges)
{
    int status1 = field_types_grow(new_num_fields, num_fields, types);
    int status2 = enlarge_ranges(new_num_fields, num_fields, ranges);
    if (status1 != 0 || status2 != 0) {
        return -1;
    }
    return 0;
}

/*
 *  Parameters
 *  ----------
 *  ...
 *  skiplines : int
 *      Number of text lines to skip before beginning to analyze the rows.
 *  numrows : int
 *      maximum number of rows to analyze (currently not implemented)
 *
 *  Return value
 *  ------------
 *  row_count > 0: number of rows. row_count might be less than numrows if the
 *      end of the file is reached.
 *  ANALYZE_FILE_ERROR:    unable to create a file buffer.
 *  ANALYZE_OUT_OF_MEMORY: out of memory (malloc failed)
 *  ...
 */

int
analyze(stream *s, parser_config *pconfig, int skiplines, int numrows,
        int *p_num_fields, field_type **p_field_types)
{
    int row_count = 0;
    int num_fields = 0;

    int ts_result = 0;
    tokenizer_state ts;
    tokenizer_init(&ts, pconfig);

    field_type *types = NULL;
    integer_range *ranges = NULL;

    char32_t decimal = pconfig->decimal;
    char32_t sci = pconfig->sci;
    char32_t imaginary_unit = pconfig->imaginary_unit;

    for (; skiplines > 0; skiplines--) {
        ts.state = TOKENIZE_GOTO_LINE_END;
        ts_result = tokenize(s, &ts, pconfig);
        if (ts_result < 0) {
            return -1;
        }
        else if (ts_result != 0) {
            /* Fewer lines than skiplines is acceptable */
            break;
        }
    }

    // In this loop, types[k].itemsize will track the largest field length
    // encountered for field k, regardless of the apparent type of the field
    // (since we don't know the field type until we've seen the whole
    // file).
    // ranges[k].imin tracks the most negative integer seen in the field.
    // (If all the values are positive, ranges[k].imin will be 0.)
    // ranges[k].umax tracks the largest positive integer seen in the field.
    // (If all the values are negative, ranges[k].umax will be 0.)
    while (row_count != numrows && ts_result == 0) {
        size_t new_num_fields;

        ts_result = tokenize(s, &ts, pconfig);
        if (ts_result < 0) {
            return -1;
        }
        if (ts.num_fields == 0) {
            continue;  /* does not add a row */
        }
        new_num_fields = ts.num_fields;
        field_info *fields = ts.fields;

        if (new_num_fields > num_fields) {
            // The first row, or a row with more fields than previously seen...
            int status = enlarge_type_tracking_arrays(new_num_fields, num_fields,
                                                      &types, &ranges);
            if (status != 0) {
                // If this occurs, types and ranges have been freed in
                // enlarge_type_tracking_array().
                return ANALYZE_OUT_OF_MEMORY;
            }
            num_fields = new_num_fields;
        }

        for (int k = 0; k < new_num_fields; ++k) {
            char typecode;
            int64_t imin;
            uint64_t umax;

            char32_t *token = ts.field_buffer + fields[k].offset;
            typecode = classify_type(token, decimal, sci, imaginary_unit,
                                     &imin, &umax,
                                     types[k].typecode);
            if (typecode == 'i' && imin < ranges[k].imin) {
                ranges[k].imin = imin;
            }
            if ((typecode == 'i' || typecode == 'u') && umax > ranges[k].umax) {
                ranges[k].umax = umax;
            }
            if (typecode != '*') {
                types[k].typecode = typecode;
            }
            if (fields[k].length > types[k].itemsize) {
                types[k].itemsize = fields[k].length;
            }
        }
        ++row_count;
    }

    // At this point, any field that contained only unsigned integers
    // or only integers (some negative) has been classified as typecode='Q'
    // or typecode='q', respectively.  Now use the integer ranges that were
    // found to refine these classifications.
    // We also "fix" the itemsize field for any non-string fields (even though
    // the itemsize of non-strings is implicit in the typecode).

    for (int k = 0; k < num_fields; ++k) {
        char typecode = types[k].typecode;
        if (typecode == 'i' || typecode == 'u') {
            // Integer type.  Use imin and umax to refine the type.
            update_type_for_integer_range(
                    &types[k].typecode, &types[k].itemsize,
                    ranges[k].imin, ranges[k].umax);
        }
        /*
         * Fix the itemsize for the remaining "types", note that if the user
         * should be able to specify using float instead of double, these
         * would have to be modified.
         */
        switch (types[k].typecode) {
            case 'f':
                types[k].itemsize = 8;
                break;
            case 'c':
                types[k].itemsize = 16;
                break;
            case 'U':
                types[k].itemsize = 8;
                break;
            default:
                assert(types[k].itemsize != 0 || types[k].typecode == '*');
        }
    }

    free(ranges);

    *p_num_fields = num_fields;
    *p_field_types = types;

    //fb_del(fb, RESTORE_INITIAL);
    /* TODO: Also clean up on failure! */
    tokenizer_clear(&ts);
    return row_count;
}
