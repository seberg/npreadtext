#ifndef STR_TO_H
#define STR_TO_H

#include <stdio.h>
#include <stdint.h>
#include <stdbool.h>
#include <string.h>
#include <ctype.h>

#include "str_to.h"


#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#define NO_IMPORT_ARRAY
#define PY_ARRAY_UNIQUE_SYMBOL npreadtext_ARRAY_API
#include "numpy/ndarraytypes.h"


/*
 * The following two string conversion functions are largely equivalent
 * in Pandas.  They are in the header file here, to ensure they can be easily
 * inline in the other function.
 */

/*
 *  On success, *error is zero.
 *  If the conversion fails, *error is nonzero, and the return value is 0.
 */
static NPY_INLINE int
str_to_int64(
        const Py_UCS4 *p_item, const Py_UCS4 *p_end,
        int64_t int_min, int64_t int_max, int64_t *result)
{
    const Py_UCS4 *p = (const Py_UCS4 *) p_item;
    bool isneg = 0;
    int64_t number = 0;
    int d;

    // Skip leading spaces.
    while (Py_UNICODE_ISSPACE(*p)) {
        ++p;
    }

    // Handle sign.
    if (*p == '-') {
        isneg = true;
        ++p;
    }
    else if (*p == '+') {
        p++;
    }

    // Check that there is a first digit.
    if (!isdigit(*p)) {
        return -1;
    }

    if (isneg) {
        // If number is greater than pre_min, at least one more digit
        // can be processed without overflowing.
        int dig_pre_min = -(int_min % 10);
        int64_t pre_min = int_min / 10;

        // Process the digits.
        d = *p;
        while (isdigit(d)) {
            if ((number > pre_min) || ((number == pre_min) && (d - '0' <= dig_pre_min))) {
                number = number * 10 - (d - '0');
                d = *++p;
            }
            else {
                return -1;
            }
        }
    }
    else {
        // If number is less than pre_max, at least one more digit
        // can be processed without overflowing.
        int64_t pre_max = int_max / 10;
        int dig_pre_max = int_max % 10;

        //printf("pre_max = %lld  dig_pre_max = %d\n", pre_max, dig_pre_max);

        // Process the digits.
        d = *p;
        while (isdigit(d)) {
            if ((number < pre_max) || ((number == pre_max) && (d - '0' <= dig_pre_max))) {
                number = number * 10 + (d - '0');
                d = *++p;
            }
            else {
                return -1;
            }
        }
    }

    // Skip trailing spaces.
    while (Py_UNICODE_ISSPACE(*p)) {
        ++p;
    }

    // Did we use up all the characters?
    if (p != p_end) {
        return -1;
    }

    *result = number;
    return 0;
}

/*
 *  On success, *error is zero.
 *  If the conversion fails, *error is nonzero, and the return value is 0.
 */
static NPY_INLINE int
str_to_uint64(
        const Py_UCS4 *p_item, const Py_UCS4 *p_end,
        uint64_t uint_max, uint64_t *result)
{
    const Py_UCS4 *p = (const Py_UCS4 *) p_item;
    uint64_t number = 0;
    int d;

    // Skip leading spaces.
    while (Py_UNICODE_ISSPACE(*p)) {
        ++p;
    }

    // Handle sign.
    if (*p == '-') {
        return -1;
    }
    if (*p == '+') {
        p++;
    }

    // Check that there is a first digit.
    if (!isdigit(*p)) {
        return -1;
    }

    // If number is less than pre_max, at least one more digit
    // can be processed without overflowing.
    uint64_t pre_max = uint_max / 10;
    int dig_pre_max = uint_max % 10;

    // Process the digits.
    d = *p;
    while (isdigit(d)) {
        if ((number < pre_max) || ((number == pre_max) && (d - '0' <= dig_pre_max))) {
            number = number * 10 + (d - '0');
            d = *++p;
        }
        else {
            return -1;
        }
    }

    // Skip trailing spaces.
    while (Py_UNICODE_ISSPACE(*p)) {
        ++p;
    }

    // Did we use up all the characters?
    if (p != p_end) {
        return -1;
    }

    *result = number;
    return 0;
}


#endif
