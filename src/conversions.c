
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <errno.h>
#include <ctype.h>
#include <stdbool.h>
#include <complex.h>

#include "typedefs.h"
#include "char32utils.h"
#include "conversions.h"

double
_Py_dg_strtod_modified(
        const char32_t *s00, char32_t **se, int *error,
        char32_t decimal, char32_t sci, bool skip_trailing);

/*
 *  `item` must be the nul-terminated string that is to be
 *  converted to a double.
 *
 *  To be successful, to_double() must use *all* the characters
 *  in `item`.  E.g. "1.q25" will fail.  Leading and trailing 
 *  spaces are allowed.
 *
 *  `sci` is the scientific notation exponent character, usually
 *  either 'E' or 'D'.  Case is ignored.
 *
 *  `decimal` is the decimal point character, usually either
 *  '.' or ','.
 *
 */
int
to_float(PyArray_Descr *descr,
        const char32_t *str, const char32_t *end, char *dataptr,
        parser_config *pconfig)
{
    if (*str == '\0') {
        /* can't parse NUL, this field is probably just empty. */
        return -1;
    }

    int error;
    char32_t *p_end;

    float val = (float)_Py_dg_strtod_modified(
            str, &p_end, &error, pconfig->decimal, pconfig->sci, true);

    if (error != 0 || p_end != end) {
        return -1;
    }
    memcpy(dataptr, &val, sizeof(float));
    if (!PyArray_ISNBO(descr->byteorder)) {
        descr->f->copyswap(dataptr, dataptr, 1, NULL);
    }
    return 0;
}


/* TODO: Used only for the type inference (which glances over the bugs though) */
bool
to_double_raw(const char32_t *str, double *res, char32_t decimal, char32_t sci)
{
    if (*str == '\0') {
        return false;
    }

    int error;
    char32_t *p_end;
    *res = _Py_dg_strtod_modified(
            str, &p_end, &error, decimal, sci, true);
    return (error == 0) && (!*p_end);
}


int
to_double(PyArray_Descr *descr,
        const char32_t *str, const char32_t *end, char *dataptr,
        parser_config *pconfig)
{
    if (*str == '\0') {
        /* can't parse NUL, this field is probably just empty. */
        return -1;
    }

    int error;
    char32_t *p_end;

    double val = _Py_dg_strtod_modified(
            str, &p_end, &error, pconfig->decimal, pconfig->sci, true);

    if (error != 0 || p_end != end) {
        return -1;
    }
    memcpy(dataptr, &val, sizeof(double));
    if (!PyArray_ISNBO(descr->byteorder)) {
        descr->f->copyswap(dataptr, dataptr, 1, NULL);
    }
    return 0;
}


bool
to_complex_raw(
        const char32_t *item, double *p_real, double *p_imag,
        char32_t sci, char32_t decimal, char32_t imaginary_unit,
        bool allow_parens)
{
    char32_t *p_end;
    int error;
    bool unmatched_opening_paren = false;

    if (allow_parens && (*item == '(')) {
        unmatched_opening_paren = true;
        ++item;
    }
    *p_real = _Py_dg_strtod_modified(item, &p_end, &error, decimal, sci, false);
    if (*p_end == '\0') {
        // No imaginary part in the string (e.g. "3.5")
        *p_imag = 0.0;
        return (error == 0) && (!unmatched_opening_paren);
    }
    if (*p_end == imaginary_unit) {
        // Pure imaginary part only (e.g "1.5j")
        *p_imag = *p_real;
        *p_real = 0.0;
        ++p_end;
        if (unmatched_opening_paren && (*p_end == ')')) {
            ++p_end;
            unmatched_opening_paren = false;
        }
    }
    else if (unmatched_opening_paren && (*p_end == ')')) {
        *p_imag = 0.0;
        ++p_end;
        unmatched_opening_paren = false;
    }
    else {
        if (*p_end == '+') {
            ++p_end;
        }

        *p_imag = _Py_dg_strtod_modified(p_end, &p_end, &error, decimal, sci, false);
        if (error || (*p_end != imaginary_unit)) {
            return false;
        }
        ++p_end;
        if (unmatched_opening_paren && (*p_end == ')')) {
            ++p_end;
            unmatched_opening_paren = false;
        }
    }
    while(*p_end == ' ') {
        ++p_end;
    }
    return *p_end == '\0';
}


int
to_cfloat(PyArray_Descr *descr,
        const char32_t *str, const char32_t *end, char *dataptr,
        parser_config *pconfig)
{
    if (*str == '\0') {
        /* can't parse NUL, this field is probably just empty. */
        return -1;
    }

    double real;
    double imag;

    // TODO: This should check the end pointer (needs fixing in to_complex_int)
    bool success = to_complex_raw(
            str, &real, &imag,
            pconfig->sci, pconfig->decimal,
            pconfig->imaginary_unit, true);

    if (!success) {
        return -1;
    }
    complex float val = real + I*imag;
    memcpy(dataptr, &val, sizeof(complex float));
    if (!PyArray_ISNBO(descr->byteorder)) {
        descr->f->copyswap(dataptr, dataptr, 1, NULL);
    }
    return 0;
}



int
to_cdouble(PyArray_Descr *descr,
        const char32_t *str, const char32_t *end, char *dataptr,
        parser_config *pconfig)
{
    if (*str == '\0') {
        /* can't parse NUL, this field is probably just empty. */
        return -1;
    }

    double real;
    double imag;

    // TODO: This should check the end pointer (needs fixing in to_complex_int)
    bool success = to_complex_raw(
            str, &real, &imag,
            pconfig->sci, pconfig->decimal,
            pconfig->imaginary_unit, true);

    if (!success) {
        return -1;
    }
    complex double val = real + I*imag;
    memcpy(dataptr, &val, sizeof(complex double));
    if (!PyArray_ISNBO(descr->byteorder)) {
        descr->f->copyswap(dataptr, dataptr, 1, NULL);
    }
    return 0;
}


/*
 * String and unicode conversion functions.
 */
int
to_string(PyArray_Descr *descr,
        const char32_t *str, const char32_t *end, char *dataptr, void *unused)
{
    const char32_t* c = str;
    size_t length = descr->elsize;

    for (size_t i = 0; i < length; i++) {
        if (c < end) {
            // TODO: This cast is wrong as it really should be encoding/error?!
            dataptr[i] = (char)(*c);
            c++;
        }
        else {
            dataptr[i] = '\0';
        }
    }
    return 0;
}


int
to_unicode(PyArray_Descr *descr,
        const char32_t *str, const char32_t *end, char *dataptr, void *unused)
{
    size_t length = descr->elsize / 4;

    if (length <= (size_t)(end - str)) {
        memcpy(dataptr, str, length * 4);
    }
    else {
        size_t given_len = end - str;
        memcpy(dataptr, str, given_len * 4);
        memset(dataptr + given_len * 4, '\0', (length -given_len) * 4);
    }

    if (!PyArray_ISNBO(descr->byteorder)) {
        descr->f->copyswap(dataptr, dataptr, 1, NULL);
    }
    return 0;
}



/*
 * Convert functions helper for the generic converter.
 */
static PyObject *
call_converter_function(PyObject *func, const char32_t *str, size_t length)
{
    PyObject *s = PyUnicode_FromKindAndData(PyUnicode_4BYTE_KIND, str, length);
    if (s == NULL || func == NULL) {
        // fprintf(stderr, "*** PyUnicode_FromKindAndData failed ***\n");
        return s;
    }
    PyObject *result = PyObject_CallFunctionObjArgs(func, s, NULL);
    Py_DECREF(s);
    if (result == NULL) {
        // fprintf(stderr, "*** PyObject_CallFunctionObjArgs failed ***\n");
    }
    return result;
}


int
to_generic(PyArray_Descr *descr,
        const char32_t *str, const char32_t *end, char *dataptr, PyObject *func)
{
    /* Converts to unicode and calls custom converter (if set) */
    PyObject *converted = call_converter_function(
            func, str, (size_t)(end - str));
    if (converted == NULL) {
        return -1;
    }
    /* TODO: Dangerous semi-copy from PyArray_Pack which this
     *       should use, but cannot (it is not yet public).
     *       This will get some casts wrong (unlike PyArray_Pack),
     *       and like it (currently) does necessarily handle an
     *       array return correctly (but maybe that is fine).
     */
    PyArrayObject_fields arr_fields = {
            .flags = NPY_ARRAY_WRITEABLE,  /* assume array is not behaved. */
    };
    Py_SET_TYPE(&arr_fields, &PyArray_Type);
    Py_SET_REFCNT(&arr_fields, 1);
    arr_fields.descr = descr;
    int res = descr->f->setitem(converted, dataptr, &arr_fields);
    Py_DECREF(converted);
    if (res < 0) {
        return -1;
    }
    return 0;
}