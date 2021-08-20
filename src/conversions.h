#ifndef CONVERSIONS_H
#define CONVERSIONS_H

#include <stdbool.h>

#include "typedefs.h"
#include "parser_config.h"

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#define NO_IMPORT_ARRAY
#define PY_ARRAY_UNIQUE_SYMBOL npreadtext_ARRAY_API
#include "numpy/arrayobject.h"

int
to_float(PyArray_Descr *descr,
        const char32_t *str, const char32_t *end, char *dataptr,
        parser_config *pconfig);

bool
to_double_raw(const char32_t *str, double *res, char32_t decimal, char32_t sci);

int
to_double(PyArray_Descr *descr,
        const char32_t *str, const char32_t *end, char *dataptr,
        parser_config *pconfig);

bool
to_complex_raw(
        const char32_t *item, double *p_real, double *p_imag,
        char32_t sci, char32_t decimal, char32_t imaginary_unit,
        bool allow_parens);

int
to_cfloat(PyArray_Descr *descr,
        const char32_t *str, const char32_t *end, char *dataptr,
        parser_config *pconfig);

int
to_cdouble(PyArray_Descr *descr,
        const char32_t *str, const char32_t *end, char *dataptr,
        parser_config *pconfig);

int
to_string(PyArray_Descr *descr,
        const char32_t *str, const char32_t *end, char *dataptr,
        void *unused);

int
to_unicode(PyArray_Descr *descr,
        const char32_t *str, const char32_t *end, char *dataptr,
        void *unused);

int
to_generic(PyArray_Descr *descr,
        const char32_t *str, const char32_t *end, char *dataptr,
        PyObject *func);

#endif
