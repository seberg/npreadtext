
#include <string.h>
#include "typedefs.h"
#include "str_to.h"
#include "error_types.h"
#include "conversions.h"
#include "parser_config.h"


// TODO: The float fallbacks are seriously awkward, why? Or at least why this way?
#define DECLARE_TO_INT(intw, INT_MIN, INT_MAX)                                      \
    int                                                                             \
    to_##intw(PyArray_Descr *descr,                                                 \
            const char32_t *str, const char32_t *end, char *dataptr,                \
            parser_config *pconfig)                                                 \
    {                                                                               \
        intw##_t x;                                                                 \
        int ierror = 0;                                                             \
                                                                                    \
        x = (intw##_t) str_to_int64(str, INT_MIN, INT_MAX, &ierror);                \
        if (ierror) {                                                               \
            if (pconfig->allow_float_for_int) {                                     \
                double fx;                                                          \
                char32_t decimal = pconfig->decimal;                                \
                char32_t sci = pconfig->sci;                                        \
                if ((*str == '\0') || !to_double_raw(str, &fx, decimal, sci)) {     \
                    return -1;                                                      \
                }                                                                   \
                else {                                                              \
                    x = (intw##_t) fx;                                              \
                }                                                                   \
            }                                                                       \
            else {                                                                  \
                return -1;                                                          \
            }                                                                       \
        }                                                                           \
        memcpy(dataptr, &x, sizeof(x));                                             \
        if (!PyArray_ISNBO(descr->byteorder)) {                                     \
            descr->f->copyswap(dataptr, dataptr, 1, NULL);                          \
        }                                                                           \
        return 0;                                                                   \
    }

#define DECLARE_TO_UINT(uintw, UINT_MAX)                                            \
    int                                                                             \
    to_##uintw(PyArray_Descr *descr,                                                \
            const char32_t *str, const char32_t *end, char *dataptr,                \
            parser_config *pconfig)                                                 \
    {                                                                               \
        uintw##_t x;                                                                \
        int ierror = 0;                                                             \
                                                                                    \
        x = (uintw##_t) str_to_uint64(str, UINT_MAX, &ierror);                      \
        if (ierror) {                                                               \
            if (pconfig->allow_float_for_int) {                                     \
                double fx;                                                          \
                char32_t decimal = pconfig->decimal;                                \
                char32_t sci = pconfig->sci;                                        \
                if ((*str == '\0') || !to_double_raw(str, &fx, decimal, sci)) {     \
                    return -1;                                                      \
                }                                                                   \
                else {                                                              \
                    x = (uintw##_t) fx;                                             \
                }                                                                   \
            }                                                                       \
            else {                                                                  \
                return -1;                                                          \
            }                                                                       \
        }                                                                           \
        memcpy(dataptr, &x, sizeof(x));                                             \
        if (!PyArray_ISNBO(descr->byteorder)) {                                     \
            descr->f->copyswap(dataptr, dataptr, 1, NULL);                          \
        }                                                                           \
        return 0;                                                                   \
    }

DECLARE_TO_INT(int8, INT8_MIN, INT8_MAX)
DECLARE_TO_INT(int16, INT16_MIN, INT16_MAX)
DECLARE_TO_INT(int32, INT32_MIN, INT32_MAX)
DECLARE_TO_INT(int64, INT64_MIN, INT64_MAX)

DECLARE_TO_UINT(uint8, UINT8_MAX)
DECLARE_TO_UINT(uint16, UINT16_MAX)
DECLARE_TO_UINT(uint32, UINT32_MAX)
DECLARE_TO_UINT(uint64, UINT64_MAX)
