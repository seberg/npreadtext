#include "numpy/ndarraytypes.h"
#include "growth.h"


/*
 * Temporary copy from numpy, before we can just include it from there
 * (this helper is private in NumPy).
 */
static NPY_INLINE int
npy_mul_with_overflow_intp(npy_intp *r, npy_intp a, npy_intp b)
{
#ifdef HAVE___BUILTIN_MUL_OVERFLOW
    return __builtin_mul_overflow(a, b, r);
#else
    const npy_intp half_sz = ((npy_intp)1 << ((sizeof(a) * 8 - 1 ) / 2));

    *r = a * b;
    /*
     * avoid expensive division on common no overflow case
     */
    if (NPY_UNLIKELY((a | b) >= half_sz) &&
        a != 0 && b > NPY_MAX_INTP / a) {
        return 1;
    }
    return 0;
#endif
}


/*
 * Helper function taking the size input and growing it (based on min_grow).
 * It further multiplies it with `itemsize` and ensures that all results fit
 * into an `npy_intp`.
 * Returns -1 if any overflow occurred or the result would not fit.
 * The user has to ensure the input is size_t (i.e. unsigned).
 */
npy_intp
grow_size_and_multiply(size_t *size, size_t min_grow, npy_intp itemsize) {
    /* min_grow must be a power of two: */
    assert(min_grow & (min_grow - 1) == 0);
    size_t growth = *size >> 2;
    if (growth <= min_grow) {
        *size += min_grow;
    }
    else {
        *size += growth + min_grow - 1;
        *size &= ~min_grow;

        if (*size > NPY_MAX_INTP) {
            return -1;
        }
    }

    npy_intp res;
    if (npy_mul_with_overflow_intp(&res, (npy_intp)*size, itemsize)) {
        return -1;
    }
    return res;
}
