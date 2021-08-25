#ifndef _TYPE_INFERENCE_H_
#define _TYPE_INFERENCE_H_


char
classify_type(Py_UCS4 *field,
        Py_UCS4 decimal, Py_UCS4 sci, Py_UCS4 imaginary_unit,
        int64_t *i, uint64_t *u, char prev_type);

void
update_type_for_integer_range(
        char *type, size_t *itemsize, int64_t imin, uint64_t umax);

#endif
