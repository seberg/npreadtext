
#ifndef _STREAM_PYTHON_FILE_BY_LINE
#define _STREAM_PYTHON_FILE_BY_LINE

#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include "stream.h"

stream *
stream_python_file(PyObject *obj, PyObject *encoding);

stream *
stream_python_iterable(PyObject *obj, PyObject *encoding);

#endif
