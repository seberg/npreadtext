//
// stream_python_file_by_line.c
//
// The public function defined in this file is
//
//     stream *stream_python_file_by_line(PyObject *obj)
//
// This function wraps a Python file object in a stream that
// can be used by the text file reader.
//

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>

#include <sys/types.h>

#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include "stream.h"


typedef struct _python_file_by_line {

    /* The Python file object being read. */
    PyObject *file;

    /* The `readline` attribute of the file object. */
    PyObject *readline;

    /* The `seek` attribute of the file object. */
    PyObject *seek;

    /* The `tell` attribute of the file object. */
    PyObject *tell;

    /* file position when the file_buffer was created. */
    off_t initial_file_pos;

    int32_t line_number;

    /* Boolean: has the end of the file been reached? */
    bool reached_eof;

    /* Python str object holding the line most recently read from the file. */
    PyObject *line;

    int kind;
    char *buffer;

    /* Length of line */
    Py_ssize_t linelen;

    /* Unicode kind of line (see the Python docs about unicode) */
    int unicode_kind;

    /* The DATA associated with line. */
    void *unicode_data;

    /* Position in the buffer of the next character to read. */
    Py_ssize_t current_buffer_pos;

    // encoding must be None or a bytes object holding an
    // ASCII-encoding string, e.g. b'utf-8'.
    PyObject *encoding;

} python_file_by_line;

#define FB(fb)  ((python_file_by_line *)fb)



/*
 *  int _fb_load(void *fb)
 *
 *  Get data from the file into the buffer.
 *
 *  Returns 0 on success.
 *  Returns STREAM_ERROR on error.
 */

static int
_fb_load(python_file_by_line *fb)
{
    Py_XDECREF(fb->line);
    fb->line = NULL;
    fb->buffer = NULL;

    PyObject *line = PyObject_CallFunctionObjArgs(fb->readline, NULL);
    fb->line = line;
    if (line == NULL) {
        return -1;
    }
    if (PyBytes_Check(line)) {
        PyObject *uline;
        char *enc;
        // readline() returned bytes, so encode it.
        // XXX if no encoding was specified, assume UTF-8.
        if (fb->encoding == Py_None) {
            enc = "utf-8";
        }
        else {
            enc = PyBytes_AsString(fb->encoding);
        }
        uline = PyUnicode_FromEncodedObject(line, enc, NULL);
        if (uline == NULL) {
            // XXX temporary printf
            printf("_fb_load: failed to decode bytes object\n");
            return -1;
        }
        Py_SETREF(fb->line, uline);
    }
    else if (!PyUnicode_Check(line)) {
        PyErr_SetString(PyExc_TypeError,
                "object.readline() returned non-string");
        Py_CLEAR(fb->line);
        return -1;
    }

    fb->linelen = PyUnicode_GET_LENGTH(fb->line);

    fb->kind = PyUnicode_KIND(fb->line);
    if (fb->kind == PyUnicode_1BYTE_KIND) {
        fb->buffer = (char *)PyUnicode_1BYTE_DATA(fb->line);
    }
    else if (fb->kind == PyUnicode_2BYTE_KIND) {
        fb->buffer = (char *)PyUnicode_2BYTE_DATA(fb->line);
        fb->linelen *= sizeof(Py_UCS2);
    }
    else if (fb->kind == PyUnicode_4BYTE_KIND) {
        fb->buffer = (char *)PyUnicode_4BYTE_DATA(fb->line);
        fb->linelen *= sizeof(Py_UCS4);
    }

    if (fb->linelen == 0) {
        return BUFFER_IS_FILEEND;
    }
    return BUFFER_IS_LINEND;
}


static int
fb_nextbuf(python_file_by_line *fb, char **start, char **end, int *kind)
{
    int status = _fb_load(fb);

    *start = fb->buffer;
    *end = fb->buffer + fb->linelen;
    *kind = fb->kind;
    return status;
}


static int
fb_seek(void *fb, long int pos)
{
    int status = 0;

    PyObject *args = Py_BuildValue("(n)", (Py_ssize_t) pos);
    // XXX Check for error, and
    // DECREF where appropriate...
    PyObject *result = PyObject_Call(FB(fb)->seek, args, NULL);
    if (result == NULL) {
        return -1;
    }
    Py_DECREF(result);
    // XXX Check for error!
    FB(fb)->line_number = 1;
    //FB(fb)->buffer_file_pos = FB(fb)->initial_file_pos;
    FB(fb)->current_buffer_pos = 0;
    //FB(fb)->last_pos = 0;
    FB(fb)->reached_eof = false;
    return status;
}


static int
stream_del(stream *strm, int restore)
{
    python_file_by_line *fb = (python_file_by_line *) (strm->stream_data);

    if (restore == RESTORE_INITIAL) {
        // XXX
        stream_seek(strm, SEEK_SET);
        //fseek(FB(fb)->file, FB(fb)->initial_file_pos, SEEK_SET);
    }
    else if (restore == RESTORE_FINAL) {
        // XXX
        stream_seek(strm, SEEK_SET);
        //fseek(FB(fb)->file, FB(fb)->buffer_file_pos + FB(fb)->current_buffer_pos, SEEK_SET);
    }

    // XXX Wrap the following clean up code in something more modular?
    Py_XDECREF(fb->file);
    Py_XDECREF(fb->readline);
    Py_XDECREF(fb->seek);
    Py_XDECREF(fb->tell);
    Py_XDECREF(fb->line);

    free(fb);
    free(strm);

    return 0;
}


stream *
stream_python_file_by_line(PyObject *obj, PyObject *encoding)
{
    python_file_by_line *fb;
    stream *strm;
    PyObject *func;

    fb = (python_file_by_line *) malloc(sizeof(python_file_by_line));
    //printf("fb = %lld\n", fb);
    if (fb == NULL) {
        // XXX handle the errors here and below properly.
        fprintf(stderr, "stream_file: malloc() failed.\n");
        return NULL;
    }

    fb->buffer = NULL;

    fb->file = NULL;
    fb->readline = NULL;
    fb->seek = NULL;
    fb->tell = NULL;
    fb->line = NULL;
    fb->encoding = encoding;

    strm = (stream *) malloc(sizeof(stream));
    if (strm == NULL) {
        // XXX Don't print to stderr!
        fprintf(stderr, "stream_file: malloc() failed.\n");
        free(fb);
        return NULL;
    }

    fb->file = obj;
    Py_INCREF(fb->file);

    func = PyObject_GetAttrString(obj, "readline");
    if (!func) {
        goto fail;
    }
    fb->readline = func;
    Py_INCREF(fb->readline);

    func = PyObject_GetAttrString(obj, "seek");
    if (!func) {
        goto fail;
    }
    fb->seek = func;
    Py_INCREF(fb->seek);

    func = PyObject_GetAttrString(obj, "tell");
    if (!func) {
        goto fail;
    }
    fb->tell = func;
    Py_INCREF(fb->tell);

    fb->line_number = 1;
    fb->linelen = 0;

    fb->current_buffer_pos = 0;
    //fb->last_pos = 0;
    fb->reached_eof = 0;

    strm->stream_data = (void *)fb;
    strm->stream_nextbuf = &fb_nextbuf;
    strm->stream_seek = &fb_seek;
    strm->stream_close = &stream_del;

    return strm;

fail:
    Py_XDECREF(fb->file);
    Py_XDECREF(fb->readline);
    Py_XDECREF(fb->seek);
    Py_XDECREF(fb->tell);

    free(fb);
    free(strm);
    return NULL;
}
