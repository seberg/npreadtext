#ifndef _STREAM_H_
#define _STREAM_H_

#include <stdint.h>


#define RESTORE_NOT     0
#define RESTORE_INITIAL 1
#define RESTORE_FINAL   2

/*
 * When getting the next line, we hope that the buffer provider can already
 * give some information about the newlines, because for Python iterables
 * we definitely expect to get line-by-line buffers.
 */
#define BUFFER_MAY_CONTAIN_NEWLINE 0
#define BUFFER_IS_PARTIAL_LINE 1
#define BUFFER_IS_LINEND 2
#define BUFFER_IS_FILEEND 3

typedef struct _stream {
    void *stream_data;
    int (*stream_nextbuf)(void *sdata, char **start, char **end, int *kind);
    int (*stream_seek)(void *sdata, long int pos);
    // Note that the first argument to stream_close is the stream pointer
    // itself, not the stream_data pointer.
    int (*stream_close)(struct _stream *strm, int);
} stream;


#define stream_nextbuf(s, start, end, kind)  \
        ((s)->stream_nextbuf((s)->stream_data, start, end, kind))
#define stream_seek(s, pos)         ((s)->stream_seek((s)->stream_data, (pos)))
#define stream_close(s, restore)    ((s)->stream_close((s), (restore)))

#endif
