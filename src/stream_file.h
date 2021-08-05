#ifndef STREAM_FILE_H
#define STREAM_FILE_H

#include "stream.h"

stream *
stream_file(FILE *f, int buffer_size);

#endif
