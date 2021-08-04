
#ifndef _ANALYZE_H_
#define _ANALYZE_H_

#include "field_types.h"
#include "parser_config.h"
#include "stream.h"


#define ANALYZE_FILE_ERROR    -1
#define ANALYZE_OUT_OF_MEMORY -2

int
analyze(stream *s, parser_config *pconfig, int skiplines, int numrows,
        int *p_num_fields, field_type **p_field_types);

#endif
