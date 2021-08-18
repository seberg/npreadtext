
#ifndef _TOKENIZE_H_
#define _TOKENIZE_H_

#include "typedefs.h"
#include "stream.h"
#include "parser_config.h"


/* Tokenization state machine states. */
#define TOKENIZE_OUTSIDE_FIELD 8
/* not in fields */
#define TOKENIZE_NEW_LINE   (1 + TOKENIZE_OUTSIDE_FIELD)
#define TOKENIZE_INIT       (2 + TOKENIZE_OUTSIDE_FIELD)
#define TOKENIZE_WHITESPACE (3 + TOKENIZE_OUTSIDE_FIELD)
#define TOKENIZE_EAT_CRLF   (4 + TOKENIZE_OUTSIDE_FIELD)
#define TOKENIZE_FINALIZE_LINE (5 + TOKENIZE_OUTSIDE_FIELD)
#define TOKENIZE_FINALIZE_FILE (6 + TOKENIZE_OUTSIDE_FIELD)
/* inside fields */
#define TOKENIZE_UNQUOTED   1
#define TOKENIZE_QUOTED     2
#define TOKENIZE_QUOTED_CHECK_DOUBLE_QUOTE 3
/* technically not necessarily inside a field, but it may be */
#define TOKENIZE_CHECK_COMMENT 4


typedef struct {
    size_t offset;
    size_t length;
    bool quoted;
} field_info;


typedef struct {
    int state;
    int buf_state;
    size_t num_fields;
    /* the buffer we are currently working on */
    char32_t *pos;
    char32_t *end;
    /*
     * In some cases, we need to copy words.  We will use this length
     * and state.  The word buffer only grows (we assume this is OK).
     * TODO: If this turns out to be used regularly, it may make sense to
     *       initialize this to a stack allocated version that is usually
     *       big enough.
     */
    size_t field_buffer_length;
    size_t field_buffer_pos;
    char32_t *field_buffer;

    field_info *fields;
    size_t fields_size;
} tokenizer_state;


void
tokenizer_clear(tokenizer_state *ts);


void
tokenizer_init(tokenizer_state *ts, parser_config *config);

int
tokenize(stream *s, tokenizer_state *ts, parser_config *const config);

#endif
