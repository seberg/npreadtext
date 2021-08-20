
#ifndef _TOKENIZE_H_
#define _TOKENIZE_H_

#include "typedefs.h"
#include "stream.h"
#include "parser_config.h"


/* Tokenization state machine states. */
#define TOKENIZE_OUTSIDE_FIELD 8
/* not in fields */
#define TOKENIZE_INIT   (0 | TOKENIZE_OUTSIDE_FIELD)
#define TOKENIZE_EAT_CRLF   (1 | TOKENIZE_OUTSIDE_FIELD)
#define TOKENIZE_LINE_END   (2 | TOKENIZE_OUTSIDE_FIELD)
#define TOKENIZE_GOTO_LINE_END (3 | TOKENIZE_OUTSIDE_FIELD)
/* inside fields */
#define TOKENIZE_CHECK_QUOTED 0
#define TOKENIZE_UNQUOTED     1
#define TOKENIZE_QUOTED       2
#define TOKENIZE_QUOTED_CHECK_DOUBLE_QUOTE 3
/* technically not necessarily inside a field, but it may be */
#define TOKENIZE_CHECK_COMMENT 4


typedef struct {
    size_t offset;
    bool quoted;
} field_info;


typedef struct {
    int state;
    int unicode_kind;
    int buf_state;
    size_t num_fields;
    /* the buffer we are currently working on */
    char *pos;
    char *end;
    /*
     * Space to copy words into.  The buffer must always be at least two NUL
     * entries longer (8 bytes) than the actual word (including initially).
     * The first byte beyond the current word is always NUL'ed on write, the
     * second byte is there to allow easy appending of an additional empty
     * word at the end (this word is also NUL terminated).
     */
    size_t field_buffer_length;
    size_t field_buffer_pos;
    char32_t *field_buffer;

    /*
     * Fields, including information about the field being quoted.  This
     * always includes one "additional" empty field.  The length of a field
     * is equal to `fields[i+1].offset - fields[i].offset - 1`.
     *
     * The tokenizer assumes at least one field is allocated.
     */
    field_info *fields;
    size_t fields_size;
} tokenizer_state;


void
tokenizer_clear(tokenizer_state *ts);


int
tokenizer_init(tokenizer_state *ts, parser_config *config);

int
tokenize(stream *s, tokenizer_state *ts, parser_config *const config);

#endif
