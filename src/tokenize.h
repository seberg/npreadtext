
#ifndef _TOKENIZE_H_
#define _TOKENIZE_H_

#include "typedefs.h"
#include "stream.h"
#include "parser_config.h"


/* Tokenization state machine states. */
#define TOKENIZE_INIT       0
#define TOKENIZE_UNQUOTED   1
#define TOKENIZE_QUOTED     2
#define TOKENIZE_WHITESPACE 3


typedef struct {
    int state;
    size_t curr_field;
    /* information about the current word, may or may not be owned by us */
    char32_t *const word_pos;
    char32_t *const word_end;
    /* the buffer we are currently working on */
    char32_t **pos;
    char32_t **end;
    /*
     * In some cases, we may need to copy words.  We will use this length
     * and state.  The word buffer only grows (we assume this is OK).
     * TODO: If this turns out to be used regularly, it may make sense to
     *       initialize this to a stack allocated version that is usually
     *       big enough.
     */
    size_t word_buffer_length;
    char32_t *word_buffer;
} tokenizer_state;


void
tokenizer_clear(*tokenizer_state);


char32_t **
tokenize(stream *fb, tokenizer_state *ts,
        char32_t *word_buffer, int word_buffer_size,
        parser_config *pconfg, int *p_num_fields, int *p_error_type);

#endif
