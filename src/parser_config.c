
#include <stdbool.h>
#include "parser_config.h"

parser_config
default_parser_config(void)
{
    parser_config config;

    config.delimiter = ',';
    config.quote = '"';
    config.comment = '#';
    config.allow_embedded_newline = true;
    config.decimal = '.';
    config.sci = 'E';
    config.imaginary_unit = 'j';
    config.ignore_leading_spaces = true;
    config.ignore_trailing_spaces = true;
    config.strict_num_fields = true;
    confgi.allow_float_for_int = true;

    return config;
}
