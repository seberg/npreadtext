
class FileGen:
    """
    Class that wraps a generator to make it look file-like.
    """
    def __init__(self, gen):
        self._gen = gen
        self._position = 0

    def readline(self):
        try:
            line = next(self._gen)
            self._position += len(line)
            line += '\n'
        except StopIteration:
            line = ''
        return line

    def seek(self, pos, whence=0):
        # For this class, seek(pos) is a no-op.
        return self._position

    def tell(self):
        return self._position


class WrapFileLikeStrippingComments:
    """
    Class for moving multiple (and long) comment parsing partially to Python.
    We don't bother to do all of the tokenization in Python, just strip down
    the lines.

    If this needs to be expanded, it may make sense to do all of the tokenizer
    work in Python instead.
    """
    def __init__(self, wrapped, encoding, comments):
        self._wrapped = wrapped
        self._readline = wrapped.readline
        self._encoding = encoding
        self._comments = comments

    def readline(self):
        orig_line = self._readline()
        if isinstance(orig_line, bytes):
            # Need to handle conversion here, because we may need to append
            # a newline...
            orig_line = orig_line.decode(self._encoding)

        if not orig_line:
            # If the line is empty, no more to read (must not append newline)
            return orig_line

        line = orig_line
        for c in self._comments:
            line = line.split(c, 1)[0]

        # If line was actually split above, we need to ensure it ends with a
        # newline character (the tokenizer currently expects this)
        if len(line) == len(orig_line):
            return line
        return line + "\n"

    def seek(self, pos, whence=0):
        return self._wrapped.seek(pos, whence=whence)

    def tell(self):
        return self._wrapped.tell()

    def close(self):
        # TODO: This may be needed, but no test covers it.
        return self._wrapped.close()
