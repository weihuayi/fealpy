import sys

def pascal(n):
    """
    Yield up to row ``n`` of Pascal's triangle, one row at a time.
    The first row is row 0.
    """
    if not isinstance (n, int):
        raise TypeError ('n must be an integer')
    if n < 0:
        raise ValueError ('n must be an integer >= 0')

    def newrow(row):
        "Calculate a row of Pascal's triangle given the previous one." 
        prev = 0
        for x in row:
            yield prev + x
            prev = x
        yield 1

    prevrow = [1]
    yield prevrow
    for x in xrange(n):
        prevrow = list(newrow(prevrow))
        yield prevrow

def pascal_latex(n, out=sys.stdout):
    """
    Generate a Pascal triangle for LaTeX embedding.
    Sends output to the file-like object ``out`` (default: sys.stdout).
    """
    out.write('\\begin{tabular}{r%s}\n' % ('c' * (2 * n + 1)))
    for i, row in enumerate(pascal(n)):
        out.write('$n=%d$:& ' % i)
        out.write('   & ' * (n - i))
        out.write(' &    & '.join ('%2d' % coeff for coeff in row))
        out.write('\\\\\\noalign{\\smallskip\\smallskip}\n')
    out.write ('\\end{tabular}\n')

if __name__ == '__main__':
    try:
        pascal_latex (int(sys.argv[1]))
    except ValueError:
        sys.stderr.write ('Please specify a positive integer.\n')
        sys.exit (1)
