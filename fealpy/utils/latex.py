def to_latex_matrix(array, max_rows=10):
    num_rows = len(array)
    num_cols = len(array[0])

    # Define the header row
    header = "\\begin{matrix}\\hline\n"
    header += "r\\backslash c & " + " & ".join([f"{i}:" for i in range(num_cols)]) + " \\\\ \\hline\n"
    
    # Define the content rows
    rows = ""
    for i in range(min(max_rows, num_rows)):
        rows += f"{i}: & " + " & ".join(map(str, array[i])) + " \\\\ \\hline\n"
    
    # Add ellipsis if necessary
    if num_rows > max_rows:
        rows += "\\vdots & " + " & ".join(["\\vdots"] * num_cols) + " \\\\ \\hline\n"
        rows += f"{num_rows-1}: & " + " & ".join(map(str, array[-1])) + " \\\\ \\hline\n"
    
    footer = "\\end{matrix}"
    
    return header + rows + footer
