

def einsum(equation, *operands):
    """
    """
    # Parse the equation
    lhs, rhs = equation.split('->')
    lhs_terms = lhs.split(',')

    # Determine the shape of the result tensor
    shape_map = {}
    for term, operand in zip(lhs_terms, operands):
        for dim, size in zip(term, operand.shape):
            if dim in shape_map:
                assert shape_map[dim] == size, f"Inconsistent sizes for dimension {dim}"
            else:
                shape_map[dim] = size
    result_shape = [shape_map[dim] for dim in rhs]
    
    # Initialize the result field
    result = ti.field(dtype=operands[0], shape=tuple(result_shape))

    # Generate the kernel
    @ti.kernel
    def compute():
        for I in ti.grouped(result):
            index_map = {dim: I[rhs.index(dim)] for dim in rhs}
            result[I] = 0.0
            for indices in ti.grouped(ti.ndrange(*[shape_map[dim] for term in lhs_terms for dim in term])):
                valid = True
                for term, operand in zip(lhs_terms, operands):
                    for dim, idx in zip(term, indices):
                        if index_map[dim] != idx:
                            valid = False
                            break
                    if not valid:
                        break
                if valid:
                    value = 1.0
                    for term, operand in zip(lhs_terms, operands):
                        for dim, idx in zip(term, indices):
                            value *= operand[indices[term.index(dim)]]
                    result[I] += value
    
    compute()
    return result

