def ranks(matrix: sp.Matrix):
    ranks = [matrix.shape[0]]  #dim(E(lambda)) = n-dim(Im(A-lambda)) = n-rk(A-lambda)
    current_matrix = sp.eye(matrix.shape[0])
    while True:
        current_rank = current_matrix.rank()
        next_matrix = current_matrix * matrix
        if current_rank == next_matrix.rank():
            break
        ranks.append(next_matrix.rank())
        current_matrix = next_matrix
    return ranks[::-1]


def chains(ranks):
    chain_count = 0
    chains_ = []

    for i in range(len(ranks) - 1):
        d = ranks[i + 1] - ranks[i]
        if d > chain_count:
            chains_.append((d - chain_count, len(ranks) - 1 - i))
            chain_count = d

    return chains_


def jordan_form(matrix: sp.Matrix):
    size = matrix.shape[0]
    blocks = []
    evas = matrix.eigenvals().keys()
    for eva in evas:
        chains_ = chains(ranks(matrix - eva * sp.eye(size)))
        for chain in chains_:
            chain_size = chain[1]
            block = sp.jordan_cell(eva, chain_size)
            blocks.extend([block] * chain[0])
    return sp.Matrix(block_diag(*blocks))


A = sp.Matrix(
    np.array(
        [
            [3, 1, -1, 1, -1],
            [-1, 5, 1, -1, 1],
            [-1, 1, 3, 1, -1],
            [0, 0, 0, 4, 2],
            [-1, 1, 1, -1, 7],
        ]
    )
)

print(jordan_form(A))