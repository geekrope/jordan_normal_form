import numpy as np
import sympy as sp
from scipy.linalg import block_diag


def lin_independent(B):
    echelon_form = sp.Matrix.hstack(*B).rref()
    return [B[i] for i in echelon_form[1]]


def disjoint(V, U):
    disjoint_ = []
    for v in V:
        projection = sp.zeros(v.shape[0], v.shape[1])
        for u in U:
            projection = projection + v.dot(u) * u
        disjoint_.append(v - projection)
    return lin_independent(disjoint_)


def length(v):
    return sp.sqrt(v.dot(v))


def orthonormalize(B):
    if len(B) == 0:
        return []

    orthonormal = [B[0] / length(B[0])]
    for b in B[1:]:
        projection = sp.zeros(b.shape[0], b.shape[1])
        for o in orthonormal:
            projection = projection + b.dot(o) * o
        diff = b - projection
        orthonormal.append(diff / length(diff))
    return orthonormal


def ranks(matrix: sp.Matrix):
    ranks_ = [
        matrix.shape[0]
    ]  # dim  E(lambda) = n-dim Im(A-lambda)=n-rk (A-lambda)
    bases_ = [[]]  # dim Ker(Id) = 0
    current_matrix = sp.eye(matrix.shape[0])
    while True:
        current_rank = current_matrix.rank()
        next_matrix = current_matrix * matrix
        if current_rank == next_matrix.rank():
            break
        ranks_.append(next_matrix.rank())
        bases_.append(orthonormalize(next_matrix.nullspace()))
        current_matrix = next_matrix
    return (ranks_[::-1], bases_[::-1])


def chains(matrix, ranks, bases):
    chain_count = 0
    chains_sizes_ = []
    chains_ = [[]]

    for i in range(len(ranks) - 1):
        d = ranks[i + 1] - ranks[i]
        chains_cont = [matrix * vector for vector in chains_[len(chains_) - 1]]
        disjoint_basis = disjoint(bases[i], bases[i + 1])

        if d > chain_count:
            new_chains = disjoint(
                orthonormalize(disjoint_basis), orthonormalize(chains_cont)
            )
            chains_cont.extend(new_chains)
            chains_sizes_.append((d - chain_count, len(ranks) - 1 - i))
            chain_count = d
        chains_.append(chains_cont)

    return (chains_sizes_, chains_)


def collect_chain(chains, i):
    chain = []
    j = len(chains) - 1
    while j >= 0 and len(chains[j]) > i:
        chain.append(chains[j][i])
        j -= 1
    return chain


def collect_chains(chains):
    collected = []
    for i in range(len(chains[len(chains) - 1])):  # starting from eigenvectors
        chain = collect_chain(chains, i)
        collected.extend(chain)
    return collected


def jordan_form(matrix: sp.Matrix, raw=False):
    size = matrix.shape[0]
    blocks = []
    raw_blocks = []
    evas = matrix.eigenvals().keys()
    collected_chains = []
    for eva in evas:
        nilpotent = matrix - eva * sp.eye(
            size
        )  # nilpotent but this doesnt tell much
        ranks_, bases_ = ranks(nilpotent)
        chains_sizes_, chains_ = chains(nilpotent, ranks_, bases_)
        collected_chains.extend(collect_chains(chains_))
        for chain in chains_sizes_:
            chain_size = chain[1]
            block = sp.jordan_cell(eva, chain_size)
            raw_blocks.extend([(eva, chain_size)] * chain[0])
            blocks.extend([block] * chain[0])
    if not raw:
        return (
            sp.Matrix(block_diag(*blocks)),
            sp.Matrix.hstack(*collected_chains),
        )
    else:
        return (raw_blocks, sp.Matrix.hstack(*collected_chains))


def euclidian_vector(size, index):
    vector = np.zeros(size)
    vector[index] = 1
    return vector


def raise_block(eva, size, power):
    columns = [eva * euclidian_vector(size, 0)]
    columns.extend(
        [
            eva * euclidian_vector(size, i) + euclidian_vector(size, i - 1)
            for i in range(1, size)
        ]
    )
    for p in range(power - 1):
        prev = columns[0].copy()
        columns[0] = eva * columns[0]
        for i in range(1, size):
            new_prev = columns[i].copy()
            columns[i] = eva * columns[i] + prev
            prev = new_prev
    return np.vstack(columns).T


def raise_matrix(matrix: sp.Matrix, power):
    J, P = jordan_form(matrix, True)
    P = sympy_to_numpy(P)
    blocks = [raise_block(block[0], block[1], power) for block in J]

    return P @ block_diag(*blocks) @ np.linalg.inv(P)


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

J, P = jordan_form(A)

print(J)
print(P * J * P.inv())
print(raise_matrix(A, 3))