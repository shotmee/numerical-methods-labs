import numpy as np
import pandas as pd

np.set_printoptions(precision=10, suppress=True)

VARIANT = 11
ROWS_IN_MATRIX = 15
EXPECTED_COLS = 14
CSV_PATH = "housing.csv"

def read_housing_csv(path):
    for header_mode in ["infer", None]:
        df = pd.read_csv(path, header=header_mode)
        numeric_df = df.select_dtypes(include=[np.number])
        if numeric_df.shape[1] == EXPECTED_COLS:
            return numeric_df

def jacobi_method(M, tol=1e-9, max_iter=5000):
    n = M.shape[0]
    A = M.copy()
    U = np.eye(n)
    for i in range(max_iter):
        off_diag = A - np.diag(np.diag(A))
        idx = np.argmax(np.abs(off_diag))
        p, q = np.unravel_index(idx, A.shape)
        max_val = abs(A[p, q])
        if max_val < tol:
            break
        if A[p, p] == A[q, q]:
            c = np.cos(np.pi / 4)
            s = np.sin(np.pi / 4)
        else:
            t = 2 * A[p, q] / (A[p, p] - A[q, q])
            sq = np.sqrt(1 + t**2)
            c = np.sqrt(0.5 * (1 + 1 / sq))
            s = np.sign(t) * np.sqrt(0.5 * t**2 / (sq * (sq + 1)))
        S = np.eye(n)
        S[p, p] = c
        S[q, q] = c
        S[p, q] = -s
        S[q, p] = s
        A = S.T @ A @ S
        U = U @ S
    eigenvalues = np.diag(A)
    idx = np.argsort(eigenvalues)[::-1]
    return eigenvalues[idx], U[:, idx]

df = read_housing_csv(CSV_PATH)
data = df.to_numpy(dtype=float)
A = data[(VARIANT - 1) * ROWS_IN_MATRIX : VARIANT * ROWS_IN_MATRIX, :]

ATA = A.T @ A
eigvals_V, V = jacobi_method(ATA, tol=1e-9)

AAT = A @ A.T
eigvals_U, U = jacobi_method(AAT, tol=1e-9)

# Filter eigenvalues of ATA based on adaptive threshold
eps_lambda = np.max(eigvals_V) * max(A.shape) * 1e-12
eigvals_V[eigvals_V < eps_lambda] = 0

singular_values = np.sqrt(np.maximum(eigvals_V, 0))

for i in range(min(A.shape)):
    if singular_values[i] > 0:
        u_expected = A @ V[:, i] / singular_values[i]
        if np.dot(U[:, i], u_expected) < 0:
            U[:, i] = -U[:, i]

singular_values = singular_values[singular_values > 0]

S = np.zeros_like(A)
for i in range(min(A.shape)):
    if i < len(singular_values):
        S[i, i] = singular_values[i]

A_approx = U @ S @ V.T
print("Norm diff:", np.linalg.norm(A - A_approx, 2))
print("Number of singular values:", len(singular_values))
