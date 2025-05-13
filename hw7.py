import numpy as np

# 定義矩陣 A 和向量 b
A = np.array([
    [4, -1, 0, -1, 0, 0],
    [-1, 4, -1, 0, -1, 0],
    [0, -1, 4, 0, 1, -1],
    [-1, 0, 0, 4, -1, -1],
    [0, -1, 0, -1, 4, -1],
    [0, 0, -1, 0, -1, 4]
], dtype=float)
b = np.array([0, -1, 9, 4, 8, 6], dtype=float)

# 初始猜測
x0 = np.zeros_like(b)
tol = 1e-8
max_iter = 1000

# Jacobi Method
def jacobi(A, b, x0, tol, max_iter):
    D = np.diag(np.diag(A))
    R = A - D
    x = x0.copy()
    for _ in range(max_iter):
        x_new = np.linalg.inv(D).dot(b - R.dot(x))
        if np.linalg.norm(x_new - x, ord=np.inf) < tol:
            break
        x = x_new
    return x

# Gauss-Seidel Method
def gauss_seidel(A, b, x0, tol, max_iter):
    x = x0.copy()
    n = len(b)
    for _ in range(max_iter):
        x_new = x.copy()
        for i in range(n):
            s1 = sum(A[i, j] * x_new[j] for j in range(i))
            s2 = sum(A[i, j] * x[j] for j in range(i+1, n))
            x_new[i] = (b[i] - s1 - s2) / A[i, i]
        if np.linalg.norm(x_new - x, ord=np.inf) < tol:
            break
        x = x_new
    return x

# SOR Method
def sor(A, b, x0, omega, tol, max_iter):
    x = x0.copy()
    n = len(b)
    for _ in range(max_iter):
        x_new = x.copy()
        for i in range(n):
            s1 = sum(A[i, j] * x_new[j] for j in range(i))
            s2 = sum(A[i, j] * x[j] for j in range(i+1, n))
            x_new[i] = x[i] + omega * ((b[i] - s1 - s2) / A[i, i] - x[i])
        if np.linalg.norm(x_new - x, ord=np.inf) < tol:
            break
        x = x_new
    return x

# Conjugate Gradient Method
def conjugate_gradient(A, b, x0, tol, max_iter):
    x = x0.copy()
    r = b - A.dot(x)
    p = r.copy()
    rs_old = np.dot(r, r)
    for _ in range(max_iter):
        Ap = A.dot(p)
        alpha = rs_old / np.dot(p, Ap)
        x = x + alpha * p
        r = r - alpha * Ap
        rs_new = np.dot(r, r)
        if np.sqrt(rs_new) < tol:
            break
        p = r + (rs_new / rs_old) * p
        rs_old = rs_new
    return x

# 執行
x_jacobi = jacobi(A, b, x0, tol, max_iter)
x_gs = gauss_seidel(A, b, x0, tol, max_iter)
x_sor = sor(A, b, x0, omega=1.25, tol=tol, max_iter=max_iter)
x_cg = conjugate_gradient(A, b, x0, tol, max_iter)

# 輸出最終結果
print("Final Solutions:")
print("Jacobi:           ", x_jacobi)
print("Gauss-Seidel:     ", x_gs)
print("SOR (omega=1.25): ", x_sor)
print("Conjugate Gradient:", x_cg)
