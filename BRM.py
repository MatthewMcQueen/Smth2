import numpy as np

def ibr_solver(A, tol=1e-6, max_iter=1000):
    m, n = A.shape
    p, q = np.ones(m) / m, np.ones(n) / n
    for _ in range(max_iter):
        uprev, vprev = np.log(np.dot(A.T, p)), np.log(np.dot(A, q))
        u = np.zeros(m)
        v = np.zeros(n)
        for i in range(m):
            u[i] = np.max(A[i, :] - vprev)
        for j in range(n):
            v[j] = np.max(A[:, j] - uprev)
        epsilon = 1 / np.max(np.abs(u - uprev))
        v = epsilon * np.log(vprev) + np.log(q)
        pnew = np.exp(u - np.max(u)) / np.sum(np.exp(u - np.max(u)))
        qnew = np.exp(v - np.max(v)) / np.sum(np.exp(v - np.max(v)))
        errp = np.max(np.abs(pnew - p))
        errq = np.max(np.abs(qnew - q))
        p, q = pnew, qnew
        if errp < tol and errq < tol:
            break
    ustar, vstar = np.dot(A, q), np.dot(A.T, p)
    return ustar, vstar, p, q

m, n = map(int, input("Введіть розмірність мантриці гри у форматі m n: ").split())
A = np.zeros((m, n))
for i in range(m):
    A[i, :] = list(map(int, input(f"Введіть елементи {i+1}-ого рядка через пробіл: ").split()))
max_iter = int(input("Введіть кількість ітерацій: "))

ustar, vstar, p, q = ibr_solver(A, max_iter=max_iter)

print("Оптимальні стратегії:")
print("Гравець 1:", p)
print("Гравець 2:", q)
print("Оптимальні виграші:")
print("Гравець 1:", ustar)
print("Гравець 2:", vstar)

input("Press Enter to continue...")