import numpy as np


N = 10

s = np.zeros((N))
s[0] = 1
#s[1] = 1
s = 2*s-1

H = (np.sum(1+s)-2)**2
H = np.sum(np.outer(1+s, 1+s)) - 4*np.sum(1+s) + 4
H = np.sum(np.outer(s, s)) + 2*N*np.sum(s) - 4*np.sum(1+s) + 4 + N**2
H = np.sum(np.outer(s, s)) + 2*(N-2)*np.sum(s) + 4 + N**2 - 4*N

J = -2*np.ones((N,N))
J = J - np.diag(np.diag(J))
h = -np.ones(N)*2*(N-2)


H = -0.5*np.sum(J*np.outer(s, s)) - np.sum(s*h) + 4 + N**2 - 3*N

print(H)

print(J @ s + h)
