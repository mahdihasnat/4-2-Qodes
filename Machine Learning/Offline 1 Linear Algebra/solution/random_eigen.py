import numpy as np


def is_invertible_matrix(a):
	x = np.linalg.det(a)
	# print(type(x))
	return np.absolute(x)>np.finfo(type(x)).eps

def get_invertible_matrix(n):
	inf = 10
	while True:
		a = np.random.randint(low = -inf , high= inf, size = (n,n))
		if is_invertible_matrix(a):
			return a

def get_matrix_from_eigen(e_val, e_vec):
    return e_vec @ np.diag(e_val) @ np.linalg.inv(e_vec)
    

n=int(input("Enter n (dimension of matrix = n x n):"))
# n=100
a=get_invertible_matrix(n)
print("a=",a)
(e_val, e_vec) = np.linalg.eig(a)
print("Eigen vectors: ",e_vec)
print("Eigen values: ",e_val)
b=get_matrix_from_eigen(e_val, e_vec)
print("b=",b)
print("Is a and b same?:",np.allclose(a,b))