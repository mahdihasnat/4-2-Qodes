import numpy as np


def is_invertible_symmetric_matrix(a):
    # if not np.allclose(a, a.T):
    #     return False
    x = np.linalg.det(a)
    # print(type(x))
    return np.absolute(x)>np.finfo(type(x)).eps

def get_invertible_symmetric_matrix(n):
    inf = 10
    while True:
        # uncomment checking for symmetric matrix if generated otherwise
        a = np.random.randint(low = -inf , high= inf, size = (n,n))
        for i in range(0,n):
            for j in range(i+1,n):
                a[i][j]=a[j][i]
        if is_invertible_symmetric_matrix(a):
            return a

def get_symetrix_matrix_from_eigen(e_val, e_vec):
    return e_vec @ np.diag(e_val) @ e_vec.conjugate().T
    
def main(n):
    a=get_invertible_symmetric_matrix(n)
    print("A=",a)
    (e_val, e_vec) = np.linalg.eigh(a)
    print("Eigen vectors: ",e_vec)
    print("Eigen values: ",e_val)
    b=get_symetrix_matrix_from_eigen(e_val, e_vec)
    print("B=",b)
    print("Is A and B same?:",np.allclose(a,b))
    assert(np.allclose(a,b))

n=int(input("Enter n (dimension of matrix = n x n):"))
# n=5
main(n)