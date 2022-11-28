import numpy as np

def check_zero(x):
    return np.absolute(x)<np.finfo(type(x)).eps

def get_matrix(n,m):
    inf = 10    
    a = np.random.randint(low = -inf , high= inf, size = (n,m))
    return a

def moore_penrose_psedudoinverse(u,d,vt):
    """
        a= u @ d @ vt
        returns v @ d+ @ ut
    """
    d_plus = np.zeros((vt.shape[1], u.shape[0]))
    for i in range(d.shape[0]):
        if not check_zero(d[i]) :
            d_plus[i,i] = 1/d[i]
    return vt.T @ d_plus @ u.T
    

n,m=map(int, input("Enter dimension of matrix (seperated by single space) n m:").split())
# n,m=3,2
a = get_matrix(n,m)
print("a:",a)
try:
    u,d,vt = np.linalg.svd(a) # a= u @ d @ vt
    print("u:",u)
    print("d:",d)
    print("vt:",vt)
    a_pinv = np.linalg.pinv(a)
    print("a_pinv:",a_pinv)
    a_pinv_my = moore_penrose_psedudoinverse(u,d,vt)
    print("a_pinv_my:",a_pinv_my)
    print("a_pinv==a_pinv_my:",np.allclose(a_pinv, a_pinv_my))
except np.linalg.LinAlgError:
    print("SVD not possible")
    exit(0)
