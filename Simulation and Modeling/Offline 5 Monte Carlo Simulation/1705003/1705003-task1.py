import numpy as np
from matplotlib import pyplot as plt

def get_uniform_random_permutation(n):
    perm = np.arange(n)
    i=1
    while i<n:
        j = np.random.randint(0,i)
        perm[i],perm[j] = perm[j],perm[i]
        i+=1
    return perm

def sample(n,m,s)->bool:
    
    perm = get_uniform_random_permutation(n)
    selected = None
    if m == 0:
        selected = perm[0]
    else:
        mn = np.min(perm[:m])
        for i in range(m,n):
            if mn>perm[i]:
                selected = perm[i]
                break
    if selected is None:
        selected = perm[-1]
    return selected<s

def prob(n,m,s,iter):
    tot=0
    for _ in range(iter):
        if sample(n,m,s):
            tot +=1
    return tot/iter

n=100
s=10
iter=1000
x = np.arange(0,n)
y = np.array([prob(n,m,s,iter) for m in x])
plt.plot(x,y)
plt.title("n={} s={} iteration={}".format(n,s,iter))
plt.xlabel("m")
plt.ylabel("P")
plt.show()