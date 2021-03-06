
# Tong Jiang 
This code is for the electronic structure of **Su-Schrieffer-Heeger Model**; The homework for Prof Shuai's *Theoretical Chemical Physics*


```python
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
```

The parameters come from Page 42 of *《高聚物中的孤子和极化子》*,
$t$ describes the interaction energy between NN atoms(eV);
$\alpha$ describes the rate for t changes with bond length(Angstrom); K is the elastic constant;
$N_e$ is the number of electrons; $N$ is the lattice length


```python
t = 2.5;alpha = 4.1;K = 21;Ne = 100
N = 101 # the lattice length
```

The eigenvalue equation writes,
$$-[t_0+(-1)^m\alpha(\phi_m+\phi_{m+1})]Z_{\mu,m+1}-[t_0+(-1)^{m-1}\alpha(\phi_{m-1}+\phi_{m})]Z_{\mu, m+1}=\varepsilon_{\mu}Z_{\mu,m}$$

Below, we note $\rm{coord\_add}[m-1]$ as $\phi_{m}+\phi_{m+1}$ , and  particularly, $\rm{coord\_add}[100]$ as $\phi_{100}+\phi_{1}$ to ensure the peoridical boundary condition

Combined with $$\phi_m+\phi_{m+1}=(-1)^m \frac{2\alpha}{K}[\sum_{\mu}'Z_{\mu, m}Z_{\mu, m+1}-\frac{1}{N}\sum_{m=1}^{N}\sum_\mu'Z_{\mu, m}Z_{\mu, m+1}]$$ to solve $\phi_m+\phi_{m+1}$ and $Z_{\mu}$ iteratively


```python
# Initialize the $\phi_{m}+\phi_{m+1}$
coord_add = np.random.rand(N-1,)*0.1 - 0.05; coord_add = coord_add.tolist()
temp = np.sum([coord_add[x-1]*(-1)**x for x in range(1, N)])
coord_add.append(-temp/((-1)**N)) # the boundary condition
```


```python
# The (SCF) iteration procedure.
iter_num = 0
while iter_num < 100:
    iter_num += 1
    # sub diagonal elements
    sub_diag = [-(t + (-1)**m * alpha * coord_add[m-1])
                for m in range(1, N)]
    # The matrix to be diagonized
    Mat = np.diag(sub_diag, k=-1) + np.diag(sub_diag, k=1)
    # The boundary condition
    Mat[0, -1] = -(t + (-1)**N * alpha * coord_add[-1])
    Mat[-1, 0] = -(t + (-1)**N * alpha * coord_add[-1])
    eigva, eigve = np.linalg.eigh(Mat)
    ZZ = []
    for i in range(1, N+1):
        if i != N:
            zz = [eigve[i-1, x] * eigve[i, x] for x in range(Ne//2)]
        else:
            zz = [eigve[0, x] * eigve[-1, x] for x in range(Ne//2)]
        ZZ.append(2 * np.sum(zz))
    # this is for the case of odd numbers of electrons, eg. N=101, Ne=101
    if Ne % 2 != 0:
        ZZ[:N-1] = [(ZZ[x] + eigve[x-1, Ne//2]*eigve[x, Ne//2])
                    for x in range(1, N)]
        ZZ[-1] = ZZ[-1] + eigve[0, Ne//2]*eigve[-1, Ne//2]
    ZZ_tot = np.sum(ZZ)
    for i in range(1, N+1):
        coord_add[i-1] = (-1)**(i) * 2 * alpha / K * (ZZ[i-1] - ZZ_tot / N)
```


```python
for i in range(N):
    plt.plot(range(2), [eigva[i]]*2, 'k')
plt.xticks([]);plt.ylabel('Energy Level(eV)');plt.show()
```


![png](output_9_0.png)


**The energy levels for $N=100$ and $N_e=101$** is shown above.


```python
coord_mean = [x/2 for x in coord_add]
plt.plot(coord_mean,linewidth=5.0);plt.ylabel('$\phi_n$');plt.xlabel('n');plt.show()
```


![png](output_11_0.png)


   **The order parameter $\phi_n^0+\phi_{n+1}^0$**

### Now we want to consider another case, what about using the open boundary condition?


```python
coord_add = [0] * (N-1)
iter_num = 0
while iter_num < 100:
    iter_num += 1
    sub_diag = [-(t + (-1)**m * alpha * coord_add[m-1])
                for m in range(1, N)]
    Mat = np.diag(sub_diag, k=-1) + np.diag(sub_diag, k=1)
    #Mat[0, -1] = -(t + (-1)**N * alpha * coord_add[-1])
    #Mat[-1, 0] = -(t + (-1)**N * alpha * coord_add[-1])
    eigva, eigve = np.linalg.eigh(Mat)
    ZZ = []
    for i in range(1, N):
        zz = [eigve[i-1, x] * eigve[i, x] for x in range(Ne//2)]
        ZZ.append(2 * np.sum(zz))
    if Ne % 2 != 0:
        ZZ[:N-2] = [(ZZ[x] + eigve[x-1, Ne//2]*eigve[x, Ne//2])
                    for x in range(1, N-1)]
        ZZ[-1] = ZZ[-1] + eigve[0, Ne//2]*eigve[-1, Ne//2]
    ZZ_tot = np.sum(ZZ)
    for i in range(1, N):
        coord_add[i-1] = (-1)**(i) * 2 * alpha / K * (ZZ[i-1] - ZZ_tot / (N-1))
for i in range(N):
    plt.plot(range(2), [eigva[i]]*2, 'k')
plt.xticks([]);plt.ylabel('Energy Level(eV)');plt.show()
```


![png](output_14_0.png)



```python
coeff = np.eye(N-1) + np.diag([1]*(N-2),k=1)
coord = np.linalg.solve(coeff, coord_add)
coord_mean = [x/2 for x in coord_add]
plt.scatter(range(N-1), coord)
plt.scatter(range(N-1), coord_mean);
plt.ylabel('$\phi_n$');plt.xlabel('n');plt.show()
```


![png](output_15_0.png)


Shown above is the order parameter, the blue line represents the displacement of one atom; 
and the yellow one is one half of the bond length.
