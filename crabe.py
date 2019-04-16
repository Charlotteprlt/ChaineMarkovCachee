##imports

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

##construct L

Y = np.loadtxt("crabe.txt")

X = np.linspace(0.582,0.694,num=29)
L=[]

for i in range(Y.size):
    for k in range (int(Y[i])):
        L.append(X[i])
L=np.array(L)


## algorithme EM

def EM(it,pi_0,mu_0,sigma_0,L,pop):
    pi=pi_0
    mu=mu_0
    sigma=sigma_0
    N=len(L)
    for iter in range(it):

        norm_rho = np.sum(norm.pdf(L,mu[i],sigma[i])*pi[i] for i in range(pop)) 
        rho = np.array([pi[i]*norm.pdf(L,mu[i],sigma[i]) for i in range(pop)]) / norm_rho 
        somme_rho = np.array([np.sum(rho[i]) for i in range(pop)])
        
        pi = somme_rho/N
        mu = np.array([np.sum(rho[i]*L) for i in range(pop)])/somme_rho
        sigma = np.sqrt(np.array([np.sum(rho[i]*(L-mu[i])**2) for i in range(pop)])/somme_rho)
        
    return pi,mu,sigma
    

## 2 populations

it= 1000
pi_0 = np.array([0.25,0.75])
mu_0 = np.array([0.57,0.67])
sigma_0 = np.array([1,1])/100

pi_2,mu_2,sigma_2 = EM(it,pi_0,mu_0,sigma_0,L,2)

## 3 populations

it= 10000
pi_0 = np.array([0.1,0.3,0.6])
mu_0 = np.array([0.57,0.67,0.62])
sigma_0 = np.array([1,1,1])/100

pi_3,mu_3,sigma_3 = EM(it,pi_0,mu_0,sigma_0,L,3)

## affichage

cov=np.eye(pop)*sigma

Gauss = norm.pdf(L,np.mean(L),np.sqrt(np.var(L)))
Gauss2 = np.sum(np.array([norm.pdf(L,mu_2[i],sigma_2[i])*pi_2[i] for i in range(2)])[i] for i in range(2))
Gauss3 = np.sum(np.array([norm.pdf(L,mu_3[i],sigma_3[i])*pi_3[i] for i in range(3)])[i] for i in range(3))

plt.hist(L,bins=29,normed=1,color="blue")
p1, = plt.plot(L,Gauss,color="green",label = "1 gaussienne")
p2, = plt.plot(L,Gauss2,color="red",label="2 gaussiennes ")
p3, = plt.plot(L,Gauss3,color="black",label = "3 gaussiennes")
plt.legend(handles=[p1,p2,p3])
plt.title("histogramme ainsi que son approximation par 1,2 et 3 gaussiennes")
plt.show()











