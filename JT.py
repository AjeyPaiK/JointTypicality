from scipy.stats import bernoulli
import commpy as comm
from collections import Counter
import numpy as np
import sys
import matplotlib.pyplot as plt
from math import log2, ceil, pow

def cook_codebook(M, n, p):
    bernoulli_sequences = []
    for i in range(0, M):
        sequence = bernoulli.rvs(size=n, p=p)
        sequence = tuple(sequence)
        bernoulli_sequences.append(sequence)
    # print("The sequences are:\n")
    # print(*bernoulli_sequences, sep="\n")
    return bernoulli_sequences

def joint_dist(Px, Pyx):
    return np.dot(Pyx,Px)

def YEE(y, n, p, M):
    P_yn = np.zeros((M,n), dtype = np.float32)
    E_Hy = []
    y = np.asarray(y)
    for i in range(0,M):
        for j in range(0,n):
            if y[i][j] == 0:
                P_yn[i][j] = p
            elif y[i][j] == 1:
                P_yn[i][j] = 1-p
    P_yn = np.apply_along_axis(np.log2, 0 , 1/P_yn)
    P_yn = np.sum(P_yn, axis=0)/n
    E_Hy.append(P_yn)
    return E_Hy

def CEE(codebook, n, p, M):
    P_xn = np.zeros((M,n), dtype = np.float32)
    E_Hx = []
    codebook = np.asarray(codebook)
    for i in range(0,M):
        for j in range(0,n):
            if codebook[i][j] == 0:
                P_xn[i][j] = p
            elif codebook[i][j] == 1:
                P_xn[i][j] = 1-p
    P_xn = np.apply_along_axis(np.log2, 0 , 1/P_xn)
    P_xn = np.sum(P_xn, axis=0)/n
    E_Hx.append(P_xn)
    return E_Hx

def true_entropy_y(Py):
    return float(Py[0]*log2(1/Py[0]) + Py[1]*log2(1/Py[1]))

def joint_entropy(Pxy):
    return Pxy[0][0]*log2(1/Pxy[0][0]) + Pxy[0][1]*log2(1/Pxy[0][1]) + Pxy[1][0]*log2(1/Pxy[1][0]) + Pxy[1][1]*log2(1/Pxy[1][1])

def entropy(p):
    H = p*(log2(1/p))+(1-p)*(log2(1/(1-p)))
    return H

def EJE(JD, codebook, y, n, M):
    P_xy = np.zeros((M,n), dtype = np.float32)
    E_Hxy = []
    codebook = np.asarray(codebook)
    for i in range(0,M):
        for j in range(0,n):
            if codebook[i][j] == 0 and y[i][j] == 0:
                P_xy[i][j] = JD[0]
            elif codebook[i][j] == 0 and y[i][j] == 1:
                P_xy[i][j] = JD[1]
            elif codebook[i][j] == 1 and y[i][j] == 0:
                P_xy[i][j] = JD[2]
            elif codebook[i][j] == 1 and y[i][j] == 1:
                P_xy[i][j] = JD[3]
    P_xy = np.prod(P_xy, axis = 1)
    return P_xy

C = 1 - entropy(0.4)
print("The Capacity of the channel is:"+ str(C) + "bits/s")
R = 0.5*C
print("The Rate chosen is = 0.5*C = "+ str(R) + "bits/s")
p = 0.4
q = 0.4
epsilon = 0.01
n = 550
print("Length of a sequence is =", n)
M = ceil(pow(2,n*R))
print("Total number of sequences in the codebook: ", M)
Pygx = np.array([[1-q, q],[q, 1-q]])
Px = np.array([p, 1-p])
Py = np.array([[p*(1-q) + (1-p)*q], [p*q + (1-p)*(1-q)]])
Hx = -p*log2(p) - (1-p)*log2(1-p) # True Entropy of Output
Hy = true_entropy_y(Py) # True Entropy of Output
Hxy = joint_entropy(Pygx) # True Joint entropy
JD = [(1-p)*(1-q), (1-p)*q, p*q, p*(1-q)]
codebook = cook_codebook(M, n, p)
print("\n")
# print("The Random Codebook is:\n")
# print(*codebook , sep='\n')
# ECE = CEE(codebook, n, p, M) # Empirical Entropy of codebook
y = []
for i in range(len(codebook)):
    a = np.asarray(codebook[i])
    output = comm.channels.bsc(a, q)
    # value = function(codebook[i], output)
    y.append(output)
print("\n")
# print("The Recieved output is:\n")
# print(*y , sep='\n')
# EEY = YEE(y,n,p,M)
EJoE = EJE(JD, codebook, y, n, M)
count = 0
right = 0
for i in range(len(EJoE)):
    if(EJoE[i] >= 2**((-n)*(Hxy + epsilon)) ) and ( EJoE[i] <= 2**((-n)*(Hxy - epsilon)) ) :
        right+=1
    else:
      count+=1
if count >=1:
    print("No.of failures = ", count)
    print("Percentage of failure = ", (count/M)*100 )
print("The number of sequences decoded successfully: ", right)
