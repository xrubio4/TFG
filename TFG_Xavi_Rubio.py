import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

#Unitats atòmiques
#Definir posicions
hbarr=1 #Esta en unitats atòmiques, si es vol canviar s'ha de fer a aqui
m0=1
rmass = 0.06
L = 70 #Ang
Lua = L*1.8897 #u.a
Lz= 10 #Ang
Lzua= Lz*1.8897
Vbarrier = 0.3 #eV
V0=Vbarrier *0.036749 #Hartree
q=1.6*10**(-19)
imag = complex(0,1)
N = 10 #Nombre de punts (Vigilar!!! posar un nombre petit ja que realment s'obtenen N^3 valors a la diagonal)
dx = 1 #Distància entre punts en x
dy = 1
dz = 1

print("Introdueix el valor del camp magnètic extern en unitats atòmiques:")
B=float(input())

X,Y,Z= np.linspace(-Lua/2,Lua/2,N), np.linspace(-Lua/2,Lua/2,N), np.linspace(-Lua/2,Lua/2,N)
I=np.identity(N)
#Operadors
def potencial(z):
    if (abs(z)<(Lz/2)):
        return 0
    else:
        return V0
#Operador energia cinètica en 1D, call de la funció D per a dx
def Dx(a):
    Hx = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            if i == j:
                Hx[i, j] = (hbarr**2 / (m0*rmass * a**2))  #Diagonal
            elif abs(i - j) == 1:
                Hx[i, j] = -(hbarr**2 / (m0*2 * rmass * a**2)) #Diagonals superior i inferior
    return Hx

#Operador energia cinètica en 1D, call de la funció D per a dy
def Dy(a):
    Hy = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            if i == j:
                Hy[i, j] = (hbarr**2 / (m0*rmass * a**2))  #Diagonal
            elif abs(i - j) == 1:
                Hy[i, j] = -(hbarr**2 / (m0*2 * rmass * a**2)) #Diagonals superior i inferior
    return Hy
#Operador energia cinètica en 1D + potencial pou quadrat, call de la funció Dz per a dz
def Dz(a):
    Hz = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            if i == j:
                Hz[i, j] = (hbarr**2 / (m0*rmass * a**2)) + potencial(i)  #Diagonal
            elif abs(i - j) == 1:
                Hz[i, j] = -(hbarr**2 / (m0*2 * rmass * a**2)) #Diagonals superior i inferior
    return Hz
#Tensor energia cinètica n^3 x n^3
T= np.kron(np.kron(Dx(dx),I),I) +np.kron(np.kron(I,Dy(dy)),I) + np.kron(np.kron(I,I),Dz(dz))
def Vmagn(a):
    V = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            if i == j:
                V[i, j] = 1/(2 * rmass*m0)*((q*B*X[i])**2)  #Diagonal
            elif abs(i - j) == 1:
                V[i, j] = np.sqrt(2*imag*hbarr*q*B*X[i]/(2*a) * np.conj(2*imag*hbarr*q*B*X[i]/(2*a)))  #Diagonals superior i inferior
    return V
############## REVISAR SI LA POSICIÓ ESTA INTRODUIDA CORRECTAMENT, el punt en el que estigu
#Tensor d'energia potencial n^3 x n^3
U= np.kron(np.kron(Vmagn(dx),I),I) +np.kron(np.kron(I,Vmagn(dy)),I) + np.kron(np.kron(I,I),Vmagn(dz))

Hamilt =T+U

eigenvalues, eigenvectors = np.linalg.eigh(Hamilt)

Energy=eigenvalues.reshape((N,N,N))

idx = Energy.argsort()[::-1]
Esort = Energy[idx]
vectors = eigenvectors[:,idx]
print(Esort)

