import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

#Unidades sistema internacional
#Definir posicions
hbarr=6.626*10**(-34) 
m0=9.11*10**(-31)
rmass = 0.06
L = 70 #Ang
Lua = L*10**(-10) #en metros
Lz= 10 #Ang
Lzua = Lz*10**(-10) #en metros
Vbarrier = 10 #eV
V0 = Vbarrier * 1.6*10**(-19) # en Joules
q=1.6*10**(-19) #Carga del electron en C
imag = complex(0,1)
N = 10 #Numero de puntos, hay que vigilar porque realmente se obtienen N^3 valores propies
dx = 10*10**-10 #Distància entre puntos en x en m
dy = 10*10**-10
dz = 10*10**-10

print("Introduce el valor del campo magnético en T:")
B=float(input())

X,Y,Z= np.linspace(-Lua/2,Lua/2,N), np.linspace(-Lua/2,Lua/2,N), np.linspace(-Lua/2,Lua/2,N) #Para graficar y tambien para uso en el potencial

I=np.identity(N) #Matriz identidad
#Operadores
#Potencial pozo cuadrado en Z
def potencial(z):
    if (abs(z)<(Lzua/2)):
        return 0
    else:
        return V0
#Operador energia cinética en 1D, hacer call de la funcion Dx para dx
def Dx(a):
    Hx = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            if i == j:
                Hx[i, j] = (hbarr**2 / (m0*rmass * a**2))  #Diagonal
            elif abs(i - j) == 1:
                Hx[i, j] = -(hbarr**2 / (m0*2 * rmass * a**2)) #Diagonals superior e inferior
    return Hx

#Operador energia cinética en 1D, hacer call de la funcion Dy para dy
def Dy(a):
    Hy = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            if i == j:
                Hy[i, j] = (hbarr**2 / (m0*rmass * a**2))  #Diagonal
            elif abs(i - j) == 1:
                Hy[i, j] = -(hbarr**2 / (m0*2 * rmass * a**2)) #Diagonals superior i inferior
    return Hy
#Operador energia cinética en 1D + potencial pou quadrat, hacer call de la funcion Dz para dz
def Dz(a):
    Hz = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            if i == j:
                Hz[i, j] = (hbarr**2 / (m0*rmass * a**2)) + potencial(i)  #Diagonal
            elif abs(i - j) == 1:
                Hz[i, j] = -(hbarr**2 / (m0*2 * rmass * a**2)) #Diagonals superior e inferior
    return Hz
#Tensor energia cinética n^3 x n^3
T= np.kron(np.kron(Dx(dx),I),I) +np.kron(np.kron(I,Dy(dy)),I) + np.kron(np.kron(I,I),Dz(dz))
def Vmagn(a):
    V = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            if i == j:
                V[i, j] = 1/(2 * rmass*m0)*((q*B*X[i])**2)  #Diagonal
            elif abs(i - j) == 1:
                V[i, j] = np.sqrt(2*imag*hbarr*q*B*X[i]/(2*a) * np.conj(2*imag*hbarr*q*B*X[i]/(2*a)))  #Diagonals superior e inferior
    return V

#Tensor d'energia potencial n^3 x n^3
U= np.kron(np.kron(Vmagn(dx),I),I) +np.kron(np.kron(I,Vmagn(dy)),I) + np.kron(np.kron(I,I),Vmagn(dz))

Hamilt =T+U #Hamiltoniano

eigenvalues, eigenvectors = np.linalg.eigh(Hamilt) 
Energy=eigenvalues.reshape((N,N,N))

idx = np.argsort(eigenvalues)
Esort = eigenvalues[idx]
EsorteV = Esort/q #Energia en eV
vectors = eigenvectors[:,idx]
print(EsorteV)

