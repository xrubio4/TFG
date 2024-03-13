import numpy as np
from scipy.sparse.linalg import eigsh
import matplotlib.pyplot as plt
#Unidades sistema internacional
#Definir posicions
hbarr=6.626*10**(-34)
m0=9.11*10**(-31)
rmass = 0.067
Vbarrier = 10 #eV
V0 = Vbarrier * 1.6*10**(-19) # en Joules
q=-1.6*10**(-19) #Carga del electron en C

L = 552 #Ang (Valor encontrado en internet de pozo de AlGaAs/GaAs/AlGaAs)
Lua = L*10**(-10) #en metros
xylower = -Lua/2.0
xyupper = Lua/2.0
Lz = 119 #L pozo en Ang (Valor encontrado en internet de pozo de AlGaAs/GaAs/AlGaAs)
Lzua = Lz*10**(-10) #en metros
zlower = -Lua/2.0
zupper = Lua/2.0 #Analizaremos la distancia Lua en Z, para analizar puntos dentro y fuera del pozo
dx = 50*10**-10 #Distància entre puntos en x en m
dy = 50*10**-10
dz = 50*10**-10
X = np.arange(xylower,xyupper+dx,dx)
Y = np.arange(xylower,xyupper+dy,dy)
Z = np.arange(zlower,zupper+dz,dz)


N=len(X) #Numero de puntos, hay que vigilar porque realmente se obtienen N^3 valores propios

##CÓDIGO DEPENDENCIA CON EL ÁNGULO
print("Introduce el valor del campo magn (B) en T:")
B=float(input())
Theta = np.linspace(0,np.pi,180)

I=np.identity(N) #Matriz identidad
#Operadores
#Potencial pozo cuadrado en Z
def potencial(z):
    if (abs(z)<(Lzua/2)):
        return 0
    else:
        return V0
#Operador T en 1D, hacer call de la funcion Dx para dx
def Dx(a):
    Hx = np.zeros((N, N),dtype=complex)
    for i in range(N):
        for j in range(N):
            if i == j:
                Hx[i, j] = (hbarr**2 / (m0*rmass * a**2))  #Diagonal
            elif abs(i - j) == 1:
                Hx[i, j] = -(hbarr**2 / (m0*2 * rmass * a**2)) #Diagonals superior e inferior
    return Hx

#Operador T en 1D, hacer call de la funcion Dy para dy
def Dy(a):
    Hy = np.zeros((N, N),dtype=complex)
    for i in range(N):
        for j in range(N):
            if i == j:
                Hy[i, j] = (hbarr**2 / (m0*rmass * a**2))  #Diagonal
            elif abs(i - j) == 1:
                Hy[i, j] = -(hbarr**2 / (m0*2 * rmass * a**2)) #Diagonals superior i inferior
    return Hy
#Operador T en 1D + potencial pou quadrat, hacer call de la funcion Dz para dz
def Dz(a):
    Hz = np.zeros((N, N),dtype=complex)
    for i in range(N):
        for j in range(N):
            if i == j:
                Hz[i, j] = (hbarr**2 / (m0*rmass * a**2)) + potencial(Z[i])  #Diagonal
            elif abs(i - j) == 1:
                Hz[i, j] = -(hbarr**2 / (m0*2 * rmass * a**2)) #Diagonals superior e inferior
    return Hz
#Tensor T n^3 x n^3
T= np.kron(np.kron(Dx(dx),I),I) +np.kron(np.kron(I,Dy(dy)),I) + np.kron(np.kron(I,I),Dz(dz))
def Vmagnz(a,thet):
    V = np.zeros((N, N),dtype=complex)
    for i in range(N):
        for j in range(N):
            if i == j:
                V[i, j] = 1/(2 * rmass*m0)*((q*B*X[i])**2)  #Diagonal
            elif (i - j) == 1:
                V[i, j] = complex(0,-2*hbarr*q*B*X[i]*np.cos(thet)/(2*a)) #Diagonal superior
            elif (i - j) == -1:
                V[i, j] = complex(0,2*hbarr*q*B*X[i]*np.cos(thet)/(2*a)) #Diagonal inferior
    return V

def Vmagny(a,thet):
    P = np.zeros((N, N),dtype=complex)
    for i in range(N):
        for j in range(N):
            if i == j:
                P[i, j] = 1/(2 * rmass*m0)*((q*B*X[i])**2)  #Diagonal
            elif (i - j) == 1:
                P[i, j] = complex(0,2*hbarr*q*B*X[i]*np.sin(thet)/(2*a)) #Diagonals superior e inferior
            elif (i - j) == -1:
                P[i, j] = complex(0,-2*hbarr*q*B*X[i]*np.sin(thet)/(2*a)) #Diagonals superior e inferior
    return P

k0 = []
k1 = []
k2 = []
for i in (Theta):
#Tensor d'energia potencial n^3 x n^3
    U= np.kron(np.kron(I,Vmagny(dy,i)),I) + np.kron(np.kron(I,I),Vmagnz(dz,i))
    Hamilt =T+U #Hamiltoniano
    eigenvalues , eigenvectors = eigsh(Hamilt, k=3,which="SM")
    eV=eigenvalues/q
    k0.append(eV[0])
    k1.append(eV[1])
    k2.append(eV[2])
plt.plot(np.degrees(Theta),k0)
plt.title ("E(k = 0)(eV) vs Theta(º)")
plt.xlabel("Theta(º)")
plt.ylabel("E(k = 0)(eV)")
plt.show()
plt.plot(np.degrees(Theta),k1)
plt.title ("E(k = 1)(eV) vs Theta(º)")
plt.xlabel("Theta(º)")
plt.ylabel("E(k = 1)(eV)")
plt.show()
plt.plot(np.degrees(Theta),k2)
plt.title ("E(k = 2)(eV) vs Theta(º)")
plt.xlabel("Theta(º)")
plt.ylabel("E(k = 2)(eV)")
plt.show()
