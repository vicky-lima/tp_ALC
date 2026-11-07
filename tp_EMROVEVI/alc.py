import numpy as np
import os
############################################################################################
# Funciones del módulo
############################################################################################

# FUNCIÓN TRASPONER 
def trasponer(A):
    m = A.shape[0]
    n = A.shape[1]
    traspuesta = []
    for i in range(n):
        fila = []
        for j in range(m):
            fila.append(A[j][i])
        traspuesta.append(fila)
    return np.array(traspuesta)


# FUNCIÓN QUE CALCULA EL VALOR ABSOLUTO 
def valorAbsoluto(n):
    if n<0:
        return n*(-1)
    return n

# FUNCIÓN QUE MULTIPLICA DOS MATRICES 
def multiplicar_matrices(A, B):

    res = []
    for filaA in A:
        f = []
        for c in range(B.shape[1]):  
            x = 0
            for i in range(filaA.size):
                x += filaA[i] * B[i][c]
            f.append(x)
        res.append(f)
    return np.array(res)


# FUNCIÓN QUE CALCULA LA NORMA P DE UN VECTOR X 
def norma(x, p): 
    res = 0

    if p == "inf": 
        vals_abs = [valorAbsoluto(elem) for elem in x]
        return max(vals_abs)


    for elem in x: 
        elem_mod = valorAbsoluto(elem)
        elem_mod**=p
        res += elem_mod
    
    res**=(1/p)

    return res

#FUNCIÓN QUE CALCULA EL PRODUCTO ENTRE MATRIZ A Y VECTOR X 
def calcularAx(A,x):
    b = []
    for i in range(A.shape[0]):
        suma = 0
        for j in range(A.shape[1]):
            suma += A[i][j] * x[j]
        b += [suma]

    return np.array(b)


#FUNCIÓN QUE CALCULA EL PRODUCTO INTERNO ENTRE DOS VECTORES     
def producto_interno(u, v):
    res = 0
    for i in range(u.size):
        res += u[i] * v[i]
    return res

#FUNCIÓN QUE CALCULA LA NORMA MATRICIAL INFINITO DE A 
def normaMatInf(A): 
    
        fila_max= 0
        for fila in A:
             norma_fila = 0

             for elem in fila: 
                  norma_fila += valorAbsoluto(elem)

             if norma_fila > fila_max: 
                  fila_max = norma_fila

        return fila_max

#FUNCIÓN QUE CALCULA LA NORMA MATRICIAL UNO  DE A 

def normaMatUno(A):
    return normaMatInf(trasponer(A))

# FUNCION QUE CALCULA LA NORMA EXACTA P DE UNA MATRIZ A 
def normaExacta(A, p=[1, 'inf']):
    if isinstance(p, int):
        if p == 1:
            return normaMatUno(A)
        else:
            return None
    elif isinstance(p, str):
        if p == "inf":
            return normaMatInf(A)
        else:
            return None
    else:
        if p[0] == 1:
            return [normaMatUno(A), normaMatInf(A)]
        else:
            return None

# FUNCIÓN QUE CALCULA EL PRODUCTO EXTERNO ENTRE DOS VECTORES 
def producto_externo(u, v):
    n = u.size
    m = v.size
    M = np.zeros((n, m))
    for i in range(n):
        for j in range(m):
            M[i][j] = u[i] * v[j]
    return M

def producto_vectorial(a,b): #a x b para vectores de 3 componentes
    return np.array([a[1]*b[2] - a[2]*b[1],
                     a[2]*b[0] - a[0]*b[2],
                     a[0]*b[1] - a[1]*b[0]])

# 
def fA(A, v):
   Av = calcularAx(A, v)
   w = 0

   if norma(Av, 2) != 0:
      w = Av / norma(Av, 2)

   return w

def fkA (A, v, K):
    k = 0
    while k<K: 
        v = fA (A, v)
        k += 1

    return v
     

# FUNCIÓN QUE UTILIZA EL MÉTODO DE LA POTENCIA 
def metpot2k(A,tol=1e-15,K=1000): 
    n = A.shape[0]
    v = np.random.uniform(0, 1, n)
    w = fkA(A, v, 2)
    e = producto_interno(w, v)
    k = 0

    while valorAbsoluto(e-1) > tol and k<K: 
        v = w
        w =fkA(A, v, 2)
        e = producto_interno(w, v)
        k += 1

    l = producto_interno(w, calcularAx(A, w))
    ep = e - 1

    return v, l, k

#FUNCIÓN QUE RECORTA UNA MATRIZ ENRTRE LOS VALORES N Y M DESEADOS --> creo que se puede reemplazar  x slicing 
def recortar_matriz (A, n, m): 
    B = []

    for i in range (m-n): 
        v = np.zeros((m-n))
        for j in range(m-n): 
            v[j] = A[i + n][j+n]
        B.append(v)

    return np.array(B)

#FUNCION QUE DIAGONALIZA A LA MATRIZ A 
def diagRH(A,tol=1e-15,K=1000): 
    n = A.shape[0]
    m = A.shape[1]

    if n != m: 
        return None
    
    v1, l1, _ = metpot2k(A, tol, K)
    w = np.eye(n)[0] - v1
    Hv1 = np.eye(n) - 2* (producto_externo(w, w/(norma(w, 2)**2)))

    if n == 2: 
        S = Hv1
        D = multiplicar_matrices(Hv1, multiplicar_matrices(A, trasponer(Hv1)))
    else: 

        B =  multiplicar_matrices(Hv1, multiplicar_matrices(A, trasponer(Hv1)))
        C = recortar_matriz(B, 1, n)
        S2, D2 = diagRH(C, tol, K)
        D = np.zeros((n,n))
        D [0][0] =  l1
        for i in range(1, n): 
            for j in range(1,n): 
                D[i][j] = D2 [i-1][j-1]
        
        S3 = np.zeros((n,n))
        S3[0][0] =  1
        for i in range(1, n): 
            for j in range(1,n): 
                S3[i][j] = S2 [i-1][j-1]
        S = multiplicar_matrices(Hv1, S3)

    return S, D

# FUNCION QUE NORMALIZA A UN ARRAY X EN NORMA P 
def normaliza(X,p):
    n = X.shape[0]
    y = []
    for i in range(n):
        m = X[i].size
        fila_normalizada = np.zeros(m)
        for j in range(m):
            elemento = X[i][j]
            elemento_normalizado = elemento/norma(X[i],p)
            
            fila_normalizada[j] = elemento_normalizado
        y += [fila_normalizada]

    return np.array(y)

# FUNCION DESCOMP SVD REDUCIDA: 
def svd_reducida_aux(A, k="max",  tol=1e-15):
    A_t = trasponer(A)  
    H = multiplicar_matrices(A_t, A)
    S, D = diagRH(H, tol, K=1000)

    S_t = trasponer(S)
    m = D.shape[0]
    n = D.shape[1]

    autovalores = []
    autovectores = []
    for i in range(m):
        if valorAbsoluto(D[i][i]) > tol :
            autovalores += [D[i][i]]
            autovectores += [S_t[i]]
   
    if k == "max":
        limite = len(autovalores)
    else:
        limite = min(k, len(autovalores))

    autoval_ordenados = []
    autovect_ordenados = []

    while len(autovalores) > 0 and len(autoval_ordenados) < limite:
        idx_max = 0
        for i in range(1, len(autovalores)):
            if autovalores[i] > autovalores[idx_max]:
                idx_max = i

        autoval_ordenados.append(autovalores[idx_max])
        autovect_ordenados.append(autovectores[idx_max])
        autovalores.pop(idx_max)
        autovectores.pop(idx_max)

    V = trasponer(np.array(autovect_ordenados))   
    autoval_ordenados = np.array(autoval_ordenados)
    r = autoval_ordenados.size
    E = np.zeros((r, r))
    for i in range(r):
        E[i][i] = autoval_ordenados[i] ** 0.5

    B = multiplicar_matrices(A, V)
    B_t = trasponer(B)
    U_t = normaliza(B_t, 2)
    U = trasponer(U_t)

    return U, E, V
    

def svd_reducida(A, k="max", tol = 1e-15):
    m = A.shape[0]
    n = A .shape[1]

    if m >= n:
        U, E, V = svd_reducida_aux(A, k, tol)
    else:
        A_t = trasponer(A)
        V, E, U = svd_reducida_aux(A_t, k, tol)

    return U, E, V

#FACTORIZACION QR CON GS
def QR_con_GS(A,tol=1e-12,retorna_nops=False):
    """
    A una matriz de n x n 
    tol la tolerancia con la que se filtran elementos nulos en R
    retorna_nops permite (opcionalmente) retornar el numero de operaciones realizado
    retorna matrices Q y R calculadas con Gram Schmidt (y como tercer argumento opcional, el numero de operaciones).
    Si la matriz A no es de n x n, debe retornar None
    """
    n, m = np.shape(A)
    if n != m:
        return False
    else:
        R = np.zeros(shape=(n,n))
        q_columnas = []
        A_traspuesta = trasponer(A) #la traspongo para que hacer A[n] me de la columna y no la fila
        a1 = A_traspuesta[0]
        q1 = a1/norma(a1,2)
        q_columnas += [q1]
        R[0,0] = norma(a1,2)
        for i in range(n-1):
            i += 1
            ui = np.array(A_traspuesta[i])

            for k in range(i):
                qk = q_columnas[k]
                rki = producto_interno(qk, ui)
                R[k][i] = rki
                ui = ui - rki * qk
            #wi = ai - proyeccion(q1,ai)
            #q1 = wi/m.norma(wi,2)
        
            R[i][i] = norma(ui,2)
            qi = ui/norma(ui,2)
            q_columnas += [qi]

        q_columnas_array = np.array(q_columnas)
        Q = trasponer(q_columnas_array)
        return Q, R

# CALCULA LU

def calculaLU(A):
    nops = 0
    if A is None:
        return None, None, 0

    A = np.array(A, dtype=float)
    Ac = A.copy()
    m,n = A.shape
        
    if m!=n:
        return None, None, 0

    
    for j in range(n-1):
        elem_diag = Ac[j][j]
        if valorAbsoluto(elem_diag) < 1e-15:
            return None, None, 0
        else:
            i = j+1
            #nops += n-j-1
            while(i != m):
                multiplicador = Ac[i][j]/elem_diag
                nops += 1
                Ac[i][j] = multiplicador
                #nops += 2*(n-j)

                p = j + 1
                while(p != n):
                    Ac[i][p] = (Ac[i][p]) - (multiplicador * Ac[j][p])
                    nops += 2
                    p += 1
                #Ac[i, j + 1:n] -= multiplicador * Ac[j, j + 1:n]

                i += 1

    L = np.eye(m)
    U = np.zeros((m,n))
    for i in range(m):
        for j in range (n):
            if i>j:
                L[i][j] = Ac[i][j]
            else:
                U[i][j] = Ac[i,j]
    return L, U, nops

#RES_TRI

def res_tri(L,b,inferior=True):
    if L is None or b is None:
        return None

    n, m = np.shape(L)
    x = np.zeros(n)

    if inferior == True:
        i = 0
        while (i != n):
            j = 0
            sumat_variables_conocidas = 0
            if valorAbsoluto(L[i][i]) < 1e-10:
                return None
            while (j!=i):
                elemento = L[i][j]
                sumat_variables_conocidas += elemento * x[j]
                j+=1

            x[i] = (b[i] - sumat_variables_conocidas)/L[i][i]
            i += 1
    elif (inferior == False):
        i = n-1
        while (i >= 0):
            j = n-1
            sumat_variables_conocidas = 0
            if L[i][i] == 0:
                return None

            while (j>i):
                elemento = L[i][j]
                sumat_variables_conocidas += elemento * x[j]
                j-=1

            x[i] = (b[i] - sumat_variables_conocidas)/L[i][i]
            i -= 1

    return x

#INVERSA CON LU

def inversa(A):
    L,U, cant_op = calculaLU(A)
    if (L is None):
        return None
    else:
        n = A.shape[0]
        inversa_sin_t = []

        #Cada col j de A^-1 cumple A.(A^-1)j = ej
        #Entonces resuelvo: L Yj = ej
        for j in range(n):
            ej = np.zeros(n)
            ej[j] = 1
            
            columnaYj = res_tri(L,ej,inferior=True)
            if columnaYj is None:
                return None
        
            columnajInv = res_tri(U,columnaYj,inferior=False)
            if columnajInv is None:
                return None
            inversa_sin_t.append(columnajInv)

        inversa_sin_t_array = np.array(inversa_sin_t)
        inversa = np.array(trasponer(inversa_sin_t_array))
        return inversa

# CALCULA LDV

def calculaLDV(A):
    L, U, cant_ops = calculaLU(A)
    if (L is None):
        return None
    n, m = L.shape
    #me copio D de U 
    D = np.zeros((n,n))
    for i in range(n):
        for j in range(m):
            if i == j:
                if U[i][j] == 0:
                    return None
                D[i][j]= U[i][j]

    V = multiplicar_matrices(inversa(D), U)
    
    return L, D, V


# ES DEFINIDA POSITIVA

def esSDP(A, atol=1e-8):
    if A is None:
        return False

    A = np.array(A, dtype=float)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        return False

    resultado = calculaLDV(A)
    if resultado is None:
        return False

    L, D, V = resultado
    if not np.allclose(V, trasponer(L), atol=atol):
        return False

    n = A.shape[0]
    for i in range(n):
        if D[i][i] <= 1e-16:
            return False

    return True

# ES SIMETRICA

def esSimetrica(A, atol=1e-8):
    A_t = trasponer(A)

    if A.shape != A_t.shape:
        return False

    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            if valorAbsoluto(A_t[i][j] - A[i][j]) > atol:
                return False

    return True

# CHOLESKY

def calculaCholesky(A,atol=1e-10):

    n,m = A.shape
    if not esSDP(A, atol) or not esSimetrica(A):
        return None

    L, U, nops = calculaLU(A)
    D = np.zeros((n,m))

    for i in range(n):
        D[i][i] = (U[i][i]**0.5)

    L_t = trasponer(L)
    L_cholesky_t = multiplicar_matrices(D, L_t)

    L_cholesky = trasponer(L_cholesky_t)

    return L_cholesky


# PSEUDOINVERSA 
def pseudoinversa(A):
    U, E, V = svd_reducida(A)
    U_t = trasponer(U)
    E_t = trasponer(E)
    E_pseudoinv = E_t
    i=0
    while i < E.shape[0] and valorAbsoluto(E[i][i]) > 1e-15:
        E_pseudoinv[i][i] = 1/E_pseudoinv[i][i]
        i+= 1
    aux = multiplicar_matrices(V, E_pseudoinv)
    Res = multiplicar_matrices(aux,U_t)
    return Res


######################################################################################################################################################################################################
# Funciones de la consigna
######################################################################################################################################################################################################

#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#EJ 2

# Resuelve una Q 
def obtenerQ3d(imgPts, objPts): # asumiendo que recibimos dos arrays 
    n = objPts.shape[0] # como sabemos el tipo de datos que nos pasan????
    Q = np.zeros((2*n, 12))

    for i in range(n): 
        U= objPts[i][0]
        V= objPts[i][1]
        W= objPts[i][2]
        u = imgPts[2*i]
        v = imgPts[2*i +1]  

        Q[2*i] = np.array([U, V, W, 1, 0, 0, 0, 0, -u*U, -u*V, -u*W, -u])
        Q[2*i + 1] = np.array([0, 0, 0, 0, U, V, W, 1, -v*U, -v*V, -v*W, -v ])

    return Q 

# docstring 
def resolverDLT3D(Q): 
    U, E, V = svd_reducida(Q)
    V_t = trasponer(V)
    n = V_t.shape[0]
    m = V_t[n-1]

    M = m.reshape(3, 4)

    return M 

# Sesgo 1 es el 0, y sesgo 2 es alfa u = alfa v
# res_tri resuelve Ax = b, Kt = m
def devolverMatricesCalibracion3D(M, sesgo1=False, sesgo2=False):
    KR = M[:,:3]
    KR_inversa = inversa(KR)
    
    R_t, K_inversa = QR_con_GS(KR_inversa)
    R = trasponer(R_t)
    K = inversa(K_inversa)

    if sesgo1 == True:
        K[0][1] = 0

    if sesgo2 == True:
        K[1][1] = K[0][0] #nos quedamos con alfa 1

    #Reescribimos R en el caso de que haya entrado a alguno de los if
    R = multiplicar_matrices(K_inversa, KR)

    columna_m = M[:,3]
    t = res_tri(K,columna_m)

    return R, t, K

#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#EJ 3)

def obtenerQ2d(imgPts, objPts): # asumiendo que recibimos dos arrays 
    n = objPts.shape[0]
    Q = np.zeros((2*n, 9))

    for i in range(n): 
        U= objPts[i][0]
        V= objPts[i][1]
        u = imgPts[i][0]
        v = imgPts[i][1]  

        Q[2*i] = np.array([U, V, 1, 0, 0, 0, -u*U, -u*V, -u])
        Q[2*i + 1] = np.array([0, 0, 0, U, V, 1, -v*U, -v*V, -v ])

    return Q 


#docstring
def obtener_H(Q):
    U, E, V = svd_reducida(Q)
    V_t = trasponer(V)
    n = V_t.shape[0]
    h = V_t[n-1]

    H = h.reshape(3, 3)


    return np.array(H) 


#docstring
def armarA(Hlst):
    n = len(Hlst)
    A = []

    for h in Hlst:
        vij_aux = []
        for i, j in [(0,0),(0,1),(1,1)]:
            vij = [h[0][i]*h[0][j],
                   h[0][i]*h[1][j] + h[1][i]*h[0][j],
                   h[1][i]*h[1][j],
                   h[2][i]*h[0][j]+h[0][i]*h[2][j],
                   h[2][i]*h[1][j]+ h[1][i]*h[2][j],
                   h[2][i]*h[2][j]]

            vij_aux += [vij]

        vij_aux = np.array(vij_aux)
        fila_1 = vij_aux[1]
        fila_2 = vij_aux[0] - vij_aux[2]

        A += [fila_1]
        A += [fila_2]

    A = np.array(A)
    
    return A

def obtener_B(A, sesgo1 = False): 
    #resolvemos Ab = 0 con ||b|| = 1: b es el vector sing de menor valor singular de A.
    if sesgo1 == True:
        A_utilizada = np.hstack((A[:, 0:1], A[:, 2:6]))
    else:
        A_utilizada = A  

    n_cols = A_utilizada.shape[1]

    escala = np.zeros(n_cols)
    for c in range(n_cols): #normlizamos columnas A
        escala[c] = norma(A_utilizada[:, c],2)

    A_esc = A_utilizada / escala

    U,E,V = svd_reducida(A_esc)
    V_t = trasponer(V)
    n = V_t.shape[0]
    
    #ponemos a b con norma = 1
    b_esc = V_t[n-1]

    b = b_esc / escala

    if sesgo1 == True:
        b = np.array([b[0], 0, b[1], b[2], b[3], b[4]])

    
    B = [[b[0], b[1], b[3]],
         [b[1], b[2], b[4]],
         [b[3], b[4], b[5]]]

    B = np.array(B)

    if B[0][0] < 0:
        B = -B
    
    return B

def calcularMatrizK2D(A):
    B = obtener_B(A, sesgo1=True)
    K_inversa_t = calculaCholesky(B)
    K = trasponer(inversa(K_inversa_t))
    K = K/K[2][2]

    return K

def obtenerPose2D(Hj, K):
    K_inversa = inversa(K)
    c1 = Hj[:,0]
    c2 = Hj[:,1]
    c3 = Hj[:,2]

    K_inv_c1 = calcularAx(K_inversa,c1)
    K_inv_c2 = calcularAx(K_inversa,c2)
    K_inv_c3 = calcularAx(K_inversa, c3)

    lamb = 1/norma(K_inv_c1,2)

    r1 = lamb * K_inv_c1
    r2 = lamb * K_inv_c2
    r3 = producto_vectorial(r1, r2)

    R_traspuesta = np.array([r1,r2,r3])
    R = trasponer(R_traspuesta)

    t = lamb * K_inv_c3

    return R, t

#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#EJ 4)

def circulo_de_obj_a_img(H, centro, radio=1.0,n=100):
    theta = np.arange(n+1)*2*np.pi/n
    a = centro[0] + radio*np.cos(theta)
    b = centro[1] + radio*np.sin(theta)

    puntos = np.vstack([a,b,np.ones(n+1)])
    X = []
    for punto in trasponer(puntos):
        X_aux = calcularAx(H, punto) #pasamos de la realidad a la img
        X += [[X_aux[0]/X_aux[2],X_aux[1]/X_aux[2]]]
    
    X = np.array(X)

    return X

#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# EJ 5) 
def normalizarPuntos(P):
    centroide = np.mean(P, axis=0)
    diferencia = P - centroide
    dist_media = np.mean(np.sqrt(np.sum(diferencia * diferencia, axis=1)))
    s = np.sqrt(2) / dist_media

    return np.array([[s, 0, -s * centroide[0]], [0, s, -s * centroide[1]], [0, 0, 1.0]])

def aplicarT(T, P):
    P_n = np.zeros((P.shape[0], 2))
    for i in range(P.shape[0]):
        p = calcularAx(T, np.array([P[i][0], P[i][1], 1.0]))
        P_n[i][0] = p[0]
        P_n[i][1] = p[1]

    return P_n

def obtener_H_normalizada(imgPts, objPts):
    T_img = normalizarPuntos(imgPts)
    T_obj = normalizarPuntos(objPts)
    H_n = obtener_H(obtenerQ2d(aplicarT(T_img, imgPts), aplicarT(T_obj, objPts)))
    H = multiplicar_matrices(inversa(T_img,), multiplicar_matrices(H_n, T_obj))

    return H / H[2][2]

def calcular_f(A, u0, v0):
    a0 = A[:, 5]
    a1 = A[:, 0] + A[:, 2] - u0 * A[:, 3] - v0 * A[:, 4] + (u0**2 + v0**2) * A[:, 5]
    x = -np.sum(a0 * a1) / np.sum(a1 * a1)
    if x <= 0:
        return None
    return 1 / np.sqrt(x)


#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# EJ 6)

def calculoLambdaMu(I0,J0): 
    p = -np.sum(I0*J0)
    d = np.sum(J0*J0) - np.sum(I0*I0)
    s = (d+ np.sqrt(d**2+4*p**2))/2
    lamb = np.sqrt(s)
    if lamb > 1e-15:
        mu = p/lamb
    else:
        mu = np.sqrt(valorAbsoluto(d))
    return lamb, mu

#Cuando decimos fx, es porque nos referimos al parametro horizontal de f, K[0][0]
def calcularSalidaPOSIT(Is, Js, fx, m0, M0):

    Z0 = fx / norma(Is,2)
    i = Is / norma(Is, 2)
    j = Js / norma(Js, 2)
    k = producto_vectorial(i, j)
    R = np.vstack([i,j,k])
    t = np.array([(Z0/fx)*m0[0], (Z0/fx)*m0[1], Z0])

    return i, j, k, Z0, R, t

def evaluarPose(R, t, obj_vecs, img_vecs, K):

    R_traspuesta = trasponer(R)
    pose = multiplicar_matrices(obj_vecs, R_traspuesta) + t
    X = pose[:,0]
    Y = pose[:,1]
    Z = pose[:,2]

    cantidad_factibles = 0
    img_vecs_factibles = []
    u_reproyectado = []
    v_reproyectado = []

    for i in range(Z.shape[0]):
        if Z[i] > 0:
            cantidad_factibles +=1
    
            u_reproyectado += [(K[0][0]*X[i] / Z[i]) + K[0][2]]
            v_reproyectado += [(K[1][1]*Y[i] / Z[i]) + K[1][2]]

            img_vecs_factibles += [img_vecs[i]]

    distancia = 0
    for i in range(cantidad_factibles):
        diferencia_u = (u_reproyectado[i]-img_vecs_factibles[i][0])**2
        diferencia_v = (v_reproyectado[i]-img_vecs_factibles[i][1])**2

        diferencia = diferencia_u + diferencia_v
        distancia += diferencia

    RMS = np.sqrt((1/cantidad_factibles)*distancia)

    return cantidad_factibles, RMS

def aplicarPOSIT(obj_vecs, img_vecs, K, MaxIt, atol):

    x = img_vecs[:,0] - K[0][2]
    y = img_vecs[:,1] - K[1][2]
    x0 = x[0]
    y0 = y[0]
    fx = K[0][0]

    a = obj_vecs - obj_vecs[0]
    a_sin_M0 = a[1:] 

    #La matriz del sistema lineal, son las mismas filas que a_sin_M0 pero solo las columnas U y V
    A = a_sin_M0[:, :2]             
    A_pseudoinv = pseudoinversa(A)

    eps = np.zeros(a_sin_M0.shape[0]) #proyeccion ortografica escalada 

    mejor_R = None
    mejor_t = None
    rms_anterior = np.inf

    for iteracion in range(MaxIt):

        #Resolvemos el sistema lineal
        b_u = x[1:]*(1+eps) - x0
        b_v = y[1:]*(1+eps) - y0

        I0 = calcularAx(A_pseudoinv, b_u)
        J0 = calcularAx(A_pseudoinv, b_v)

        #Buscamos lambda y mu
        lamb, mu = calculoLambdaMu(I0, J0)

        #Probamos con las dos poses con los dos signos de lambda y mu 
        mejor_pose = None
        mejor_rms = np.inf

        for signo in (+1,-1):
            Is = np.array([I0[0], I0[1], signo*lamb])
            Js = np.array([J0[0], J0[1], signo*mu])
            pose = calcularSalidaPOSIT(Is, Js, fx, (x0,y0), None)

            R = pose[4]
            t = pose[5]
            cant_validos, rms = evaluarPose(R, t, a, img_vecs, K)

            if cant_validos == a.shape[0] and rms < mejor_rms:
                mejor_pose = pose
                mejor_rms = rms
             
        
        i, j, k, Z0, mejor_R, mejor_t = mejor_pose


        #Convergencia: si el RMS casi no cambio entonces devolvemos la ultima pose
        if valorAbsoluto(rms_anterior - mejor_rms) < atol:
            return mejor_R, mejor_t
        
        #Recalculamos eps
        eps = calcularAx(a_sin_M0, k) / Z0

        
        rms_anterior = mejor_rms

    return mejor_R, mejor_t

#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------