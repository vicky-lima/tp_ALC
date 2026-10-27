import numpy as np

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
        if D[i][i] > tol :
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
    E = autoval_ordenados ** (1/2)
    r = autoval_ordenados.size

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

# PARA LA INVERSA NECESITAMOS RES_TRI

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
            if L[i][i] == 0:
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

def esSimetrica(A):
    return trasponer(A) == A

# CHOLESKY

def calculaCholesky(A,atol=1e-10):

    n,m = A.shape
    if not esSDP(A, atol) or not esSimetrica(A):
        return None

    L, U = calculaLU(A)
    D = np.zeros(n,m)

    for i in range(n):
        D[i][i] = (U[i][i]**1/2)

    L_t = trasponer(L)
    L_cholesky_t = multiplicar_matrices(D, L_t)

    L_cholesky = trasponer(L_cholesky_t)

    return L_cholesky





    