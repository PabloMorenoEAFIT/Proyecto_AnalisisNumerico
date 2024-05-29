import sympy as sympy
import sympy as sp
from sympy import Symbol, sympify, Abs, diff
from sympy.abc import x
import numpy as np
from scipy import linalg
from math import exp
from sympy import symbols, log, sin, evalf
import math
from scipy.interpolate import interp1d, CubicSpline

"""
variables y uso:

a, b        = puntos (intervalos)
tol         = tolerancia
x0          = valor inicial
fx o fun    = funcion
f(x)        = funcion evaluada en x (x cualquier valor)

"""

# sistemas de 1 variable
def biseccion(fx, Tol, Niter, a, b):
    output = {
        "columns": ["iter", "a", "xm", "b", "f(xm)", "E"],
        "iterations": Niter,
        "errors": list()
    }

    # Configuraciones iniciales
    datos = list()
    x = Symbol('x')
    i = 1
    error = 1.0000000
    Fun = sympify(fx)

    Fa = Fun.subs(x, a) # Funcion evaluada en a
    Fa = Fa.evalf()

    xm0 = 0.0
    Fxm = 0

    xm = (a + b)/2 # Punto intermedio

    Fxm = Fun.subs(x, xm) # Funcion evaluada en Xm
    Fxm = Fxm.evalf()

    try:
        datos.append([0, '{:^15.7f}'.format(a), '{:^15.7f}'.format(xm), '{:^15.7f}'.format(b), '{:^15.7E}'.format(Fxm)]) # Datos con formato dado
        while (error > Tol) and (i < Niter): # Se repite hasta que el intervalo sea lo pequeño que se desee
            if (Fa*Fxm < 0): # Se elecciona un intervalo inicial, donde el valor de la funcion cambie de signo en [a,b]
                b = xm
            else:
                a = xm # Cambia de signo en [m,b]

            xm0 = xm
            xm = (a+b)/2 # Se calcula el punto intermedio del intervalo - Divide el intervalo a la mitadd

            Fxm = Fun.subs(x, xm)
            Fxm = Fxm.evalf() # Se evalua el punto intermedio en la funcion

            error = Abs(xm-xm0) # Se calcula el error

            datos.append([i, '{:^15.7f}'.format(a), '{:^15.7f}'.format(xm), 
                            '{:^15.7f}'.format(b), '{:^15.7E}'.format(Fxm), '{:^15.7E}'.format(error)]) # Se van agregando las soluciones con el formato deseado

            i += 1
    except BaseException as e:
        if str(e) == "can't convert complex to float":
            output["errors"].append(
                "Error in data: found complex in calculations")
        else:
            output["errors"].append("Error in data: " + str(e))

        return output

    output["results"] = datos
    output["root"] = xm
    return output

def puntoFijo(X0, Tol, Niter, fx, gx):
    output = {
        "columns": ["iter", "xi", "g(xi)", "f(xi)", "E"],
        "iterations": Niter,
        "errors": []
    }

    # Configuración inicial
    datos = []
    x = sympy.Symbol('x')
    i = 1
    error = 1.0

    Fx = sympy.sympify(fx)
    Gx = sympy.sympify(gx)

    # Iteración 0
    xP = X0  # Valor inicial (Punto de evaluación)
    xA = 0.0

    Fa = Fx.subs(x, xP).evalf()  # Función evaluada en el valor inicial
    Ga = Gx.subs(x, xP).evalf()  # Función G evaluada en el valor inicial

    datos.append([1, '{:^15.7f}'.format(float(xA)), '{:^15.7f}'.format(float(Ga)), '{:^15.7E}'.format(float(Fa))])

    try:
        while error > Tol and i < Niter:  # Se repite hasta que el error sea menor a la tolerancia
            # Se evalúa el valor inicial en G, para posteriormente evaluar este valor en la función F
            # siendo-> Xn = G(x) y F(xn) = F(G(x))
            Ga = Gx.subs(x, xP).evalf()  # Función G evaluada en el punto inicial
            xA = Ga

            Fa = Fx.subs(x, xA).evalf()  # Función evaluada en el valor de la evaluación de G

            error = abs(xA - xP)  # Se calcula el error

            xP = xA  # Nuevo punto de evaluación (Punto inicial)

            datos.append([i + 1, '{:^15.7f}'.format(float(xA)), '{:^15.7f}'.format(float(Ga)),
                          '{:^15.7E}'.format(float(Fa)), '{:^15.7E}'.format(float(error))])

            i += 1

    except BaseException as e:
        output["errors"].append("Error in data: " + str(e))
        return output

    output["results"] = datos
    output["root"] = xA
    return output

def newton(x0, Tol, Niter, fx, df):

    output = {
        "columns": ["N", "xi", "F(xi)", "E"],
        "errors": list()
    }

    #configuración inicial
    datos = list()
    x = sympy.Symbol('x')
    Fun = sympify(fx)
    DerF = sympify(df)


    xn = []
    derf = []
    xi = x0 # Punto de inicio
    f = Fun.evalf(subs={x: x0}) #función evaluada en x0
    derivada = DerF.evalf(subs={x: x0}) #función derivada evaluada en x0
    c = 0
    Error = 100
    xn.append(xi)

    try:
        datos.append([1, '{:^15.7f}'.format(x0), '{:^15.7f}'.format(f)])

        # Al evaluar la derivada en el punto inicial,se busca que sea diferente de 0, ya que al serlo nos encontramos en un punto de inflexion
        #(No se puede continuar ya que la tangente es horinzontal)
        while Error > Tol and f != 0 and derivada != 0 and c < Niter: # El algoritmo converge o se alcanzo limite de iteraciones fijado

            xi = xi-f/derivada # Estimacion del siguiente punto aproximado a la raiz (nuevo valor inicial)
            derivada = DerF.evalf(subs={x: xi}) # Evaluacion de la derivada con el nuevo valor inicial (xi)
            f = Fun.evalf(subs={x: xi}) # Evaluacion de la derivada con el nuevo valor inicial (xi)
            xn.append(xi)
            c = c+1
            Error = abs(xn[c]-xn[c-1]) # Se reduce entre cada iteracion (Representado por el tramo)
            derf.append(derivada)
            datos.append([c, '{:^15.7f}'.format(float(xi)), '{:^15.7E}'.format(
                float(f)), '{:^15.7E}'.format(float(Error))])

    except BaseException as e:
        if str(e) == "can't convert complex to float":
            output["errors"].append(
                "Error in data: found complex in calculations")
        else:
            output["errors"].append("Error in data: " + str(e))
        return output

    output["results"] = datos
    output["root"] = xi
    return output

def reglaFalsa(a, b, Niter, Tol, fx):
    output = {
        "columns": ["iter", "a", "xm", "b", "f(xm)", "E"],
        "iterations": Niter,
        "errors": list()
    }

    # Configuración inicial
    datos = list()
    x = sympy.Symbol('x')
    i = 1
    cond = Tol
    error = 1.0

    Fun = sympify(fx)

    xm = 0
    xm0 = 0
    Fa = Fun.subs(x, a).evalf()
    Fb = Fun.subs(x, b).evalf()

    try:
        while error > cond and i <= Niter:
            xm = (Fb * a - Fa * b) / (Fb - Fa)
            Fx_3 = Fun.subs(x, xm).evalf()

            if i == 1:
                datos.append([i, f'{a:^15.7f}', f'{xm:^15.7f}', f'{b:^15.7f}', f'{Fx_3:^15.7E}', '-'])
            else:
                error = Abs(xm - xm0).evalf()
                datos.append([i, f'{a:^15.7f}', f'{xm:^15.7f}', f'{b:^15.7f}', f'{Fx_3:^15.7E}', f'{error:^15.7E}'])

            if Fa * Fx_3 < 0:
                b = xm
                Fb = Fx_3
            else:
                a = xm
                Fa = Fx_3

            xm0 = xm
            i += 1
    except BaseException as e:
        if str(e) == "can't convert complex to float":
            output["errors"].append("Error in data: found complex in calculations")
        else:
            output["errors"].append("Error in data: " + str(e))
        return output

    output["results"] = datos
    output["root"] = xm
    return output

def secante(fx, tol, Niter, x0, x1):
    output = {
        "columns": ["iter", "xi", "f(xi)", "E"],
        "errors": list()
    }

    results = list()
    x = Symbol('x')
    i = 0
    cond = tol
    error = 1.0000000

    Fun = sympify(fx)

    y = x0
    Fx0 = Fun
    Fx1 = Fun

    try:
        while((error > cond) and (i < Niter)): #criterios de parada
            if i == 0:
                Fx0 = Fun.subs(x, x0) #Evaluacion en el valor inicial X0
                Fx0 = Fx0.evalf()
                results.append([i, '{:^15.7f}'.format(float(x0)), '{:^15.7E}'.format(float(Fx0))])
            elif i == 1:
                Fx1 = Fun.subs(x, x1)#Evaluacion en el valor inicial X1
                Fx1 = Fx1.evalf()
                results.append([i, '{:^15.7f}'.format(float(x1)), '{:^15.7E}'.format(float(Fx1))])
            else:
                y = x1 
                # Se calcula la secante
                x1 = x1 - (Fx1*(x1 - x0)/(Fx1 - Fx0)) # Punto de corte del intervalo usando la raiz de la secante, (xi+1)
                x0 = y

                Fx0 = Fun.subs(x, x0) #Evaluacion en el valor inicial X0
                Fx0 = Fx1.evalf() 

                Fx1 = Fun.subs(x, x1)#Evaluacion en el valor inicial X1
                Fx1 = Fx1.evalf()

                error = Abs(x1 - x0) # Tramo

                results.append([i, '{:^15.7f}'.format(float(x1)), '{:^15.7E}'.format(float(Fx1)), '{:^15.7E}'.format(float(error))])
            i += 1
    except BaseException as e:
        if str(e) == "can't convert complex to float":
            output["errors"].append(
                "Error in data: found complex in calculations")
        else:
            output["errors"].append("Error in data: " + str(e))

        return output

    output["results"] = results
    output["root"] = y
    return output

def raicesMultiples(fx, x0, tol, niter):

    output = {
        "columns": ["iter", "xi", "f(xi)", "E"],
        "iterations": niter,
        "errors": list()
    }
    
    # Configuraciones inciales
    results = list()
    x = Symbol('x')
    cond = tol
    error = 1.0000
    ex = sympify(fx)

    d_ex = diff(ex, x)  # Primera derivada de Fx
    d2_ex = diff(d_ex, x)  # Segunda derivada de Fx


    xP = x0
    ex_2 = ex.subs(x, x0)  # Funcion evaluada en x0
    ex_2 = ex_2.evalf() 

    d_ex2 = d_ex.subs(x, x0) # primera derivada evaluada en x0
    d_ex2 = d_ex2.evalf()

    d2_ex2 = d2_ex.subs(x, x0) # segunda derivada evaluada en x0
    d2_ex2 = d2_ex2.evalf()

    i = 0
    results.append([i, '{:^15.7E}'.format(x0), '{:^15.7E}'.format(ex_2)]) # Datos con formato dado
    try:
        while((error > cond) and (i < niter)): # Se repite hasta que el intervalo sea lo pequeño que se desee
            if(i == 0):
                ex_2 = ex.subs(x, xP) # Funcion evaluada en valor inicial
                ex_2 = ex_2.evalf()
            else:
                d_ex2 = d_ex.subs(x, xP) # Funcion evaluada en valor inicial
                d_ex2 = d_ex2.evalf()

                d2_ex2 = d2_ex.subs(x, xP) # Funcion evaluada en valor inicial 
                d2_ex2 = d2_ex2.evalf()

                xA = xP - (ex_2*d_ex2)/((d_ex2)**2 - ex_2*d2_ex2) # Método de Newton-Raphson modificado

                ex_A = ex.subs(x, xA) # Funcion evaluada en xA
                ex_A = ex_A.evalf()

                error = Abs(xA - xP)
                error = error.evalf() # Se calcula el error
                er = error

                ex_2 = ex_A #se establece la nueva aproximación
                xP = xA

                results.append([i, '{:^15.7E}'.format(float(xA)), '{:^15.7E}'.format(
                    float(ex_2)), '{:^15.7E}'.format(float(er))])
            i += 1
    except BaseException as e:
        if str(e) == "can't convert complex to float":
            output["errors"].append(
                "Error in data: found complex in calculations")
        else:
            output["errors"].append("Error in data: " + str(e))

        return output

    output["results"] = results
    output["root"] = xA
    return output

#Sistemas de ecuaciones


def jacobi(Ma, Vb, x0, tol, niter):
    output = {
        "iterations": niter,
        "errors": list(),
    }

    A = np.matrix(Ma)
    sX = np.size(x0)
    xA = np.zeros((sX, 1))

    b = np.array(Vb)
    s = b.size
    b = np.reshape(b, (s, 1)) # Rehace el tamaño del vector b

    D = np.diag(np.diag(A)) # Saca la diagonal de la matriz A
    L = -1 * np.tril(A) + D # Saca la matriz Lower de la matriz A
    U = -1 * np.triu(A) + D # Saca la matriz Upper de la matriz A
    LU = L + U

    T = np.linalg.inv(D) @ LU # Obtiene la matriz de Transición multiplicando el inverso de D por la matriz LU
    tFinal = max(abs(np.linalg.eigvals(T)))
    C = np.linalg.inv(D) @ b # Obtiene la matriz de coeficientes multiplicando el inverso de la matriz de D por la matriz b

    output["t"] = T
    output["c"] = C

    xP = x0
    E = 1000
    cont = 0

    steps = {'Step 0': np.copy(xA)}
    try:
        while (E > tol and cont < niter):
            xA = T @ xP + C
            E = np.linalg.norm(xP - xA)
            xP = xA
            cont = cont + 1
            steps[f'Step {cont}'] = np.copy(xA)

    except Exception as e:
        output["errors"].append(str(e))
        return output

    output["steps"] = steps
    output["spectral_radius"] = tFinal
    output["root"] = xA
    return output


def gaussSeidel(Ma, Vb, x0, tol, niter):
    iteraciones=[]
    informacion=[]
    error=[]

    sX = np.size(x0)
    xA = np.zeros((sX, 1))

    A = np.matrix(Ma)

    b = np.array(Vb)
    s = b.size
    b = np.reshape(b, (s, 1)) #Rehace el tamaño del vector b

    D = np.diag(np.diag(A)) #saca la diagonal de la matriz A
    L = -1*np.tril(A)+D #saca la matriz Lower de la matriz A
    U = -1*np.triu(A)+D #Saca la matriz Upper de la matriz A

    T = np.linalg.inv(D-L) @ U #Obtiene la matriz de Transicion multiplicando el inverso de D-L por la matriz U
    tFinal=max(abs(np.linalg.eigvals(T)))
    C = np.linalg.inv(D-L) @ b #Obtiene la matriz Coeficientes multiplicando el inverso de D-L por la matriz b

    xP = x0
    E = 1000
    cont = 0


    nose=[]
    steps = {'Step 0': np.copy(xA)}
    while(E > tol and cont < niter):
        xA = T@xP + C
        E = np.linalg.norm(xP - xA)
        xP = xA
        cont = cont + 1
        steps[f'Step {cont+1}'] = np.copy(xA)
        print(xA [: , 1])



    datos=zip(iteraciones, error, informacion)
    resultado={"t":T,
                "c":C,
                "esp":tFinal,
                "informacion":datos}
    return resultado


def sor(Ma, Vb, x0, w, tol, niter):

    A = np.matrix(Ma)

    b = np.array(Vb)
    s = b.size
    b = np.reshape(b, (s, 1)) #Rehace el tamaño del vector b

    D = np.diag(np.diag(A)) #saca la diagonal de la matriz A
    L = -1*np.tril(A)+D #saca la matriz Lower de la matriz A
    U = -1*np.triu(A)+D #Saca la matriz Upper de la matriz A

    T = np.linalg.inv(D-L) @ U #Obtiene la matriz de Transicion multiplicando el inverso de D-L por la matriz U
    tFinal=max(abs(np.linalg.eigvals(T)))
    C = np.linalg.inv(D-L) @ b #Obtiene la matriz Coeficientes multiplicando el inverso de D-L por la matriz b

    iteraciones=[]
    informacion=[]
    cumple=False
    n=len(Ma)
    k=0

    while(not cumple and k<niter):
        xk1=np.zeros(n)
        for i in range(n):
            s1=np.dot(Ma[i][:i],xk1[:i]) #Multiplica los valores de la Matriz A hasta el final de la matriz xk1
            s2=np.dot(Ma[i][i+1:], x0[i+1:])# Multiplica la matrizA con el vector de inicio
            xk1[i]=(Vb[i]-s1-s2)/Ma[i][i]*w+(1-w)*x0[i] #Hace las operaciones para obtener el resultado del metodo
        norma=np.linalg.norm(x0-xk1)
        x0=xk1 #actualiza los valores para el proximo ciclo
        print('Iteracion:{}->{} norma {}'.format(k, xk1, norma))
        iteraciones.append(k)
        informacion.append(xk1)
        cumple=norma<tol
        k+=1
    
    if k<niter:
        datos=zip(iteraciones, informacion) #guarda el contador, informacion
        resultado={"solucion":x0,
                    "t":T,
                    "c":C,
                    "esp":tFinal,
                    "informacion":datos}
        return resultado
    else:
        return "el sistem no converge"

#Metodos interpolacion

def splineLineal(X, Y):
    output = {
        "errors": list(),
        "results": None,
        "tracers": None
    }

    try:
        X = np.array(X)
        Y = np.array(Y)

        # Verifica que haya suficientes puntos para una interpolación lineal
        if len(X) < 2:
            output["errors"].append(
                "Se requieren al menos 2 puntos para una interpolación lineal.")
            return output

        # Realiza la interpolación lineal
        linear_interpolation = interp1d(X, Y, kind='linear')

        # Genera puntos de muestra para graficar el polinomio interpolado
        x_vals = np.linspace(min(X), max(X), 500)
        y_vals = linear_interpolation(x_vals)

        # Calcula los coeficientes del polinomio
        coef = []
        for i in range(len(X) - 1):
            slope = (Y[i + 1] - Y[i]) / (X[i + 1] - X[i])
            intercept = Y[i] - slope * X[i]
            coef.append([slope, intercept])

        # Formatea los polinomios para mostrarlos
        tracers = [f"S{i}(x) = {slope:.2f}*x + {intercept:.2f}" for
                   i, (slope, intercept) in enumerate(coef)]

        # Guarda los resultados en el output
        output["results"] = (X, Y, x_vals, y_vals)
        output["coef"] = coef
        output["tracers"] = tracers
    except Exception as e:
        output["errors"].append("Error in data: " + str(e))

    return output

def splineCuadratica(X, Y):
    output = {
        "errors": list(),
        "results": None,
        "tracers": None
    }

    try:
        X = np.array(X)
        Y = np.array(Y)

        # Verifica que haya suficientes puntos para una interpolación cuadrática
        if len(X) < 3:
            output["errors"].append(
                "Se requieren al menos 3 puntos para una interpolación cuadrática.")
            return output

        # Realiza la interpolación cuadrática
        quadratic_interpolation = interp1d(X, Y, kind='quadratic')

        # Genera puntos de muestra para graficar el polinomio interpolado
        x_vals = np.linspace(min(X), max(X), 500)
        y_vals = quadratic_interpolation(x_vals)

        # Calcula los coeficientes del polinomio
        coef = []
        for i in range(len(X) - 1):
            x_section = X[i:i + 3]
            y_section = Y[i:i + 3]
            poly = np.polyfit(x_section, y_section, 2)
            coef.append(poly)

        # Formatea los polinomios para mostrarlos
        tracers = [
            f"S{i}(x) = {poly[0]:.2f}*x^2 + {poly[1]:.2f}*x + {poly[2]:.2f}" for
            i, poly in enumerate(coef)]

        # Guarda los resultados en el output
        output["results"] = (X, Y, x_vals, y_vals)
        output["coef"] = coef
        output["tracers"] = tracers
    except Exception as e:
        output["errors"].append("Error in data: " + str(e))

    return output

def splineCubica(X, Y):
    output = {
        "errors": list(),
        "results": None,
        "tracers": None
    }

    try:
        X = np.array(X)
        Y = np.array(Y)

        # Verifica que haya suficientes puntos para una interpolación cúbica
        if len(X) < 4:
            output["errors"].append(
                "Se requieren al menos 4 puntos para una interpolación cúbica.")
            return output

        # Realiza la interpolación cúbica
        cs = CubicSpline(X, Y, bc_type='natural')

        # Genera puntos de muestra para graficar el polinomio interpolado
        x_vals = np.linspace(min(X), max(X), 500)
        y_vals = cs(x_vals)

        # Calcula los coeficientes del polinomio
        coef = cs.c.T  # Coeficientes de los polinomios

        # Formatea los polinomios para mostrarlos
        tracers = [
            f"S{i}(x) = {coef[i, 0]:.2f}*x^3 + {coef[i, 1]:.2f}*x^2 + {coef[i, 2]:.2f}*x + {coef[i, 3]:.2f}"
            for i in range(len(coef))]

        # Guarda los resultados en el output
        output["results"] = (X, Y, x_vals, y_vals)
        output["coef"] = coef
        output["tracers"] = tracers
    except Exception as e:
        output["errors"].append("Error in data: " + str(e))

    return output

def SplineGeneral(X, Y, n):
    if n == 1:
        return splineLineal(X,Y)
    elif n== 2:
        return splineCuadratica(X,Y)
    elif n== 3:
        return splineCubica(X,Y)
    else: 
        pass

def vandermonde(a,b):
    copiaB=np.copy(b)
    longitudMatriz=len(a)
    matrizVandermonde=np.vander(a) #Obtiene la matriz vandermonde con la matrizA
    coeficientes=np.linalg.solve(matrizVandermonde, copiaB) #Encuentra la Matriz A con vector B
            
    print(coeficientes)
    x=sympy.Symbol('x')
    polinomio=0
    for i in range(0, longitudMatriz, 1): #ciclo para asignarle las x y la potencias al polinomio
        potencia=(longitudMatriz-1)-i
        termino=coeficientes[i]*(x**potencia)
        polinomio=polinomio+termino

    print(polinomio)
    datos={
        "matriz":matrizVandermonde,
        "coeficientes":coeficientes,
        "polinomio":polinomio,
    }

    return datos

def newtonInt(X, Y):
    output = {}

    X = np.array(X)
    n = X.size

    Y = np.array(Y)

    D = np.zeros((n,n))

    D[:,0]=Y.T
    for i in range(1,n):
        aux0 = D[i-1:n,i-1]
        aux = np.diff(aux0)
        aux2 = X[i:n] - X[0:n-i]
        D[i:n,i] = aux/aux2.T  
        Coef = np.diag(D)
    output["D"] = D
    output["Coef"] = Coef
    print(D)
    print(Coef)

    return output

def lagrange(x, y):
    n = len(x)
    polynomial = np.poly1d([0.0])

    for i in range(n):
        Li = np.poly1d([1.0])
        den = 1.0
        for j in range(n):
            if j != i:
                Li *= np.poly1d([1.0, -x[j]])
                den *= (x[i] - x[j])
        polynomial += y[i] * Li / den

    return polynomial.coeffs