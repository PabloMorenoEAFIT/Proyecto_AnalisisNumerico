from django.shortcuts import render
from django.http import HttpResponse
from metodos.metodos import *
import numpy as np
import numpy.matlib
import matplotlib.pyplot as plt
import io
import urllib, base64

def landing(request):
    return render(request, 'landing.html')

def menu(request):
    # debe tener la explicacion general de cada uno de las soluciones propuestas
    return render(request, 'menuPrincipal.html')

def menuSolucionUnaSolaVariable(request):
    return render(request, 'solucion-numerica.html')

def menuSistEcu(request):
    return render(request, 'solucion-sistemas.html')

def menuInterpolacion(request):
    return render(request, 'interpolacion.html')

def graficas(request):
    return render(request, 'grafica.html')

# metodos
# solucion numerica para sistemas de una sola variable
def vistaBiseccion(request):
    try:
        if request.method == "POST":
            function_text = request.POST.get("funcion")
            a = float(request.POST.get("a"))
            b = float(request.POST.get("b"))
            tol = float(request.POST.get("tol"))
            max_count = int(request.POST.get("max_count"))

            try:
                results = biseccion(function_text, tol, max_count, a, b)
                    
                # # Generar el contenido del archivo
                # file_content = results
                # # Crear la respuesta HTTP con el archivo
                # for dato in results:
                #     for valores in dato:
                #         file_content += f"{valores}\n"
                # response = HttpResponse(file_content, content_type='text/plain')
                # response['Content-Disposition'] = 'attachment; filename="biseccion_results.txt"'
                
                # if (response is not None):
                #     try:
                #         return response
                #     except ValueError as e:
                #         return render(request, 'error.html')
                
                return render(request, './cadaMetodo/biseccion.html', {"results": results})
            
            except ValueError as e:
                return render(request, 'error.html')
            
            
            
        return render(request, './cadaMetodo/biseccion.html')
    except:
        return render(request, 'error.html')
    
def vistaPuntoFijo(request):
    if request.method == 'POST':
        try:
            fx = request.POST.get("funcion_F")
            gx = request.POST.get("funcion_G")
            x0 = float(request.POST.get("x0"))
            tol = float(request.POST.get("tolerancia"))
            niter = int(request.POST.get("iteraciones"))

            try:
                datos = puntoFijo(x0, tol, niter, fx, gx)
                
                # # Genera el contenido del archivo de texto
                # txt_content = "Iteración\tRaíz\n"
                # for i, sol in enumerate(data["soluciones"]):
                #     txt_content += f"{i+1}\t{sol}\n"
                
                # # Crea la respuesta HTTP con el contenido del archivo
                # response = HttpResponse(txt_content, content_type='text/plain')
                
                # # Establece el encabezado Content-Disposition para indicar la descarga del archivo
                # response['Content-Disposition'] = 'attachment; filename="punto_fijo_solution.txt"'
                
                return render(request, './cadaMetodo/punto-fijo.html', {"data": datos})
            
            except Exception as e:
                return render(request, 'error.html')
        except Exception as e:
            return render(request, 'error.html')
    else:
        return render(request, './cadaMetodo/punto-fijo.html')

def vistaNewton(request):
    datos = ()
    if request.method == 'POST':
        try:
            fx = request.POST.get("funcion")
            derivF = request.POST.get("df")
            x0 = float(request.POST.get("x0"))
            tol = float(request.POST.get("tolerancia"))
            niter = int(request.POST.get("iteraciones"))

            try:
                datos = newton(x0, tol, niter, fx, derivF)
                return render(request, './cadaMetodo/newton.html', {"data": datos})
            except Exception as e:
                return render(request, 'error.html')
        except Exception as e:
            return render(request, 'error.html')
    else:
        return render(request, './cadaMetodo/newton.html')

def vistaReglaFalsa(request):
    datos = ()
    if request.method == 'POST':
        try:
            fx = request.POST.get("funcion")
            a = float(request.POST.get("a"))
            b = float(request.POST.get("b"))
            tol = float(request.POST.get("tolerancia"))
            niter = int(request.POST.get("iteraciones"))

            try:
                datos = reglaFalsa(a, b, niter, tol, fx)
                return render(request, './cadaMetodo/regla-falsa.html', {"data": datos})
            except Exception as e:
                return render(request, 'error.html')
        except Exception as e:
            return render(request, 'error.html')
    else:
        return render(request, './cadaMetodo/regla-falsa.html')

def vistaSecante(request):
    datos = ()
    if request.method == 'POST':
        try:
            fx = request.POST.get("funcion")
            xs = float(request.POST.get("xs"))
            xi = float(request.POST.get("xi"))
            tol = float(request.POST.get("tolerancia"))
            niter = int(request.POST.get("iteraciones"))

            try:
                datos = secante(fx, tol, niter, xs, xi)
                return render(request, './cadaMetodo/secante.html', {"data": datos})
            except Exception as e:
                return render(request, 'error.html')
        except Exception as e:
            return render(request, 'error.html')
    else:
        return render(request, './cadaMetodo/secante.html')

def vistaReicesMultiples(request):
    datos = ()
    if request.method == 'POST':
        try:
            fx = request.POST.get("funcion")
            x0 = float(request.POST.get("x0"))
            tol = float(request.POST.get("tolerancia"))
            niter = int(request.POST.get("iteraciones"))

            try:
                datos = raicesMultiples(fx, x0, tol, niter)
                return render(request, './cadaMetodo/raices-multiples.html', {"data": datos})
            except Exception as e:
                return render(request, 'error.html')
        except Exception as e:
            return render(request, 'error.html')
    else:
        return render(request, './cadaMetodo/raices-multiples.html')

# solucion sistemas de ecuaciones
def vistaJacobi(request):
    datos = []
    if request.method == 'POST':
        try:
            matrizA = toMatrix(request.POST.get("mA"))
            Vectorx0 = toVector(request.POST.get("X0"))
            vectorB = toVector(request.POST.get("B"))

            tol = float(request.POST.get("tolerancia"))
            niter = int(request.POST.get("iteraciones"))

            try:
                # datos = jacobi(mA, vB, vx0, tol, niter)
                datos = jacobi(matrizA, vectorB, Vectorx0, tol, niter)
                return render(request, './cadaMetodo/jacobi.html', {"data": datos})
            except Exception as e:
                return render(request, 'error.html')
        except Exception as e:
            return render(request, 'error.html')
    else:
        return render(request, './cadaMetodo/jacobi.html')

def vistaGaussSeidel(request):
    datos = []
    if request.method == 'POST':
        try:
            matrizA = toMatrix(request.POST.get("mA"))
            Vectorx0 = toVector(request.POST.get("X0"))
            vectorB = toVector(request.POST.get("B"))

            tol = float(request.POST.get("tolerancia"))
            niter = int(request.POST.get("iteraciones"))

            try:
                datos = gaussSeidel(matrizA, vectorB, Vectorx0, tol, niter)
                # data_t = datos["t"]
                # data_c = datos["c"]
                # print(data_t)

                # # Convertir los datos a formato de texto
                # text_data = data_c 

                # # Configurar la respuesta HTTP con el archivo de texto
                # response = HttpResponse(text_data, content_type='text/plain')
                # response['Content-Disposition'] = 'attachment; filename="gauss_seidel_results.txt"'
                
                # Retornar la respuesta HTTP junto con los datos para el HTML
                return render(request, './cadaMetodo/gauss-seidel.html', {"data": datos})
            except Exception as e:
                return render(request, 'error.html')
        except Exception as e:
            return render(request, 'error.html')
    else:
        return render(request, './cadaMetodo/gauss-seidel.html')

def vistaSor(request):
    datos = []
    if request.method == 'POST':
        try:
            mA = toMatrix(request.POST["matrizA"])
            Vx0 = toVector(request.POST["vectorX0"])
            Vb = toVector(request.POST["vectorB"])

            w = request.POST["wValue"]
            W = float(w)

            niter = request.POST["iteraciones"]
            Niter = int(niter)

            Tol = request.POST["tolerancia"]
            Tol = float(Tol)

            try:
                datos = sor(mA,Vb,Vx0,W,Tol, Niter)
                return render(request, './cadaMetodo/sor.html', {"data": datos})
            except Exception as e:
                return render(request, 'error.html')
        except Exception as e:
            return render(request, 'error.html')
    else:
        return render(request, './cadaMetodo/sor.html')

# Interpolacion
def vistaVandermonde(request):
    if request.method == 'POST':
        try:
            if request.method=='POST':
                vectorX= toVector(request.POST["vectorX"])
                vectorY= toVector(request.POST["vectorY"])
                
            try:
                datos=vandermonde(vectorX, vectorY)
                return render(request, './cadaMetodo/vandermonde.html', {"data": datos})
            except Exception as e:
                return render(request, 'error.html')
        except Exception as e:
            return render(request, 'error.html')
    else:
        return render(request, './cadaMetodo/vandermonde.html')

def vistaSpline(request):
    Coef = []
    Tracers = []
    error_message = None
    plot_url = None
    output = {"errors": []}

    if request.method == 'POST':
        try:
            x = request.POST["x"]
            X = toVector(x)
            y = request.POST["y"]
            Y = toVector(y)
            tipo = int(request.POST["tipo"])    

            output = SplineGeneral(X, Y, tipo)
            print(output)

            if not output["errors"]:
                X, Y, x_vals, y_vals = output["results"]
                Coef = output["coef"]
                Tracers = output["tracers"]

                

                # Graficar los resultados
                plt.figure(figsize=(6, 3))
                plt.scatter(X, Y, color='g')
                plt.plot(x_vals, y_vals)

                plt.title('Método Spline')
                plt.xlabel('x')
                plt.ylabel('f(x)')
                plt.legend()
                buf = io.BytesIO()
                plt.savefig(buf, format='png')
                buf.seek(0)
                string = base64.b64encode(buf.read())
                plot_url = urllib.parse.quote(string)
            else:
                error_message = output["errors"][0]
            try:
                return render(request, './cadaMetodo/spline.html', {
                    "coef":Coef,
                    "tracers":Tracers ,
                    "errors":error_message,
                    'plot_url':plot_url})
            except Exception as e:
                return render(request, 'error.html')
        except Exception as e:
            return render(request, 'error.html')
    else:
        return render(request, './cadaMetodo/spline.html')

def vistaNewtonInt(request):
    datos = ()

    error_message = None
    plot_url = None
    coefficients = []
    if request.method == 'POST':
        try:
            
            x = request.POST["x"]
            X = toVector(x)
            y = request.POST["y"]
            Y = toVector(y)

            output = newtonInt(X, Y)

            if "errors" not in output:
                D = output["D"]
                Coef = output["Coef"]

                # Graficar los resultados
                x_vals = np.linspace(min(X), max(X), 100)
                y_vals = [newton_poly(Coef, X, x) for x in x_vals]

                plt.figure(figsize=(6, 3))
                plt.scatter(X, Y, color='g')
                plt.plot(x_vals, y_vals)

                plt.xlabel('x')
                plt.ylabel('f(x)')
                plt.legend()
                buf = io.BytesIO()
                plt.savefig(buf, format='png')
                buf.seek(0)
                string = base64.b64encode(buf.read())
                plot_url = urllib.parse.quote(string)

            try:
                datos = newtonInt(X,Y) 
                return render(request, './cadaMetodo/newton-interpolante.html', {
                    "datos": datos,
                    'coefficients': coefficients,
                    'error_message': error_message,
                    'plot_url': plot_url
                })
            except Exception as e:
                return render(request, 'error.html')
        except Exception as e:
            return render(request, 'error.html')
    else:
        return render(request, './cadaMetodo/newton-interpolante.html')

def vistaLagrange(request):
    error_message = None
    plot_url = None
    coefficients = []

    if request.method == 'POST':
        try:
            x = request.POST.get('x')
            y = request.POST.get('y')

            # Convertir a float
            X = list(map(float, x.split(',')))
            Y = list(map(float, y.split(',')))

            # Calcular los coeficientes del polinomio
            coefficients = lagrange(X, Y)

            # Graficar los resultados
            x_vals = np.linspace(min(X), max(X), 100)
            y_vals = np.polyval(coefficients, x_vals)

            plt.figure(figsize=(6, 3))
            plt.scatter(X, Y, color='g')
            plt.plot(x_vals, y_vals)

            plt.xlabel('x')
            plt.ylabel('f(x)')
            plt.legend()
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            string = base64.b64encode(buf.read())
            plot_url = urllib.parse.quote(string)

        except Exception as e:
            return render(request, 'error.html')

    return render(request, './cadaMetodo/lagrange.html', {
        'coefficients': coefficients,
        'error_message': error_message,
        'plot_url': plot_url
    })

# error
def vistaError(request):
    return render(request, 'error.html')

def toMatrix(matrixStr):
    matrixStr = matrixStr.replace(" ","")
    matrixStr = matrixStr.replace("\n","")
    rows = matrixStr.split(";")
    auxM = []
    for row in rows:
        splitedRow = row.split(",")
        auxR = []
        for num in splitedRow:
            auxR.append(float(num))
        auxM.append(auxR)
    return auxM

def toVector(vectorStr):

    splitedVector = vectorStr.split(",")
    auxV = list()
    for num in splitedVector:
        auxV.append(float(num))
    return auxV

def splineOutput(output):
    stringOutput = f'\n"Metodo"\n'
    stringOutput += "\nResults:\n"
    stringOutput += "\nTracer coefficients:\n\n"
    rel = output["results"]
    i = 0
    aux = rel.shape
    while i < aux[0] :
        j = 0
        while j < aux[1]:
            stringOutput += '{:^6f}'.format(rel[i,j]) +"  "
            j += 1
        i += 1
        stringOutput += "\n"
    stringOutput += "\n Tracers:\n"
    i = 0
    while i < aux[0] :
        j = 0
        if aux[1] == 2:
            stringOutput += format(rel[i,0],"6f") +"x"
            stringOutput += format(rel[i,1],"+.6f") 
        elif aux[1] == 3:
            stringOutput += format(rel[i,0],"6f") +"x^2"
            stringOutput += format(rel[i,1],"+.6f") +"x"
            stringOutput += format(rel[i,2],"+.6f")
        elif aux[1] == 4:
            stringOutput += format(rel[i,0],"6f") +"x^3"
            stringOutput += format(rel[i,1],"+.6f") +"x^2"
            stringOutput += format(rel[i,2],"+.6f") +"x"
            stringOutput += format(rel[i,3],"+.6f")

        i += 1
        stringOutput += "\n"
    stringOutput += "\n______________________________________________________________\n"

    return stringOutput

def newtonDiffDivOutput(output):

    stringOutput = f'\n"Metodo"\n'
    stringOutput += "\nResults:\n"
    stringOutput += "\nDivided differences table:\n\n"
    rel = output["D"]
    stringOutput += '{:^7f}'.format(rel[0,0]) +"   //L \n"

    stringOutput += "\nNewton's polynomials coefficents:\n\n"
    rel = output["Coef"]
    
    stringOutput += "\nNewton interpolating polynomials:\n\n"
    rel = output["Coef"]
    i = 0
    while i < len(rel) :
        stringOutput += '{:^7f}'.format(rel[i,0]) +"x^3"
        stringOutput += format(rel[i,1],"+.6f") + "x^2"
        stringOutput += format(rel[i,2],"+.6f") + "x" 
        stringOutput += format(rel[i,3],"+.6f") + "   //L \n"
        i += 1

    stringOutput += "\n______________________________________________________________\n"
    return stringOutput

def newton_poly(coef, x_data, x):
    n = len(x_data) - 1
    p = coef[n]
    for k in range(1, n + 1):
        p = coef[n - k] + (x - x_data[n - k]) * p
    return p