from django.shortcuts import render
from django.http import HttpResponse
from metodos.metodos import *


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
            function_text = request.POST.get("function_text")
            a = float(request.POST.get("a"))
            b = float(request.POST.get("b"))
            tol = float(request.POST.get("tol"))
            max_count = int(request.POST.get("max_count"))

            try:
                results = biseccion(function_text, tol, max_count,a, b)
                return render(request, './cadaMetodo/biseccion.html', {"results": results})
            except ValueError as e:
                return render(request, 'error.html')
            
        return render(request, './cadaMetodo/biseccion.html')
    except:
        return render(request, 'error.html')
    
def vistaPuntoFijo(request):
    if request.method == 'POST':
        try:
            fx = request.POST.get("funcion-F")
            gx = request.POST.get("funcion-G")
            x0 = float(request.POST.get("vInicial"))
            tol = float(request.POST.get("tolerancia"))
            niter = int(request.POST.get("iteraciones"))

            try:
                data = puntoFijo(x0, tol, niter, fx, gx)
                
                # Genera el contenido del archivo de texto
                txt_content = "Iteración\tRaíz\n"
                for i, sol in enumerate(data["soluciones"]):
                    txt_content += f"{i+1}\t{sol}\n"
                
                # Crea la respuesta HTTP con el contenido del archivo
                response = HttpResponse(txt_content, content_type='text/plain')
                
                # Establece el encabezado Content-Disposition para indicar la descarga del archivo
                response['Content-Disposition'] = 'attachment; filename="punto_fijo_solution.txt"'
                
                return response
            except Exception as e:
                return render(request, 'error.html')
        except Exception as e:
            return render(request, 'error.html')
    else:
        return render(request, './cadaMetodo/punto-fijo.html')




def vistaNewton(request):
    data = ()
    if request.method == 'POST':
        try:
            fx = request.POST.get("funcion")
            derivF = request.POST.get("funcion-df")
            x0 = float(request.POST.get("vInicial"))
            tol = float(request.POST.get("tolerancia"))
            niter = int(request.POST.get("iteraciones"))

            try:
                data = newton(x0, tol, niter, fx, derivF)
                return render(request, './cadaMetodo/newton.html', {"data": data})
            except Exception as e:
                return render(request, 'error.html')
        except Exception as e:
            return render(request, 'error.html')
    else:
        return render(request, './cadaMetodo/newton.html')

def vistaReglaFalsa(request):
    return render(request, './cadaMetodo/regla-falsa.html')

def vistaSecante(request):
    return render(request, './cadaMetodo/secante.html')

def vistaReicesMultiples(request):
    return render(request, './cadaMetodo/reices-multiples.html')

# solucion sistemas de ecuaciones
def vistaJacobi(request):
    return render(request, './cadaMetodo/jacobi.html')

def vistaGaussSeidel(request):
    return render(request, './cadaMetodo/gauss-seidel.html')

def vistaSor(request):
    return render(request, './cadaMetodo/sor.html')

# Interpolacion
def vistaVandermonde(request):
    return render(request, './cadaMetodo/vandermonde.html')

def vistaSpline(request):
    return render(request, './cadaMetodo/spline.html')

def vistaNewtonInt(request):
    return render(request, './cadaMetodo/newton-interpolante.html')

def vistaLagrange(request):
    return render(request, './cadaMetodo/lagrange.html')

# error
def vistaError(request):
    return render(request, 'error.html')