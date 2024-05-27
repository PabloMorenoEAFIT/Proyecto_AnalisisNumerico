from django.shortcuts import render
from django.http import HttpResponse


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

# metodos
# solucion numerica para sistemas de una sola variable
def vistaBiseccion(request):
    return render(request, './cadaMetodo/biseccion.html')

def vistaPuntoFijo(request):
    return render(request, './cadaMetodo/punto-fijo.html')

def vistaNewton(request):
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