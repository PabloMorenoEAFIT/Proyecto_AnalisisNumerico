from django.shortcuts import render
from django.http import HttpResponse

# Hechos
def landing(request):
    return render(request, 'landing.html')

# falta por hacer
def menu(request):
    # debe tener la explicacion general de cada uno de las soluciones propuestas
    return render(request, 'menuPrincipal.html')

def menuSolucionUnaSolaVariable(request):
    return render(request, 'menu1Variable.html')

def menuIterativos(request):
    return render(request, 'menuIterativos.html')

def menuInterpolacion(request):
    return render(request, 'menuInterpolacion.html')