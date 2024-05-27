"""final_analisis URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/4.1/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path, include
from metodos import views

urlpatterns = [
# basico
    path('admin/', admin.site.urls),
    path("", views.landing, name="landing"),
    path("menuPrincipal", views.menu, name="menuPrincipal"),
    path("solucion-numerica", views.menuSolucionUnaSolaVariable, name="solucion-numerica"),
    path("solucion-sistemas", views.menuSistEcu, name="solucion-sistemas"),
    path("interpolacion", views.menuInterpolacion, name="interpolacion"),

# solucion para sistemas de una sola variables
    path("biseccion", views.vistaBiseccion, name="biseccion"),
    path('punto-fijo', views.vistaPuntoFijo, name="punto-fijo"),
    path('newton', views.vistaNewton, name="newton"),
    path('regla-falsa', views.vistaReglaFalsa, name="regla-falsa"),
    path('secante', views.vistaSecante, name="secante"),
    path('raices-multiples', views.vistaReicesMultiples, name="raices-multiples"),
    path('biseccion', views.vistaBiseccion, name="biseccion"),

# solucion para sistemas de ecuaciones
    path('jacobi', views.vistaJacobi, name="jacobi"),
    path('gauss-seidel', views.vistaGaussSeidel, name="gauss-seidel"),
    path('sor', views.vistaSor, name="sor"),

# interpolacion
    path('vandermonde', views.vistaVandermonde, name="vandermonde"),
    path('spline', views.vistaSpline, name="spline"),
    path('newton-interpolante', views.vistaNewtonInt, name="newton-interpolante"),
    path('lagrange', views.vistaLagrange, name="lagrange"),

# error
    path('error', views.vistaError, name="error"),
# graficas
    path('grafica', views.graficas, name="grafica"),
]
