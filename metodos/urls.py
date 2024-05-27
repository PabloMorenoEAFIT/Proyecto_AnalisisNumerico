from django.urls import path
from . import views

urlpatterns = [
    path("", views.landing, name="landing.html"),
    path("menuPrincipal", views.menu, name="menuPrincipal.html"),
    path("solucion-numerica", views.menuSolucionUnaSolaVariable, name="solucion-numerica.html"),
    path("solucion-sistemas", views.menuSistEcu, name="solucion-sistemas.html"),
    path("interpolacion", views.menuInterpolacion, name="interpolacion.html"),
    path('biseccion', views.vistaBiseccion, name="biseccion.html"),
    path('punto-fijo', views.vistaPuntoFijo, name="punto-fijo.html"),
    path('newton', views.vistaNewton, name="newton.html"),
    path('regla-falsa', views.vistaReglaFalsa, name="regla-falsa.html"),
    path('secante', views.vistaSecante, name="secante.html"),
    path('raices-multiples', views.vistaReicesMultiples, name="raices-multiples.html"),
    path('biseccion', views.vistaBiseccion, name="biseccion.html"),

    path('jacobi', views.vistaJacobi, name="jacobi.html"),
    path('gauss-seidel', views.vistaGaussSeidel, name="gauss-seidel.html"),
    path('sor', views.vistaSor, name="sor.html"),

    path('vandermonde', views.vistaVandermonde, name="vandermonde.html"),
    path('spline', views.vistaSpline, name="spline.html"),
    path('newton-interpolante', views.vistaNewtonInt, name="newton-interpolante.html"),
    path('lagrange', views.vistaLagrange, name="lagrange.html"),

    # error
    path('error', views.vistaError, name="error.html"),

    # grafica
    path('grafica', views.graficas, name="grafica.html"),
]