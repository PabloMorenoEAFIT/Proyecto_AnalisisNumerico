
const botonConfirmar=document.querySelector(".boton");


var resultado=botonConfirmar.addEventListener("click", añadirTabla);
var tamañoGuardado;

function guardarVariable(variable){
    var variableGuardada=variable;
    tamañoGuardado=variableGuardada;
}

function añadirTabla(){
    const buscador=document.querySelector(".inputTamaño");
    const tamaño=buscador.value;
    buscador.setAttribute("hidden", true);
    const recuadro=document.querySelector(".card-body");
    const formulario=document.createElement('form');
    recuadro.append(formulario);

    for(let i=0; i<tamaño; i++){
        var divFilas=document.createElement('div');
        divFilas.classList.add('fila'+(i+1));
        formulario.append(divFilas);

        for(let j=0; j<tamaño; j++){
            var inputTabla=document.createElement("input");
            inputTabla.setAttribute("type", "text");
            inputTabla.setAttribute("name", [i]+","+[j]);
            inputTabla.setAttribute("class", [i]+","+[j]);
            inputTabla.setAttribute("value", "1");
            divFilas.append(inputTabla);
        }
    }

    botonConfirmar.setAttribute("hidden", true);
    const botonDos=document.querySelector(".botonDos");
    botonDos.removeAttribute("hidden");
    var conversionTamaño=Number(tamaño);

    var matrizPag= new Array(conversionTamaño);
    for (var i=0; i<matrizPag.length; i++){
        matrizPag[i]=new Array(conversionTamaño);
    }
    guardarVariable(tamaño);
}

var numeroConvertido=Number(tamañoGuardado);
console.log(numeroConvertido);

console.log(tamañoGuardado);
const botonDos=document.querySelector(".botonDos");
botonDos.addEventListener("click", obtenerDatos);



function obtenerDatos(){
    for(var i=0; i<tamaño; i++){
        for(var j=0; j<tamaño; j++){
            var valor=document.querySelector("."[i][j]);
            matriz[i][j]=valor;
        }
    }
}




