//1. Desarrollar un algoritmo en scala que calcule el radio de un circulo
import scala.io.StdIn.readLine

def radio(): Unit = {
    println("Ingresa el diametro de un circulo: ")
    var D = scala.io.StdIn.readInt()
    var r = D/2
    println("El radio del circulo es: " + r)
}

radio()

//2. Desarrollar un algoritmo en scala que me diga si un número es primo
def primo(): Unit = {
    println("Ingresa un numero: ")
    val n = scala.io.StdIn.readInt()
    if(n%2==0){
        println(s"$n es par")
    } else {
        println(s"$n es impar")
    }
}

primo()

//3.Dada la variable  var bird = "tweet", utiliza interpolación de strings para
//imprimir "Estoy ecribiendo un tweet"
var bird  = "tweet"
val message = s"Estoy ecribiendo un $bird"

//4.Dada la variable var mensaje = "Hola Luke yo soy tu padre!" utiliza slice para extraer la
//secuencia "Luke"
 var mensaje = "Hola Luke yo soy tu padre!"
 mensaje slice  (5,9)

 //5.Cúal es la diferencia entre value (val) y una variable (var) en scala?
Val: Son constantes, una vez asignado su valor no puede cambiar a lo largo del programa, por lo que no puede ser reasignado a partir de otro valor o variable
Var: Son varuables, lo que indica que su valor puede cambiar a lo largo del programa (el tipo de dato debe de ser el mismo)

//6.Dada la tupla (2,4,5,1,2,3,3.1416,23) regresa el número 3.1416
val my_tup = (2,4,5,1,2,3,3.1416,23)
my_tup._6