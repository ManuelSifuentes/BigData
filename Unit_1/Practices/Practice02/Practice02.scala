//Practice 02

//1. Develop a scala algorithm that calculates the radius of a circle
import scala.io.StdIn.readLine

def radio(): Unit = {
    println("Ingresa el diametro de un circulo: ")
    var D = scala.io.StdIn.readInt()
    var r = D/2
    println("El radio del circulo es: " + r)
}

radio()

//2. Develop a scala algorithm that tells me if a number is prime
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

//3. Given the variable var bird = "tweet", use string interpolation to print "I'm writing a tweet"
var bird  = "tweet"
val message = s"Estoy ecribiendo un $bird"

//4. Given the variable var message = "Hi Luke, I'm your father!" uses slice to extract the sequence "Luke"
//5. What is the difference between value (val) and a variable (var) in scala?
//6. Given the tuple (2,4,5,1,2,3,3,1416,23) returns the number 3.1416
