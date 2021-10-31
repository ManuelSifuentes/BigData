# Practice #02

1. Develop a scala algorithm that calculates the radius of a circle
2. Develop a scala algorithm that tells me if a number is prime
3. Given the variable var bird = "tweet", use string interpolation to
   print "I'm writing a tweet"
4. Given the variable var message = "Hi Luke, I'm your father!" uses slice to extract the
   sequence "Luke"
5. What is the difference between value (val) and a variable (var) in scala?
6. Given the tuple (2,4,5,1,2,3,3,1416,23) returns the number 3.1416

### 1. Develop a scala algorithm that calculates the radius of a circle

```r
import scala.io.StdIn.readLine

def radio(): Unit = {
    println("Ingresa el diametro de un circulo: ")
    var D = scala.io.StdIn.readInt()
    var r = D/2
    println("El radio del circulo es: " + r)
}

radio()

scala> radio()
Ingresa el diametro de un circulo:
El radio del circulo es: 5
```

### 2. Develop a scala algorithm that tells me if a number is prime

```r
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

scala> primo()
Ingresa un numero:
11 es impar
```

### 3. Given the variable var bird = "tweet", use string interpolation to print "I'm writing a tweet"

```r
var bird  = "tweet"
val message = s"Estoy ecribiendo un $bird"

scala> var bird  = "tweet"
bird: String = tweet

scala> val message = s"Estoy ecribiendo un $bird"
message: String = Estoy ecribiendo un tweet
```

### 4. Given the variable var message = "Hi Luke, I'm your father!" uses slice to extract the sequence "Luke"

```r
val mensaje = "Hola Luke yo soy tu padre!"
mensaje.slice(5,9)

scala> mensaje.slice(5,9)
res1: String = Luke

```

### 5. What is the difference between value (val) and a variable (var) in scala?

- “val” creates an immutable variable, since its value cannot change once declared.
- “var” creates a mutable variable, it is possible to modify its original value.

### 6. Given the tuple (2,4,5,1,2,3,3,1416,23) returns the number 3.1416

```r
val tupla = (2,4,5,1,2,3,3.1416,23)
tupla._7

scala> tupla._7
res2: Double = 3.1416

```
