# Practice #04

## Given the pseudo-code of the Fibonacci sequence in the link provided, implement with Scala:

### 1. Algorithm 1

```r
Algoritmo 1:
def fib1(n: Int): Int = {
    if(n < 2) return n else fib1(n-1) + fib1(n-2)
}

scala> fib1(15)
res22: Int = 610
```

### 2. Algorithm 2

```r
import scala.math.sqrt
import scala.math.pow

def fib2(n: Int): Int = {
    if(n < 2) {
        return n 
    }
    else {
        var p = ((1+sqrt(5))/2)
        var j = ( (pow(p,n) - pow(1-p,n) ) / sqrt(5))
        return (j).toInt
    }
}

scala> fib2(15)
res8: Int = 610
```

### 3. Algorithm 3

```r
    def fibo_3(n: Int): Int = {
      var n1 = n - 1;
      var a = 0;
      var b = 1;
      var c = 0;
      for ( k <- 0 to n1)
      {
        c = b + a;
        a = b;
        b = c;
      }
      return a
    }

    scala> fibo_3(15)
    res1: Int = 610
```

### 4. Algorithm 4

```r
    def fibo_4(n: Int): Int = {
      var n1 = n -1;
      var a = 0;
      var b = 1;
      for ( k <- 0 to n1)
      {
        b = b + a;
        a = b - a;
      }
      return(a)
    }

    scala> fibo_4(15)
    res2: Int = 610
```

### 5. Algorithm 5

```r
    def fibo_5(n: Int): Int = {
      if ( n < 2) return n
      else {
        var z = new Array[Int](n + 2);
        z(0) = 0;
        z(1) = 1;
        for ( k <- 2 to (n + 1))
          {
            z(k) = z(k - 1) + z(k - 2);
          }
        return(z(n))
      }
    }

    scala> fibo_5(15)
    res3: Int = 610
```
