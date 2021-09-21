//Given the pseudo-code of the Fibonacci sequence in the link provided, implement with Scala:
//Algorithm 1
//Algorithm 2
//Algorithm 3
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
//Algorithm 4
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
//Algorithm 5
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