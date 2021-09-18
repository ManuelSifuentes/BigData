# Practice #01

Test the Law Of Large Numbers for N random normally distributed numbers with mean = 0, stdev=1:

Create an R script that will count how many of these numbers fall between -1 and 1 and divide by the total quantity of N

You know that E(X) = 68.2%
Check that Mean(Xn)->E(X) as you rerun your script while increasing N

Hint:
1. Initialize sample size
2. Initialize counter
3. loop for(i in rnorm(size))
4. Check if the iterated variable falls
5. Increase counter if the condition is true
6. return a result <- counter / N


### 1. Initialize sample size
``` r
Numbers = 1:30
```

### 2. Initialize counter
``` r
counter = 1
```

### 3. loop for(i in rnorm(size))
``` r
for(i in rnorm(Numbers))
  {
  print(i)
  }
```

### 4. Check if the iterated variable falls
``` r
  if(i >= -1 & i <= 1){

  }
```

### 5. Increase counter if the condition is true
``` r
    counter <- counter + 1
```

### 6. return a result <- counter / N
``` r
result = counter/Numbers 

print(result*100)
```
