# Practice #03

1. Create a list called "list" with the elements "red", "white", "black"
2. Add 5 more items to "list" "green", "yellow", "blue", "orange", "pearl"
3. Bring the items from "list" "green", "yellow", "blue"
4. Create an array of numbers in the range 1-1000 in steps of 5 by 5
5. What are the unique elements of the list List (1,3,3,4,6,7,3,7) use conversion to sets
6. Create a mutable map named names that contains the following
     "José", 20, "Luis", 24, "Ana", 23, "Susana", "27"
   6 a. Print all keys on the map
   6 b. Add the following value to the map ("Miguel", 23)


### 4. Create an array of numbers in the range 1-1000 in steps of 5 by 5
``` r
val arr = Array.range(0, 1001, 5)

```

### 5. What are the unique elements of the list List (1,3,3,4,6,7,3,7) use conversion to sets
``` r
val Lista = List(1,3,3,4,6,7,3,7)
Lista.toSet

scala> Lista.toSet
res1: scala.collection.immutable.Set[Int] = Set(1, 6, 7, 3, 4)
```

### 6. Create a mutable map named names that contains the following "José", 20, "Luis", 24, "Ana", 23, "Susana", "27"
``` r
val mutmap = collection.mutable.Map(( "Jose", 20), ("Luis", 24), ("Ana", 23), ("Susana", "27"))
```
### 6a. Print all keys on the map
``` r
scala> mutmap.keys
res0: Iterable[String] = Set(Susana, Ana, Luis, Jose)
```

### 6b. Add the following value to the map ("Miguel", 23)
``` r
scala> mutmap += ("Miguel" -> 23)
res1: mutmap.type = Map(Susana -> 27, Ana -> 23, Miguel -> 23, Luis -> 24, Jose -> 20)
```
