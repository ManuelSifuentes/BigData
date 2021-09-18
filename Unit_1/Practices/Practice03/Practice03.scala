//Practice 03

//1. Create a list called "list" with the elements "red", "white", "black"
//2. Add 5 more items to "list" "green", "yellow", "blue", "orange", "pearl"//
//3. Bring the items from "list" "green", "yellow", "blue"
//4. Create an array of numbers in the range 1-1000 in steps of 5 by 5
val arr = Array.range(0, 1001, 5)
arr(999)

//5. What are the unique elements of the list List (1,3,3,4,6,7,3,7) use conversion to sets
val Lista = List(1,3,3,4,6,7,3,7)
Lista.toSet

//6. Create a mutable map named names that contains the following "José", 20, "Luis", 24, "Ana", 23, "Susana", "27"
val mutmap = collection.mutable.Map(( "Jose", 20), ("Luis", 24), ("Ana", 23), ("Susana", "27"))


//  6 a. Print all keys on the map
mutmap.keys


//   6 b. Add the following value to the map ("Miguel", 23)
mutmap += ("Miguel" -> 23)
