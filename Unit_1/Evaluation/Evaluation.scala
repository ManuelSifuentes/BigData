
// a. ¿Cuántos días fue la columna “Close” inferior a $ 600?
df.filter($"Close"<600).count()
// b. ¿Qué porcentaje del tiempo fue la columna “High” mayor que $ 500?
(df.filter($"High">500).count()*1.0/df.count())*100
// c. ¿Cuál es la correlación de Pearson entre columna “High” y la columna “Volumen”?
df.select(corr("High","Volume")).show()
// d. ¿Cuál es el máximo de la columna “High” por año?
val df_year = df.withColumn("Year",year(df("Date")))
val df_max = df_year.select($"Year",$"High").groupBy("Year").max()
df_max.select($"Year",$"max(High)").show()
// e. ¿Cuál es el promedio de columna “Close” para cada mes del calendario?
val df_month = df.withColumn("Month",month(df("Date")))
val month_avgs = df_month.select($"Month",$"Close").groupBy("Month").mean()
month_avgs.select($"Month",$"avg(Close)").show()