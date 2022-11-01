# Consigna(ish)
Cosas a probar usando feets:

* Iterar entre poner puntos en fase y agregar puntos (con procesos gaussianos) mejora el período?
* Agarrar caca y agregarle puntos. Nos creemos que deja de ser caca? Para elegir caca, agarrar algo no clasificado como nada y que tenga bajo std. Samplear distintos niveles de caca.
* Las observaciones están correlacionadas. Si sacás random, seguro no se pierde eso. Si sacás muchas observaciones juntas, hay más chances de romper algo. Fijate eso.

Aclaraciones:
* Y perr? A pensarlo, es tu tesina. Por ahora es 1 std del proceso gaussiano.
* El clasificador ya hecho de JB de dónde lo robo? RandomForest de scikit, parametros del paper.
* Citar 4.1. A Non-uniform Nyquist Limit? del Paper de Jake (al tener uneven samples no hay )
  
# Para la próxima reunión

Cosas que hice:
* Hacer side to side del ruido en el gp para ver si con incerteza mas grande se caga menos.
* Agarro las RRab del b278. Les agrego puntos con gp (dejas de agregar cuando deja de "aportar informacion", segun periodls, period_fit). Recalculo esos 2 features. Comparo con los datos originales con error cuadrático medio por cada punto extra.
Cuando ande eso lo hago para todas las RRLyrae de todos los tiles.
* Googlear regularización en gp (como evitar overfitting?)
* Dos formas de ruido: ruido gaussiano sobre los puntos y outliers.


Cosas que no hice:
* Escribir tesina. Me tengo que sentar con el Mitchell.
* Armar pipeline con random forest?
* Hacer una función que plotee los puntos observados y los puntos generados con colores distintos.
* Escribir 1h por dia :)
* Agarrar observaciones cercanas y promediar (y sumar los errores).

# Notas de Reunión
Ver como hacer al gp mas resistente al ruido. Agarro varios subsamples y promedio?
Se puede por ejemplo agarrar todos los puntos menos uno y ver si ese queda en el gp de los demas. Si no queda probablemente sea caca.

Con eso calculas cuantas agregar.
Agarro todas las RRLyrae de un tile y todas las desconocidas de un tile. Agrego la cantidad de puntos que dije en el experimento pasado, le tiro un RF como clasificador con los parámetros de JB y hago la Precision Recall AUC estable.



Graficar bien mse tanto para scikit como jorge pelado y subir a slides.
Commitear todo lindo.
Armar la pipeline entera con RF.
Intentar hacer andar mcmc con jorge.
Arreglar lo del overleaf.
