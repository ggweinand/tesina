# Consigna(ish)
Cosas a probar usando feets:

* Agarrar caca y agregarle puntos. Nos creemos que deja de ser caca? Para elegir caca, agarrar algo no clasificado como nada y que tenga bajo std. Samplear distintos niveles de caca.
* Las observaciones están correlacionadas. Si sacás random, seguro no se pierde eso. Si sacás muchas observaciones juntas, hay más chances de romper algo. Fijate eso.

Aclaraciones:
* Y perr? A pensarlo, es tu tesina. Por ahora es 1 std del proceso gaussiano.
* El clasificador ya hecho de JB de dónde lo robo? RandomForest de scikit, parametros del paper.
* Citar 4.1. A Non-uniform Nyquist Limit? del Paper de Jake (al tener uneven samples no hay )
  
# Para la próxima reunión

Cosas que hice:
* Subir todo lindo a Github.
* Graficar bien mse tanto para scikit como jorge pelado y subir a slides.
RR* Wrapper de RF.

Cosas que no hice:

# Notas de Reunión
Agregale 1000 puntos a cada lc en vez de 40.

Supported criteria are “gini” for the Gini impurity and “log_loss” and “entropy” both for the Shannon information gain. Cual usa JB? El paper dice: We created 500 decision trees with Information-Gain as metric.
Es entropy.

Qué features recalculo con feets sobre las curvas de luz aumentadas?
Me parece que todas las que pueda con feets: https://feets.readthedocs.io/en/latest/tutorial.html#The-Features
Sí.

Qué pasa con las fuentes de luz en features que no tienen vs_type? Son las no periódicas y ya (son todas una misma clase)? Creo que sí. "Second, there are FNs. These are already known RRLs that our pipeline assigned to the unknown-star class."
Nos creemos que son todas una clase y son todas nada (no RR-Lyrae).

Qué es un TP? Importa el tipo de RR-Lyrae?
No. Hay dos clases. Una que son todas las RR-Lyrae: rr_lyrae = ["RRLyr-RRab", "RRLyr-RRc", "RRLyr-RRd"]
La otra que son todas las demás.

Tiene sentido borrar id, cnt?
Sí. Mirá en el paper qué features usar.
Table 2. Features selected for the creation of catalogs of RRL stars on tiles of VVV

Armar la pipeline entera con RF: Agarro todas las RRLyrae de un tile y todas las desconocidas de un tile. Agrego la cantidad de puntos que dije en el experimento pasado, le tiro un RF como clasificador con los parámetros de JB y hago la Precision Recall AUC estable.

La pipeline es:
Armar un df grande con las features de todas las tiles.
A cada curva de luz aumentada recalcularle las features.
Entrenar un RF con el df grande y estimar qué? ROC-AUC?

Intentar hacer andar mcmc con jorge.

Arreglar lo del overleaf.