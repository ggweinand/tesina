# Consigna(ish)
Cosas a probar usando feets:

* Cuantas observaciones le podemos quitar a las estrellas/cuantas necesitamos como minimo para que ande.
* Iterar entre poner puntos en fase y agregar puntos (con procesos gaussianos) mejora el período?
* Agarrar caca y agregarle puntos. Nos creemos que deja de ser caca? Para elegir caca, agarrar algo no clasificado como nada y que tenga bajo std. Samplear distintos niveles de caca.
* Las observaciones están correlacionadas. Si sacás random, seguro no se pierde eso. Si sacás muchas observaciones juntas, hay más chances de romper algo. Fijate eso.

Aclaraciones:
* Y perr? A pensarlo, es tu tesina. Por ahora es 1 std del proceso gaussiano.
* El clasificador ya hecho de JB de dónde lo robo? RandomForest de scikit, parametros del paper.
  
# Para la próxima reunión

Cosas que hice:
* Probar sigma clipping asumiendo que fuera remove noise: https://feets.readthedocs.io/en/latest/api/feets.html#module-feets.preprocess
* Aiuda con poner la clase aparte, no sé python elemental aparentemente.
* Period Deterioration con subsamples sucesivos (e historial). Hice bien el monkeypatch? No entiendo. Parece que anda bien con feets sin modificar también.
* Filter por SNR no agarra nada. Debo estar haciendo algo mal.
* Sigma clipping de feets me anda mal.

Cosas que no hice:
* Leer el paper de Lomb-Scargle.
* Elegir una estrella más amigable. Hace falta? Ya dio algo feliz.
* Usar una caché para los notebooks.

# Notas de Reunión
* Qué borré para llegar a un pico de período con LS?