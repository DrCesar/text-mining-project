

# Proyecto de Sentiment Análisis.

### Josué Jacobs 


En ese proyecto se realizo un scrapping de reviews the trip advisor.


Se clasificaron los reviews en 2 categorias Positivo y negativo.
Si el review posee un rating mayor o igual a 40 es un review Positivo.
Si el review posee un rating menor a 40 es un review Negativo.


En base a los reviews que se scrappearon se construyo un vocabulario.
Despues para cada tipo de review se construyo un Bag of Words y se calculo probabilidad
para cada palabra.

Estas probabilidades se guardaron como nuestro modelo.


El proyecto cuenta con un api.

Un endpoint para entrenar el modelo.
- /train

Un endpoint para predecir si un review es bueno o malo 
- /predict?reivews="review to predict"