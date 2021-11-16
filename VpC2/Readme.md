# Visión por computadora 2

_Cátedra del Grado de Especialización en Inteligencia Artificial (CEIA). Universidad de Buenos Aires (UBA)_

Este repositorio se utiliza para documentar los desafíos desarrollados durante el aprendizaje del curso VPC2. 📖


---

## Tarea N°1: 
Utilizando el dataset de señas de manos, crear una red neuronal con caracteristicas similares a las vistas en clase (**LeNet-5, AlexNet y VGG**) y entrenarla hasta obtener un accuracy de, como minimo, 85% evitando sobreentrenamiento. Aplicar las técnicas de **data augmentation** que consideren necesarias y el uso de **ImageDataGenerator** de Keras.

**Conclusión:** Se realiza la tarea de dos maneras distintas en ambos procesos utilice la red de 4 capas convolucionales seguidas de maxpooling, Con la única diferencia de que en la primera utilicé ImageDataGenerator solamente para un reescalado mientras que en el segundo caso lo use para variar dimensiones, zoom y brillo entre otros. Al utilizar el mismo modelo con los mismos hiper parámetros encontramos qué el trabajo sin DataAugmentation logra mayor precisión y su evolución con las épocas es más suave mientras que al utilizar DataAugmentation, al evaluarlo con el set de validación tiene más ruido y menos accuracy.

🖇️[Link to code](https://github.com/sebasesto/CEIA/blob/master/VpC2/Clase_2_Tarea_con_DataAugmentation_Sesto.ipynb)

---

## Author  ✒️
Federico Sebastián Sesto

## Contact 📌
Contact me by mail _sestofederico@gmail.com_ or by my personal [LinkedIn](https://www.linkedin.com/in/federico-sebasti%C3%A1n-sesto/)

---
