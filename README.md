<!-- mdformat off(b/169948621#comment2) -->

# Spanish VAD on Arduino Nano 33 BLE Sense

Desarrollo de un algoritmo de detección de voz en castellano con TinyML

Adaptado de https://github.com/tensorflow/tflite-micro-arduino-examples

Implementación de un algoritmo VAD en castellano basado en la detección
de fonemas vocálicos por medio de una red neuronal y la inferencia de
presencia de voz aprovechando las caraterísticas fonéticas del castellano.

La aplicación activará el LED verde al detectar una vocal.
Activará el LED azul cuando supone la presencia de voz en ausencia de vocal.
Si estima que no hay ninguna voz presenta, dejará los LED apagados.
