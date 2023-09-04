/* micro_features_micro_model_settings.h */
//=======================================//

/*
    Desarrollo de un algoritmo de detección de voz en castellano con TinyML
    Sep-23, Roberto Tejedor Ferrero
    Máster Universitario en Análisis y Visualización de Datos Masivos
    Escuela Superior de Ingeniería y Tecnología
    Universidad Internacional de La Rioja

    Adaptado de https://github.com/tensorflow/tflite-micro-arduino-examples

*/

/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef TENSORFLOW_LITE_MICRO_EXAMPLES_MICRO_SPEECH_MICRO_FEATURES_MICRO_MODEL_SETTINGS_H_
#define TENSORFLOW_LITE_MICRO_EXAMPLES_MICRO_SPEECH_MICRO_FEATURES_MICRO_MODEL_SETTINGS_H_

// Keeping these as constant expressions allow us to allocate fixed-sized arrays
// on the stack for our working memory.

// Ventanas de audio de 256 muestras -> FFT de 256 puntos
constexpr int kMaxAudioSampleSize = 256;   
constexpr int kAudioSampleFrequency = 16000;

// Se mantiene la nomenclatura original
constexpr int kFeatureSliceSize = 129;  // Número de parámetros resultante de la FFT
constexpr int kFeatureSliceCount = 1;   // Después de cada ventana -> inferencia
constexpr int kFeatureElementCount = (kFeatureSliceSize * kFeatureSliceCount);
constexpr int kFeatureSliceStrideMs = 8;   // Desplazamiento para el enventanado
constexpr int kFeatureSliceDurationMs = 16;  // Tamaño de ventana -> 256 muestras

// Variables for the model's output categories.
constexpr int kSilenceIndex = 0;
constexpr int kUnknownIndex = 1;
// If you modify the output categories, you need to update the following values.
constexpr int kCategoryCount = 3;
extern const char* kCategoryLabels[kCategoryCount];

#endif  // TENSORFLOW_LITE_MICRO_EXAMPLES_MICRO_SPEECH_MICRO_FEATURES_MICRO_MODEL_SETTINGS_H_
