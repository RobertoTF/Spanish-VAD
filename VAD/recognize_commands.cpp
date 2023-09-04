/* recognize_commands.cpp */
//========================//

/*
    Desarrollo de un algoritmo de detección de voz en castellano con TinyML
    Sep-23, Roberto Tejedor Ferrero
    Máster Universitario en Análisis y Visualización de Datos Masivos
    Escuela Superior de Ingeniería y Tecnología
    Universidad Internacional de La Rioja

    Adaptado de https://github.com/tensorflow/tflite-micro-arduino-examples

*/

/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#include "recognize_commands.h"

#include <limits>

#undef DEBUG_MICRO_SPEECH

RecognizeCommands::RecognizeCommands(int32_t average_window_duration_ms,
                                     uint8_t detection_threshold,
                                     int32_t suppression_ms,
                                     int32_t minimum_count)
    : average_window_duration_ms_(average_window_duration_ms),
      detection_threshold_(detection_threshold),
      suppression_ms_(suppression_ms),
      minimum_count_(minimum_count),
      previous_results_() {
  previous_top_label_ = kCategoryLabels[0];  // silence
  previous_top_label_time_ = std::numeric_limits<int32_t>::min();
}

TfLiteStatus RecognizeCommands::ProcessLatestResults(
    const TfLiteTensor* latest_results, const int32_t current_time_ms,
    const char** found_command, uint8_t* score, bool* is_new_command) {

  static int32_t VAD_time_extension = 0;  // Tiempo de VAD después del fin de tramo vocal
  static int32_t init_vocal_time_ms = 0;  // Inicio de tramo vocálico
  static bool previous_vocal = false;     // Resultado de anterior inferencia
 
  if ((latest_results->dims->size != 2) ||
      (latest_results->dims->data[0] != 1) ||
      (latest_results->dims->data[1] != (kCategoryCount-1))) {
    MicroPrintf(
        "The results for recognition should contain %d elements, but there are "
        "%d in an %d-dimensional shape",
        kCategoryCount, latest_results->dims->data[1],
        latest_results->dims->size);
    return kTfLiteError;
  }

  if (latest_results->type != kTfLiteUInt8) {
    MicroPrintf(
        "The results for recognition should be uint8_t elements, but are %d",
        latest_results->type);
    return kTfLiteError;
  }
  *is_new_command = false;
  
  // Se comparan las dos salidas de la red neuronal: primer valor "No vocal" 
  // y segundo valor "Vocal". 	
  if (*((latest_results->data.uint8)+1) > *latest_results->data.uint8) {  // Vocal
    *found_command = kCategoryLabels[0];  
    if (!previous_vocal) { // Primera vocal -> inicio de tramo vocálico
      init_vocal_time_ms = current_time_ms; // Se almacen el tiempo de inicio
      previous_vocal = true;  // Se preserva el estado para la siguiente inferencia  
      *is_new_command = true; // Actualizar leds
    }
  }
  else {  // No vocal
    if (previous_vocal) { // Fin de tramo vocálico
      previous_vocal = false;
      *found_command = kCategoryLabels[1];
      *is_new_command = true;  // Actualizar leds
      // Se calcula el tiempo que ha de mantenerse activa la salida VAD -> tramo consonántico estimado
      VAD_time_extension = current_time_ms + (uint32_t)((1.64f)*((float)(current_time_ms-init_vocal_time_ms)));
    }
    else if (VAD_time_extension > current_time_ms) // Dentro de tramo consonántico
      *found_command = kCategoryLabels[1];
    else {  // Fin de tramo consonántico -> Silencio 
      *found_command = kCategoryLabels[2];
    }
  }
  return kTfLiteOk;
}
