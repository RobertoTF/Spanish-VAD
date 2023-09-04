/* frontend.c */
//============//

/*
    Desarrollo de un algoritmo de detección de voz en castellano con TinyML
    Sep-23, Roberto Tejedor Ferrero
    Máster Universitario en Análisis y Visualización de Datos Masivos
    Escuela Superior de Ingeniería y Tecnología
    Universidad Internacional de La Rioja

    Adaptado de https://github.com/tensorflow/tflite-micro-arduino-examples

*/

/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/lite/experimental/microfrontend/lib/frontend.h"

#include "tensorflow/lite/experimental/microfrontend/lib/bits.h"

struct FrontendOutput FrontendProcessSamples(struct FrontendState* state,
                                             const int16_t* samples,
                                             size_t num_samples,
                                             size_t* num_samples_read) {
  struct FrontendOutput output;
  output.values = NULL;
  output.size = 0;

  // Try to apply the window - if it fails, return and wait for more data.
  if (!WindowProcessSamples(&state->window, samples, num_samples,
                            num_samples_read)) {
    return output;
  }

  // Apply the FFT to the window's output (and scale it so that the fixed point
  // FFT can have as much resolution as possible).
  int input_shift =
      15 - MostSignificantBit32(state->window.max_abs_output_value);
  FftCompute(&state->fft, state->window.output, input_shift);

  // We can re-ruse the fft's output buffer to hold the energy.
  int32_t* energy = (int32_t*)state->fft.output;

  // Se calculan el cuadrado de los módulos de los valores complejos de la FFT	
  FilterbankConvertFftComplexToEnergy(&state->filterbank, state->fft.output,
                                      energy);
  // Se ha adaptado la función de manera que se limita a transmitir los valores
  // de los cuadrados de los modulos
  FilterbankAccumulateChannels(&state->filterbank, energy);
  // Se ha adaptado la función de manera que calcula la raiz cuadrada
  // resultando el módulo de los valores FFT -> entrada para la red neuronal
  uint32_t* scaled_filterbank = FilterbankSqrt(&state->filterbank, input_shift);

  output.size = state->filterbank.num_channels;
  output.values = scaled_filterbank; //logged_filterbank;
  return output;
}

void FrontendReset(struct FrontendState* state) {
  WindowReset(&state->window);
  FftReset(&state->fft);
  FilterbankReset(&state->filterbank);
  NoiseReductionReset(&state->noise_reduction);
}
