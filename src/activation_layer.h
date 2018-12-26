#ifndef ACTIVATION_LAYER_H
#define ACTIVATION_LAYER_H

#include "activations.h"
#include "layer.h"
#include "network.h"



layer make_activation_layer(int batch, int inputs, ACTIVATION activation, int out_c, int out_w, int out_h, int prelu_Switch_Flag);

void forward_activation_layer(layer l, network net);
void backward_activation_layer(layer l, network net);
void update_activation_layer(layer l, update_args a);


#ifdef GPU
void forward_activation_layer_gpu(layer l, network net);
void backward_activation_layer_gpu(layer l, network net);
void update_activation_layer_gpu(layer l, update_args a);

#endif

#endif

