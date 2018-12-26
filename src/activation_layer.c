#include "activation_layer.h"
#include "utils.h"
#include "blas.h"
#include "gemm.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void update_activation_layer(layer l, update_args a)
{
    float learning_rate = a.learning_rate*l.learning_rate_scale;
    float momentum = a.momentum;
    float decay = a.decay;
    int batch = a.batch;
/*     axpy_cpu(l.outputs, learning_rate/batch, l.bias_updates, 1, l.biases, 1);
    scal_cpu(l.outputs, momentum, l.bias_updates, 1); */
/* 
    if(l.batch_normalize){
        axpy_cpu(l.outputs, learning_rate/batch, l.scale_updates, 1, l.scales, 1);
        scal_cpu(l.outputs, momentum, l.scale_updates, 1);
    } */

    axpy_cpu(l.inputs*l.outputs, -decay*batch, l.weights, 1, l.weight_updates, 1);
    axpy_cpu(l.inputs*l.outputs, learning_rate/batch, l.weight_updates, 1, l.weights, 1);
    scal_cpu(l.inputs*l.outputs, momentum, l.weight_updates, 1);
}


layer make_activation_layer(int batch, int inputs, ACTIVATION activation, int out_c, int out_w, int out_h, int prelu_Switch_Flag)
{
    layer l = {0};
    l.type = ACTIVE;
    l.inputs = inputs;
    l.outputs = inputs;
    l.batch=batch;
    l.out_w = out_w;
    l.out_h = out_h;
    l.out_c = out_c;
    l.prelu_Switch_Flag = prelu_Switch_Flag;

    l.output = calloc(batch*inputs, sizeof(float*));
    l.delta = calloc(batch*inputs, sizeof(float*));
    l.weights = calloc(batch*out_c, sizeof(float*)); 
    l.weight_updates = calloc(batch*out_c, sizeof(float*));
    int i;
    
#if 0
    float scale = sqrt(2./(out_h*out_w));
    for(i = 0; i < out_c; i++)
    {
    	l.weights[i] =  scale * rand_scale(out_c);
    }
#endif

    for(i = 0; i < out_c; i++)
    {
    		l.weights[i] =  0.25;
    }

    l.forward = forward_activation_layer;
    l.backward = backward_activation_layer;
    l.update = update_activation_layer;
#ifdef GPU
    l.forward_gpu = forward_activation_layer_gpu;
    l.backward_gpu = backward_activation_layer_gpu;
    l.weights_gpu = cuda_make_array(l.weights, out_c*batch);
    l.weight_updates_gpu = cuda_make_array(l.weight_updates, out_c*batch);
    l.output_gpu = cuda_make_array(l.output, inputs*batch);
    l.delta_gpu = cuda_make_array(l.delta, inputs*batch);
    l.update_gpu = update_activation_layer_gpu;
#endif
    l.activation = activation;
    fprintf(stderr, "Activation Layer: %d inputs\n", inputs);
    return l;
}


void forward_activation_layer(layer l, network net)
{
    copy_cpu(l.outputs*l.batch, net.input, 1, l.output, 1);
    if(l.prelu_Switch_Flag)
    {
    	int i, j, k;
		for(i = 0; i < l.batch; i++)
		{
			for(j = 0; j<l.out_c; j++)
			{
				int in_index = j + i * l.out_c;
				for(k = 0; k< l.out_w * l.out_h; k++)
				{
					int out_index = k + in_index*l.out_w * l.out_h;
					int factor =  l.output[out_index] >= 0 ? 1 : l.weights[in_index];
					l.output[out_index] *= factor;
				}
			}
		}
    }
    else
    {
    	activate_array(l.output, l.outputs*l.batch, l.activation);
    }
}

void backward_activation_layer(layer l, network net)
{
    if(l.prelu_Switch_Flag)
    {
		int i, j, k;
		//calculate the backward delta
		for(i = 0; i < l.batch; i++)
		{
			for(j = 0; j < l.out_c; j++)
			{
				int in_index = j + i * l.out_c;
				for(k = 0; k < l.out_w*l.out_h ; k++)
				{
					int out_index = k + in_index*l.out_w*l.out_h;
					int back_factor = net.input[out_index] >= 0 ? 1 : l.weights[in_index];
					if(net.delta) net.delta[out_index] += l.delta[out_index] * back_factor;
					
					int weights_updates_factor = net.input[out_index] >= 0 ? 0 : net.input[out_index];
					l.weight_updates[in_index] +=  l.delta[out_index]*weights_updates_factor;  //is this right?
				}
			}
		}
    }
    else
    {
    	gradient_array(l.output, l.outputs*l.batch, l.activation, l.delta);
    	copy_cpu(l.outputs*l.batch, l.delta, 1, net.delta, 1);
    }
}

#ifdef GPU

void pull_activation_layer(layer l)
{
    cuda_pull_array(l.weights_gpu, l.weights, l.out_c);
    //cuda_pull_array(l.biases_gpu, l.biases, l.outputs);
    cuda_pull_array(l.weight_updates_gpu, l.weight_updates, l.out_c);
    //cuda_pull_array(l.bias_updates_gpu, l.bias_updates, l.outputs);
/*     if (l.batch_normalize){
        cuda_pull_array(l.scales_gpu, l.scales, l.outputs);
        cuda_pull_array(l.rolling_mean_gpu, l.rolling_mean, l.outputs);
        cuda_pull_array(l.rolling_variance_gpu, l.rolling_variance, l.outputs);
    } */
}

void push_activation_layer(layer l)
{
    cuda_push_array(l.weights_gpu, l.weights, l.out_c);
    //cuda_push_array(l.biases_gpu, l.biases, l.outputs);
    cuda_push_array(l.weight_updates_gpu, l.weight_updates, l.out_c);
    //cuda_push_array(l.bias_updates_gpu, l.bias_updates, l.outputs);
/*     if (l.batch_normalize){
        cuda_push_array(l.scales_gpu, l.scales, l.outputs);
        cuda_push_array(l.rolling_mean_gpu, l.rolling_mean, l.outputs);
        cuda_push_array(l.rolling_variance_gpu, l.rolling_variance, l.outputs);
    } */
}

void forward_activation_layer_gpu(layer l, network net)
{
 	copy_gpu(l.outputs*l.batch, net.input_gpu, 1, l.output_gpu, 1);
	if(l.prelu_Switch_Flag)
	{
		forward_activation_layer_kernel_gpu(l.out_c*l.batch, l.out_w, l.out_h, l.out_c, net.input_gpu, l.output_gpu, l.weights_gpu);	
	}
	else
	{
		activate_array_gpu(l.output_gpu, l.outputs*l.batch, l.activation);
	}
}

void backward_activation_layer_gpu(layer l, network net)
{
	if(l.prelu_Switch_Flag)
	{    
		backward_activation_layer_kernel_gpu(l.out_c*l.batch, l.out_w, l.out_h, l.out_c, net.input_gpu, l.output_gpu, net.delta_gpu, l.delta_gpu, l.weights_gpu, l.weight_updates_gpu);
	}
	else
	{	
		gradient_array_gpu(l.output_gpu, l.outputs*l.batch, l.activation, l.delta_gpu);
		copy_gpu(l.outputs*l.batch, l.delta_gpu, 1, net.delta_gpu, 1);
	} 
}

void update_activation_layer_gpu(layer l, update_args a)
{
  float learning_rate = a.learning_rate*l.learning_rate_scale;
  float momentum = a.momentum;
  float decay = a.decay;
  int batch = a.batch;
	
/* 	axpy_gpu(l.outputs, learning_rate/batch, l.bias_updates_gpu, 1, l.biases_gpu, 1);
	scal_gpu(l.outputs, momentum, l.bias_updates_gpu, 1); */

/*         if(l.batch_normalize){
            axpy_gpu(l.outputs, learning_rate/batch, l.scale_updates_gpu, 1, l.scales_gpu, 1);
            scal_gpu(l.outputs, momentum, l.scale_updates_gpu, 1);
        } */
	//printf("start to update prelu weights\n");
	axpy_gpu(l.out_c, -decay*batch, l.weights_gpu, 1, l.weight_updates_gpu, 1);
	axpy_gpu(l.out_c, learning_rate/batch, l.weight_updates_gpu, 1, l.weights_gpu, 1);
	scal_gpu(l.out_c, momentum, l.weight_updates_gpu, 1);
	cuda_pull_array(l.weights_gpu, l.weights, l.out_c);
	int i;
	for(i = 0; i < l.out_c; i++)
	{
		if(l.weights[i] < 0)
		{
			l.weights[i] = 0.01;
		}
		if(l.weights[i] > 1)
		{
			l.weights[i] = 0.99;
		}
	}
	cuda_push_array(l.weights_gpu, l.weights, l.out_c);
}

#endif
