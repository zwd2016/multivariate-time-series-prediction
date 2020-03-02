#! -*- coding: utf-8 -*-
"""
Created on Apr 19 14:11:10 2019

@author: Wendong Zheng
"""
from keras import backend as K
from keras.engine.topology import Layer
from keras.layers import Input,LSTM,Merge,Add
from keras.models import Model
class My_Dot(Layer):  
    def __init__(self,output_dim,**kwargs):
        self.output_dim = output_dim
        super(My_Dot, self).__init__(**kwargs)
    def build(self, input_shape):
        self.kernel = self.add_weight(name='kernel',shape=(input_shape[-1],self.output_dim),
                                      initializer='uniform',trainable=True)
        super(My_Dot, self).build(input_shape)

    def call(self, inputs, **kwargs):
        print('Mydot:',K.dot(inputs,self.kernel))
        return K.dot(inputs,self.kernel)

    def compute_output_shape(self, input_shape):
        return (input_shape[0],input_shape[1],self.output_dim)

class My_Transpose(Layer): 
    def __init__(self,axis,**kwargs):
        self.axis = axis
        super(My_Transpose, self).__init__(**kwargs)
    def build(self, input_shape):
        super(My_Transpose, self).build(input_shape)
    def call(self, inputs, **kwargs):
        return K.permute_dimensions(inputs,pattern=self.axis)
    def compute_output_shape(self, input_shape):
        return (input_shape[self.axis[0]],input_shape[self.axis[1]],input_shape[self.axis[2]])



