
from keras.models import Model, load_model
from keras.layers import BatchNormalization, Input, concatenate, Conv2D, add, Conv3D, Reshape, SeparableConv2D, Dropout, MaxPool2D,MaxPool3D, UpSampling2D, ZeroPadding2D, Activation, ReLU, Lambda
from keras.preprocessing import image
import keras

import numpy as np 
import tensorflow as tf 
class ModelInit():  

        def __init__(self):
                super().__init__()
        
        #helper function: attention gate 
        def attention_gate(self, g, s, num_filters):
            Wg = Conv2D(num_filters, 1, strides = (2,2), padding="valid")(g)
            Ws = Conv2D(num_filters, 1, padding="same")(s)

            out = Activation("relu")(Wg + Ws)
            out = Conv2D(1, 1, padding="same")(out)
            out = Activation("sigmoid")(out)
            out = UpSampling2D()(out)
            return out * g
        
        def Model_tf(self):


                """The deep learning architecture gets defined here"""

                ## Input Multi-Dimensional Fluorescence ##
                inFL_beg = Input(shape=(self.params['xX'],self.params['yY'],self.params['nF']))

                ## NOTE: Batch normalization can cause instability in the validation loss
                
                ## Fluorescence Input Branch ##
                #inFL = Reshape((inFL_beg.shape[1], inFL_beg.shape[2], 1,inFL_beg.shape[3]))(inFL_beg)
                input_shape = inFL_beg.shape
                        
                random_mask = np.random.choice([1, 0], size=(32, self.params['xX'],self.params['yY'],self.params['nF']), p=[1 - 0.5, 0.5])
                mask = tf.constant(random_mask, dtype=tf.float32)  # shape: (xX, yY, nF, 1)

                def apply_mask(x):
                        return x * mask

                inFL_beg = Lambda(apply_mask)(inFL_beg)

                inFL_beg = Reshape((inFL_beg.shape[1], inFL_beg.shape[2], inFL_beg.shape[3], 1))(inFL_beg)


                inFL = Conv3D(filters=self.params['nFilters3D']//2, kernel_size=self.params['kernelConv3D'], strides=self.params['strideConv3D'], 
                        padding='same', activation=self.params['activation'], input_shape=input_shape[1:], data_format="channels_last")(inFL_beg)
                #inFL = Dropout(0.5)(inFL)

                inFL = Conv3D(filters=int(self.params['nFilters3D']/2), kernel_size=self.params['kernelConv3D'], strides=self.params['strideConv3D'], 
                        padding='same', activation=self.params['activation'], data_format="channels_last")(inFL)
                #inFL = Dropout(0.5)(inFL)
                inFL = Conv3D(filters=int(self.params['nFilters3D']/2), kernel_size=self.params['kernelConv3D'], strides=self.params['strideConv3D'], 
                        padding='same', activation=self.params['activation'], data_format="channels_last")(inFL)
                #inFL = Dropout(0.5)(inFL)

                ## Concatenate Branch ##
                inFL = Reshape((inFL.shape[1], inFL.shape[2], inFL.shape[3] * inFL.shape[4]))(inFL)

                Max_Pool_1 = MaxPool2D()(inFL)

                Conv_1 = Conv2D(filters=256, kernel_size=self.params['kernelConv2D'], strides=self.params['strideConv2D'], padding='same', 
                        activation=self.params['activation'], data_format="channels_last")(Max_Pool_1)
                Conv_1 = Conv2D(filters=256, kernel_size=self.params['kernelConv2D'], strides=self.params['strideConv2D'], padding='same', 
                        activation=self.params['activation'], data_format="channels_last")(Conv_1)
                
                Max_Pool_2 = MaxPool2D()(Conv_1)

                Conv_2 = Conv2D(filters=512, kernel_size=self.params['kernelConv2D'], strides=self.params['strideConv2D'], padding='same', 
                        activation=self.params['activation'], data_format="channels_last")(Max_Pool_2)
                Conv_2 = Conv2D(filters=512, kernel_size=self.params['kernelConv2D'], strides=self.params['strideConv2D'], padding='same', 
                        activation=self.params['activation'], data_format="channels_last")(Conv_2)

                Max_Pool_3 = MaxPool2D()(Conv_2)

                Conv_3 = Conv2D(filters=1024, kernel_size=(self.params['kernelConv2D']), strides=self.params['strideConv2D'], padding='same', 
                        activation=self.params['activation'], data_format="channels_last")(Max_Pool_3)
                Conv_3 = Conv2D(filters=1024, kernel_size=self.params['kernelConv2D'], strides=self.params['strideConv2D'], padding='same', 
                        activation=self.params['activation'], data_format="channels_last")(Conv_3)

                #decoder 

                #adjust size of Conv_2
                long_path_1 = Conv_2[:,0:Conv_2.shape[1] - 1, 0:Conv_2.shape[2] - 1, :]
                attention_1 = self.attention_gate(long_path_1, Conv_3, 512)

                Up_conv_1 = UpSampling2D()(Conv_3)

                
                Up_conv_1 = Conv2D(filters=512, kernel_size = (2,2), strides=(1,1), padding='same', 
                        activation=self.params['activation'], data_format="channels_last")(Up_conv_1)

                #attention block 
                concat_1 = concatenate([Up_conv_1,attention_1],axis=-1)

                Conv_4 = Conv2D(filters=512, kernel_size=self.params['kernelConv2D'], strides=self.params['strideConv2D'], padding='same', 
                        activation=self.params['activation'], data_format="channels_last")(concat_1)

                Conv_4 = Conv2D(filters=512, kernel_size=self.params['kernelConv2D'], strides=self.params['strideConv2D'], padding='same', 
                                activation=self.params['activation'], data_format="channels_last")(Conv_4)
                
                long_path_2 = Conv_1
                Conv_4_zero_pad = ZeroPadding2D(padding = ((1,0), (1,0)))(Conv_4)

                attention_2 = self.attention_gate(long_path_2, Conv_4_zero_pad, 256)

                Up_conv_2 = UpSampling2D()(Conv_4)

                Up_conv_2 = Conv2D(filters=256, kernel_size = (2,2), strides=(1,1), padding='same', 
                        activation=self.params['activation'], data_format="channels_last")(Up_conv_2)

                Up_conv_2 = ZeroPadding2D()(Up_conv_2)

                concat_2 = concatenate([Up_conv_2,attention_2],axis=-1)

                Conv_5 = Conv2D(filters=256, kernel_size=self.params['kernelConv2D'], strides=self.params['strideConv2D'], padding='same', 
                        activation=self.params['activation'], data_format="channels_last")(concat_2)
                Conv_5 = Conv2D(filters=256, kernel_size=self.params['kernelConv2D'], strides=self.params['strideConv2D'], padding='same', 
                        activation=self.params['activation'], data_format="channels_last")(Conv_5)
                
                long_path_3 = ZeroPadding2D(padding = ((1,0), (1,0)))(inFL)
                Conv_5_zero_pad = ZeroPadding2D(padding = ((1,0), (1,0)))(Conv_5)

                attention_3 = self.attention_gate(long_path_3, Conv_5_zero_pad, 128)

                Up_conv_3 = UpSampling2D()(Conv_5)
                Up_conv_3 = Conv2D(filters=128, kernel_size = (2,2), strides=(1,1), padding='same', 
                        activation=self.params['activation'], data_format="channels_last")(Up_conv_3)
                                
                Up_conv_3 = ZeroPadding2D(padding = ((1,0), (1,0)))(Up_conv_3)

                attention_3 = attention_3[:,0:attention_3.shape[1] - 1, 0:attention_3.shape[2] - 1, :]
                concat_3 = concatenate([Up_conv_3,attention_3],axis=-1)  

                Conv_6 = Conv2D(filters=128, kernel_size=self.params['kernelConv2D'], strides=self.params['strideConv2D'], padding='same', 
                        activation=self.params['activation'], data_format="channels_last")(concat_3)
                
                ## Depth Fluorescence Output Branch ##
                #first DF layer 
                outFL = Conv2D(filters=64, kernel_size=self.params['kernelConv2D'], strides=self.params['strideConv2D'], padding='same', 
                        activation=self.params['activation'], data_format="channels_last")(Conv_6)

                outFL = Conv2D(filters=32, kernel_size=self.params['kernelConv2D'], strides=self.params['strideConv2D'], padding='same', 
                        activation=self.params['activation'], data_format="channels_last")(outFL)

                #outDF = BatchNormalization()(outDF)

                
                outFL = Conv2D(filters=6, kernel_size=self.params['kernelConv2D'], strides=self.params['strideConv2D'], padding='same', 
                        data_format="channels_last")(outFL)

                ## Defining and compiling the model ##
                self.modelD = Model(inputs=[inFL_beg], outputs=[outFL])#,outFL])
                self.modelD.compile(loss=['mae'],
                        optimizer=getattr(keras.optimizers,self.params['optimizer'])(learning_rate=self.params['learningRate']),
                        metrics=['mae'])
                self.modelD.summary()
                return None



