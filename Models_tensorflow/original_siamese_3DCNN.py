from tensorflow.keras.models import Model
from keras.layers import BatchNormalization, Input, concatenate, Conv2D, add, Conv3D, Reshape, SeparableConv2D, Dropout, MaxPool2D,MaxPool3D, UpSampling2D, ZeroPadding2D, Activation, ReLU, Lambda
from tensorflow import keras

class Siamese():  
    
        
        def __init__(self, params):
            self.params = params
            self.modelD = None
            
        def drop_out(self, x, drop_out = None):
            if drop_out: 
                x = Dropout(drop_out)(x, training = True)
        
            return x 
        

    
        def resblock_2D(self, num_filters, size_filter, stride_filter, x):
            """Residual block for 2D input excluding batch normalization layers"""
            Fx = Conv2D(filters=num_filters, kernel_size=size_filter, strides=stride_filter,padding='same', activation='relu', 
                        data_format="channels_last")(x)
            Fx = Conv2D(filters=num_filters, kernel_size=size_filter, padding='same', activation='relu', data_format="channels_last")(Fx)
            output = add([Fx, x])
            return output

        def resblock_2D_BN(self,num_filters, size_filter, stride_filter, x):
            """Residual block for 2D input including batch normalization layers"""
            Fx = Conv2D(filters=num_filters, kernel_size=size_filter, strides=stride_filter, padding='same', activation='relu', 
                        data_format="channels_last")(x)
            Fx = BatchNormalization()(Fx)
            Fx = Conv2D(filters=num_filters, kernel_size=size_filter, strides=stride_filter, padding='same', activation='relu', 
                        data_format="channels_last")(Fx)
            Fx = BatchNormalization()(Fx)
            output = add([Fx, x])
            return output

        def resblock_3D(self,num_filters, size_filter, stride_filter, x):
            """Residual block for 3D input excluding batch normalization layers"""
            Fx = Conv3D(filters=num_filters, kernel_size=size_filter, strides=stride_filter, padding='same', activation='relu', 
                        data_format="channels_last")(x)
            Fx = Conv3D(filters=num_filters, kernel_size=size_filter, strides=stride_filter, padding='same', activation='relu', 
                        data_format="channels_last")(Fx)
            output = add([Fx, x])
            return output
        def build_model(self):
            """The deep learning architecture gets defined here"""
            
            ## Input Optical Properties ##
            inOP_beg = Input(shape=(self.params['xX'],self.params['yY'],2))
            ## Input Multi-Dimensional Fluorescence ##
            inFL_beg = Input(shape=(self.params['xX'],self.params['yY'],self.params['nF']))
            
            
            ## NOTE: Batch normalization can cause instability in the validation loss

            ## Optical Properties Branch ##
            inOP = Conv2D(filters=self.params['nFilters2D'], kernel_size=self.params['kernelConv2D'], strides=self.params['strideConv2D'], 
                            padding='same', activation=self.params['activation'], data_format="channels_last")(inOP_beg)
            inOP = BatchNormalization()(inOP)  

            inOP = Dropout(0.5)(inOP)
            

            inOP = Conv2D(filters=int(self.params['nFilters2D']/2), kernel_size=self.params['kernelConv2D'], strides=self.params['strideConv2D'], 
                            padding='same', activation=self.params['activation'], data_format="channels_last")(inOP)
            

            inOP = BatchNormalization()(inOP)

            inOP = Dropout(0.5)(inOP)

            inOP = Conv2D(filters=int(self.params['nFilters2D']/2), kernel_size=self.params['kernelConv2D'], strides=self.params['strideConv2D'], 
                            padding='same', activation=self.params['activation'], data_format="channels_last")(inOP)
            

            inOP = BatchNormalization()(inOP)

            inOP = Dropout(0.5)(inOP)

            inOP = self.resblock_2D(int(self.params['nFilters2D']/2), self.params['kernelResBlock2D'], self.params['strideConv2D'], inOP)

            inOP = Dropout(0.5)(inOP)

            ## Fluorescence Input Branch ##
        
            inFL_beg = Input(shape=(self.params['xX'],self.params['yY'],self.params['nF'], 1))

            input_shape = inFL_beg.shape

            #reshape to allow 3D conv 
            #inFL = Reshape(shape = (self.params['xX'],self.params['yY'],1, self.params['nF']))(inFL_beg)
            inFL = Conv3D(filters=self.params['nFilters3D'], kernel_size=self.params['kernelConv3D'], strides=self.params['strideConv3D'], 
                            padding='same', activation=self.params['activation'], input_shape=input_shape[1:], data_format="channels_last")(inFL_beg)
            inFL = BatchNormalization()(inFL)

            inFL = Dropout(0.5)(inFL)

            #inFL = Conv2D(filters=int(self.params['nFilters2D']/2), kernel_size=self.params['kernelConv2D'], strides=self.params['strideConv2D'], 
            #                padding='same', activation=self.params['activation'], data_format="channels_last")(inFL)

            inFL = Conv3D(filters=self.params['nFilters3D'], kernel_size=self.params['kernelConv3D'], strides=self.params['strideConv3D'], 
                            padding='same', activation=self.params['activation'], input_shape=input_shape[1:], data_format="channels_last")(inFL)
            

            inFL = BatchNormalization()(inFL)

            inFL = Dropout(0.5)(inFL)

            #inFL = Conv2D(filters=int(self.params['nFilters2D']/2), kernel_size=self.params['kernelConv2D'], strides=self.params['strideConv2D'], 
            #                padding='same', activation=self.params['activation'], data_format="channels_last")(inFL)
            
            inFL = Conv3D(filters=self.params['nFilters3D'], kernel_size=self.params['kernelConv3D'], strides=self.params['strideConv3D'], 
                            padding='same', activation=self.params['activation'], input_shape=input_shape[1:], data_format="channels_last")(inFL)
            
            inFL = BatchNormalization()(inFL)
            inFL = Dropout(0.5)(inFL)

            inFL = Conv3D(filters=self.params['nFilters3D'], kernel_size=self.params['kernelConv3D'], strides=self.params['strideConv3D'], 
                            padding='same', activation=self.params['activation'], input_shape=input_shape[1:], data_format="channels_last")(inFL)
            
            inFL = BatchNormalization()(inFL)
            inFL = Dropout(0.5)(inFL)

            inFL = Conv3D(filters=self.params['nFilters3D'], kernel_size=self.params['kernelConv3D'], strides=self.params['strideConv3D'], 
                            padding='same', activation=self.params['activation'], input_shape=input_shape[1:], data_format="channels_last")(inFL)
            
            inFL = BatchNormalization()(inFL)
            inFL = Dropout(0.5)(inFL)

            #reshape to allow 2D conv 

            print(inFL.shape)

            inFL = Reshape((self.params['xX'],self.params['yY'],self.params['nF']))(inFL)

            inFL = self.resblock_2D(int(self.params['nFilters2D']/2), self.params['kernelResBlock2D'], self.params['strideConv2D'], inFL)
            
            inFL = Dropout(0.5)(inFL)

            ## Concatenate Branch ##
            concat = concatenate([inOP,inFL],axis=-1)
            

            concat = SeparableConv2D(filters=self.params['nFilters2D'], kernel_size=self.params['kernelConv2D'], 
                                        strides=self.params['strideConv2D'], padding='same', activation=self.params['activation'], 
                                        data_format="channels_last")(concat)
            
   
            concat = BatchNormalization()(concat)
            concat = self.resblock_2D(self.params['nFilters2D'], self.params['kernelResBlock2D'], self.params['strideConv2D'], concat) 

            ## Quantitative Fluorescence Output Branch ##
            outQF = Conv2D(filters=64, kernel_size=self.params['kernelConv2D'], strides=self.params['strideConv2D'], padding='same', 
                            activation=self.params['activation'], data_format="channels_last")(concat)

            outDF = Conv2D(filters=64, kernel_size=self.params['kernelConv2D'], strides=self.params['strideConv2D'], padding='same', 
                            activation=self.params['activation'], data_format="channels_last")(concat)

            outQF = BatchNormalization()(outQF)
            outQF = Conv2D(filters=32, kernel_size=self.params['kernelConv2D'], strides=self.params['strideConv2D'], padding='same', 
                            activation=self.params['activation'], data_format="channels_last")(outQF)        
            
            outQF = Conv2D(filters=1, kernel_size=self.params['kernelConv2D'], strides=self.params['strideConv2D'], padding='same', 
                            data_format="channels_last")(outQF)

            ## Depth Map Output Branch ##

            outDF = BatchNormalization()(outDF)
            outDF = Conv2D(filters=32, kernel_size=self.params['kernelConv2D'], strides=self.params['strideConv2D'], padding='same', 
                            activation=self.params['activation'], data_format="channels_last")(outDF)       
            outDF = Conv2D(filters=1, kernel_size=self.params['kernelConv2D'], strides=self.params['strideConv2D'], padding='same', 
                            data_format="channels_last")(outDF)

            self.modelD = Model(inputs=[inOP_beg,inFL_beg], outputs=[outQF, outDF])#,outFL])
            self.modelD.compile(loss=['mae', 'mae'],
                            optimizer=getattr(keras.optimizers,self.params['optimizer'])(learning_rate=self.params['learningRate']),
                            metrics=['mae', 'mae'])
            self.modelD.summary()
