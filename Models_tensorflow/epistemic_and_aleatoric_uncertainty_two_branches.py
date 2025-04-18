
from keras.models import Model, load_model
from keras.layers import BatchNormalization, Input, concatenate, Conv2D, add, Conv3D, Reshape, SeparableConv2D, Dropout, MaxPool2D, UpSampling2D, ZeroPadding2D, Activation, SpatialDropout2D
from keras.preprocessing import image
import keras
import tensorflow as tf

class MonteCarloDropout(Dropout):
    def call(self, inputs):
        return super().call(inputs, training=True)
    
        
class ModelInit():  

    def attention_gate(self, g, s, num_filters):
        print("g in shape: ", g.shape)
        Wg = Conv2D(num_filters, 1, strides = (2,2), padding="valid")(g)
        print("Wg shape: ", Wg.shape)
        #Wg = BatchNormalization()(Wg)
        print("s in shape: ", s.shape)

        Ws = Conv2D(num_filters, 1, padding="same")(s)
        #Ws = BatchNormalization()(Ws)
        print("Ws shape: ", Ws.shape)

        out = Activation("relu")(Wg + Ws)
        out = Conv2D(1, 1, padding="same")(out)
        out = Activation("sigmoid")(out)
        out = UpSampling2D()(out)
        print("out shape: ", out.shape)
        print("g shape: ", g.shape)

        return out * g

    def drop_out(self, x, drop_out):
        if drop_out: 
            x = MonteCarloDropout(0.5)(x, training = True)

        return x 


    def Model(self):
            """The deep learning architecture gets defined here"""

            drop_out = 0.5

            print("drop_out", drop_out)
            inOP_beg = Input(shape=(self.params['xX'],self.params['yY'],2))
            ## Input Multi-Dimensional Fluorescence ##
            inFL_beg = Input(shape=(self.params['xX'],self.params['yY'],self.params['nF']))
            #inFL = FlData

            ## NOTE: Batch normalization can cause instability in the validation loss

            ## Optical Properties Branch ##
            inOP = Conv2D(filters=self.params['nFilters2D'], kernel_size=self.params['kernelConv2D'], strides=self.params['strideConv2D'], 
                          padding='same', activation=self.params['activation'], data_format="channels_last", kernel_initializer='he_normal')(inOP_beg)
            #outOP1 = inOP
            inOP = BatchNormalization()(inOP)  

            inOP = self.drop_out(inOP, drop_out) #drop out 1
            inOP = BatchNormalization()(inOP)

            inOP = Conv2D(filters=self.params['nFilters2D']//2, kernel_size=self.params['kernelConv2D'], strides=self.params['strideConv2D'], 
                          padding='same', activation=self.params['activation'], data_format="channels_last", kernel_initializer='he_normal')(inOP)
            #outOP2 = inOP

            inOP = BatchNormalization()(inOP)

            inOP = self.drop_out(inOP, drop_out) #drop out 1

            inOP = Conv2D(filters=self.params['nFilters2D']//2, kernel_size=self.params['kernelConv2D'], strides=self.params['strideConv2D'], 
                          padding='same', activation=self.params['activation'], data_format="channels_last", kernel_initializer='he_normal')(inOP)
            #outOP3 = inOP
            inOP = self.drop_out(inOP, drop_out) #drop out 1

            inOP = self.resblock_2D_dropout(self.params['nFilters2D']//2, self.params['kernelResBlock2D'], self.params['strideConv2D'], inOP, drop_out)

            ## Fluorescence Input Branch ##
            input_shape = inFL_beg.shape

            inFL = Conv2D(filters=self.params['nFilters2D'], kernel_size=self.params['kernelConv2D'], strides=self.params['strideConv2D'], 
                          padding='same', activation=self.params['activation'], input_shape=input_shape, data_format="channels_last", kernel_initializer='he_normal')(inFL_beg)

            inFL = BatchNormalization()(inFL)

            inFL = self.drop_out(inFL, drop_out) #drop out 1

            inFL = Conv2D(filters=self.params['nFilters2D']//2, kernel_size=self.params['kernelConv2D'], strides=self.params['strideConv2D'], 
                          padding='same', activation=self.params['activation'], data_format="channels_last", kernel_initializer='he_normal')(inFL)
            #outFL2 = inFL
            inFL = BatchNormalization()(inFL)

            inFL = self.drop_out(inFL, drop_out) #drop out 1

            inFL = Conv2D(filters=self.params['nFilters2D']//2, kernel_size=self.params['kernelConv2D'], strides=self.params['strideConv2D'], 
                          padding='same', activation=self.params['activation'], data_format="channels_last", kernel_initializer='he_normal')(inFL)
            #outFL3 = inFL
            inFL = BatchNormalization()(inFL)

            inFL = self.drop_out(inFL, drop_out) #drop out 1

            inFL = self.resblock_2D_dropout(self.params['nFilters2D']//2, self.params['kernelResBlock2D'], self.params['strideConv2D'], inFL, drop_out)

            ## Concatenate Branch ##
            concat = concatenate([inOP,inFL],axis=-1)
            concat = self.drop_out(concat, drop_out) #drop out 3

            concat = SeparableConv2D(filters=self.params['nFilters2D'], kernel_size=self.params['kernelConv2D'], 
                                     strides=self.params['strideConv2D'], padding='same', activation=self.params['activation'], 
                                     data_format="channels_last")(concat)
            concat = BatchNormalization()(concat)

            concat = self.drop_out(concat, drop_out) #drop out 3

            #concat = BatchNormalization()(concat)
            concat = self.resblock_2D_dropout(self.params['nFilters2D'], self.params['kernelResBlock2D'], self.params['strideConv2D'], concat, drop_out) 
            concat = self.drop_out(concat, drop_out) #drop out 3


            #create four branches: QF, QF uncertainty, DF, DF_uncertainty

            ## Quantitative Fluorescence Output Branch ##
            outQF = Conv2D(filters=64, kernel_size=self.params['kernelConv2D'], strides=self.params['strideConv2D'], padding='same', 
                            data_format="channels_last", kernel_initializer='he_normal')(concat)
            #outQF = BatchNormalization()(outQF)

            outQF = Conv2D(filters=32, kernel_size=self.params['kernelConv2D'], strides=self.params['strideConv2D'], padding='same', 
                           data_format="channels_last", kernel_initializer='he_normal')(outQF)

            #outQF = BatchNormalization()(outQF)

            outQF = Conv2D(filters=1, kernel_size=self.params['kernelConv2D'], strides=self.params['strideConv2D'], padding='same', 
                           data_format="channels_last", kernel_initializer='he_normal')(outQF)
            

            #QF uncertainty branch 

            outQF_uncertainty = Conv2D(filters=64, kernel_size=self.params['kernelConv2D'], strides=self.params['strideConv2D'], padding='same', 
                            data_format="channels_last", kernel_initializer='he_normal')(concat)
            #outQF = BatchNormalization()(outQF)

            outQF_uncertainty = Conv2D(filters=32, kernel_size=self.params['kernelConv2D'], strides=self.params['strideConv2D'], padding='same', 
                           data_format="channels_last", kernel_initializer='he_normal')(outQF_uncertainty)

            #outQF = BatchNormalization()(outQF)

            outQF_uncertainty = Conv2D(filters=1, kernel_size=self.params['kernelConv2D'], strides=self.params['strideConv2D'], padding='same', 
                           data_format="channels_last", kernel_initializer='he_normal', activation='sigmoid')(outQF_uncertainty)

            ## Depth Fluorescence Output Branch ##
            outDF = Conv2D(filters=64, kernel_size=self.params['kernelConv2D'], strides=self.params['strideConv2D'], padding='same', 
                           data_format="channels_last", kernel_initializer='he_normal')(concat)

            #outDF = BatchNormalization()(outDF)


            outDF = Conv2D(filters=32, kernel_size=self.params['kernelConv2D'], strides=self.params['strideConv2D'], padding='same', 
                          data_format="channels_last", kernel_initializer='he_normal')(outDF)
            #outDF = BatchNormalization()(outDF)

            outDF = Conv2D(filters=1, kernel_size=self.params['kernelConv2D'], strides=self.params['strideConv2D'], padding='same', 
                           data_format="channels_last", kernel_initializer='he_normal')(outDF)
            
            #outDF uncertainty branch 
            outDF_uncertainty = Conv2D(filters=64, kernel_size=self.params['kernelConv2D'], strides=self.params['strideConv2D'], padding='same', 
                            data_format="channels_last", kernel_initializer='he_normal')(concat)
            #outQF = BatchNormalization()(outQF)

            outDF_uncertainty = Conv2D(filters=32, kernel_size=self.params['kernelConv2D'], strides=self.params['strideConv2D'], padding='same', 
                           data_format="channels_last", kernel_initializer='he_normal')(outDF_uncertainty)

            #outQF = BatchNormalization()(outQF)

            outDF_uncertainty = Conv2D(filters=1, kernel_size=self.params['kernelConv2D'], strides=self.params['strideConv2D'], padding='same', 
                           activation='sigmoid', data_format="channels_last", kernel_initializer='he_normal')(outDF_uncertainty)
            
            QF_output = concatenate([outQF,outQF_uncertainty],axis=-1)
            DF_output = concatenate([outDF,outDF_uncertainty],axis=-1)



            self.modelD = Model(inputs=[inOP_beg,inFL_beg], outputs=[QF_output, DF_output])#,outFL])
            self.modelD.compile(loss=[self.laplacian_loss, self.laplacian_loss],
                          optimizer=getattr(keras.optimizers,self.params['optimizer'])(learning_rate=self.params['learningRate']),
                          metrics=[self.mae_on_first_channel, self.mae_on_first_channel])
            self.modelD.summary()
    def laplacian_loss(self, y_true, y_pred):
        mean_true = y_true[:, :, :, 0]
        mean_pred = y_pred[:, :, :, 0]
        log_var = y_pred[:, :, :, 1]
        #loss = tf.abs(tf.math.divide(tf.math.abs(mean_true - mean_pred), scale_pred + 1e-2) + tf.math.log(scale_pred + 1e-2))

        loss = tf.reduce_mean(tf.exp(-log_var) *tf.square( (mean_pred-mean_true) ) ) + tf.reduce_mean(log_var)

        return loss

    def mae_on_first_channel(self, y_true, y_pred):
        mean_true = y_true[:, :, :, 0]
        mean_pred = y_pred[:, :, :, 0]
        log_var = y_pred[:, :, :, 1]

        loss = tf.reduce_mean(tf.abs(mean_true - mean_pred))
        return loss 