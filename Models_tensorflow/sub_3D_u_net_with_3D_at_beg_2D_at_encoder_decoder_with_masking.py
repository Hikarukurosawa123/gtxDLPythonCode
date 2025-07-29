from tensorflow.keras.models import Model
from keras.layers import BatchNormalization, Input, concatenate, Conv2D, add, Conv3D, Reshape, SeparableConv2D, Dropout, MaxPool2D,MaxPool3D, UpSampling2D, ZeroPadding2D, Activation, ReLU, Lambda
from tensorflow import keras

import tensorflow as tf 

class UnetModel():
    def __init__(self, params):
        self.params = params
        self.modelD = None

    #helper function: attention gate 
    def attention_gate(self, g, s, num_filters):
        Wg = Conv2D(num_filters, 1, strides = (2,2), padding="valid")(g)
        Ws = Conv2D(num_filters, 1, padding="same")(s)

        out = Activation("relu")(Wg + Ws)
        out = Conv2D(1, 1, padding="same")(out)
        out = Activation("sigmoid")(out)
        out = UpSampling2D()(out)
        return out * g
    

    # Apply binary mask with 1 - rate probability of keeping a pixel
    def random_masking(self, x, rate):
        # Apply binary mask with 1 - rate probability of keeping a pixel
        def apply_mask(tensor):
                shape = tf.shape(tensor)
                # Random values between 0 and 1
                random_tensor = tf.random.uniform(shape, 0, 1)
                # Mask where values < (1 - rate) are kept, rest are zeroed
                binary_mask = tf.cast(random_tensor > rate, tensor.dtype)
                return tensor * binary_mask
        
        return Lambda(apply_mask)(x)
        
    def block_masking(self, x, block_size=17, masking_ratio=0.5):
        def mask_fn(tensor):
                input_shape = tf.shape(tensor)
                batch_size, h, w, c = input_shape[0], input_shape[1], input_shape[2], input_shape[3]

                # Compute number of full blocks
                num_blocks_h = h // block_size
                num_blocks_w = w // block_size
                total_blocks = num_blocks_h * num_blocks_w

                # Compute number of blocks to keep
                num_keep_blocks = tf.cast(
                tf.round((1.0 - masking_ratio) * tf.cast(total_blocks, tf.float32)),
                tf.int32
                )

                # Create a flat binary mask: 1 for kept blocks, 0 for masked
                keep_indices = tf.random.shuffle(tf.range(total_blocks))[:num_keep_blocks]
                flat_mask = tf.scatter_nd(
                indices=tf.expand_dims(keep_indices, 1),
                updates=tf.ones([num_keep_blocks], dtype=tf.float32),
                shape=[total_blocks]
                )

                # Reshape and expand mask to full spatial size
                mask_grid = tf.reshape(flat_mask, [num_blocks_h, num_blocks_w])
                mask_grid = tf.repeat(mask_grid, block_size, axis=0)
                mask_grid = tf.repeat(mask_grid, block_size, axis=1)
                mask_grid = mask_grid[:h, :w]  # Crop in case dimensions aren't exact multiples

                # Expand to (batch_size, height, width, channels)
                mask = tf.expand_dims(mask_grid, axis=0)    # (1, H, W)
                mask = tf.expand_dims(mask, axis=-1)        # (1, H, W, 1)
                mask = tf.tile(mask, [batch_size, 1, 1, c])  # (B, H, W, C)

                return tensor * tf.cast(mask, tensor.dtype)

        return Lambda(mask_fn, output_shape=lambda s: s)(x)

    

    def block_masking_per_channel(self, x, block_size=17, num_keep_blocks=6):
        def mask_fn(tensor):
                input_shape = tf.shape(tensor)
                batch_size, h, w, c = input_shape[0], input_shape[1], input_shape[2], input_shape[3]

                # Number of blocks along height and width
                num_blocks_h = h // block_size
                num_blocks_w = w // block_size
                total_blocks = num_blocks_h * num_blocks_w

                # Prepare a full mask of shape (batch, h, w, c)
                def generate_mask_per_sample(_):
                        # Per channel masking
                        masks = []
                        for _ in range(c):
                                # Randomly pick keep blocks
                                keep_idx = tf.random.shuffle(tf.range(total_blocks))[:num_keep_blocks]
                                flat_mask = tf.scatter_nd(
                                indices=tf.expand_dims(keep_idx, 1),
                                updates=tf.ones([num_keep_blocks], dtype=tf.float32),
                                shape=[total_blocks]
                                )
                                # Reshape to 2D block grid
                                mask_grid = tf.reshape(flat_mask, [num_blocks_h, num_blocks_w])
                                # Expand blocks
                                mask_grid = tf.repeat(mask_grid, block_size, axis=0)
                                mask_grid = tf.repeat(mask_grid, block_size, axis=1)
                                # Trim to original size
                                mask_grid = mask_grid[:h, :w]
                                masks.append(mask_grid)

                        # Stack masks for all channels: (H, W, C)
                        sample_mask = tf.stack(masks, axis=-1)
                        return sample_mask

                # Apply per-sample masking for the whole batch
                full_mask = tf.map_fn(generate_mask_per_sample, tf.range(batch_size), dtype=tf.float32)

                # Multiply input by the mask
                return tensor * tf.cast(full_mask, tensor.dtype)

        return Lambda(mask_fn)(x)

        

    def build_model(self):        
        """The deep learning architecture gets defined here"""
        
        # Input Optical Properties ##
        inOP_beg = Input(shape=(self.params['xX'],self.params['yY'],2))
        ## Input Multi-Dimensional Fluorescence ##
        inFL_beg = Input(shape=(self.params['xX'],self.params['yY'],self.params['nF'], 1))

        ## NOTE: Batch normalization can cause instability in the validation loss

        #3D CNN for all layers

        ## Optical Properties Branch ##
        inOP = Conv2D(filters=self.params['nFilters2D']//2, kernel_size=self.params['kernelConv2D'], strides=self.params['strideConv2D'], 
                padding='same', activation=self.params['activation'], data_format="channels_last")(inOP_beg)
        #inOP = Dropout(0.5)(inOP)
        #inOP = self.random_masking(inOP, 0.5)
        inOP = self.block_masking(inOP)

        inOP = Conv2D(filters=int(self.params['nFilters2D']/2), kernel_size=self.params['kernelConv2D'], strides=self.params['strideConv2D'], 
                padding='same', activation=self.params['activation'], data_format="channels_last")(inOP)
        #inOP = Dropout(0.5)(inOP)
        #inOP = self.random_masking(inOP, 0.5)
        inOP = self.block_masking(inOP)

        inOP = Conv2D(filters=int(self.params['nFilters2D']/2), kernel_size=self.params['kernelConv2D'], strides=self.params['strideConv2D'], 
                padding='same', activation=self.params['activation'], data_format="channels_last")(inOP)
        #inOP = Dropout(0.5)(inOP)  
        #inOP = self.random_masking(inOP, 0.5)
        inOP = self.block_masking(inOP)

        ## Fluorescence Input Branch ##
        #inFL = Reshape((inFL_beg.shape[1], inFL_beg.shape[2], 1,inFL_beg.shape[3]))(inFL_beg)
        input_shape = inFL_beg.shape

        inFL = Conv3D(filters=self.params['nFilters3D']//2, kernel_size=self.params['kernelConv3D'], strides=self.params['strideConv3D'], 
                padding='same', activation=self.params['activation'], input_shape=input_shape[1:], data_format="channels_last")(inFL_beg)
        #inFL = Dropout(0.5)(inFL)
        #inFL = self.random_masking(inFL, 0.5)
        inFL = self.block_masking(inFL)

        inFL = Conv3D(filters=int(self.params['nFilters3D']/2), kernel_size=self.params['kernelConv3D'], strides=self.params['strideConv3D'], 
                padding='same', activation=self.params['activation'], data_format="channels_last")(inFL)
        #inFL = Dropout(0.5)(inFL)
        #inFL = self.random_masking(inFL, 0.5)
        inFL = self.block_masking(inFL)

        inFL = Conv3D(filters=int(self.params['nFilters3D']/2), kernel_size=self.params['kernelConv3D'], strides=self.params['strideConv3D'], 
                padding='same', activation=self.params['activation'], data_format="channels_last")(inFL)
        #inFL = Dropout(0.5)(inFL)
        #inFL = self.random_masking(inFL, 0.5)
        inFL = self.block_masking(inFL)

        ## Concatenate Branch ##
        inFL = Reshape((inFL.shape[1], inFL.shape[2], inFL.shape[3] * inFL.shape[4]))(inFL)

        concat = concatenate([inOP,inFL],axis=-1)

        Max_Pool_1 = MaxPool2D()(concat)

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
        
        long_path_3 = ZeroPadding2D(padding = ((1,0), (1,0)))(concat)
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

        ## Quantitative Fluorescence Output Branch ##
        outQF = Conv2D(filters=64, kernel_size=self.params['kernelConv2D'], strides=self.params['strideConv2D'], padding='same', 
                activation=self.params['activation'], data_format="channels_last")(Conv_6)

        outQF = Conv2D(filters=32, kernel_size=self.params['kernelConv2D'], strides=self.params['strideConv2D'], padding='same', 
                activation=self.params['activation'], data_format="channels_last")(outQF) #outQF
        
        #outQF = BatchNormalization()(outQF)
        
        outQF = Conv2D(filters=1, kernel_size=self.params['kernelConv2D'], strides=self.params['strideConv2D'], padding='same', 
                        data_format="channels_last")(outQF)

        ## Depth Fluorescence Output Branch ##
        #first DF layer 
        outDF = Conv2D(filters=64, kernel_size=self.params['kernelConv2D'], strides=self.params['strideConv2D'], padding='same', 
                activation=self.params['activation'], data_format="channels_last")(Conv_6)

        outDF = Conv2D(filters=32, kernel_size=self.params['kernelConv2D'], strides=self.params['strideConv2D'], padding='same', 
                activation=self.params['activation'], data_format="channels_last")(outDF)

        #outDF = BatchNormalization()(outDF)

        
        outDF = Conv2D(filters=1, kernel_size=self.params['kernelConv2D'], strides=self.params['strideConv2D'], padding='same', 
                data_format="channels_last")(outDF)

        ## Defining and compiling the model ##
        self.modelD = Model(inputs=[inOP_beg,inFL_beg], outputs=[outQF, outDF])#,outFL])
        self.modelD.compile(loss=['mae', 'mae'],
                optimizer=getattr(keras.optimizers,self.params['optimizer'])(learning_rate=self.params['learningRate']),
                metrics=['mae', 'mae'])
        self.modelD.summary()
        return self.modelD 
