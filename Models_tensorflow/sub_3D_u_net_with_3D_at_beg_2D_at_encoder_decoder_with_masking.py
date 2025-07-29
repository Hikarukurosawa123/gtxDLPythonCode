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
                batch_size, h, w, d, c = input_shape[0], input_shape[1], input_shape[2], input_shape[3], input_shape[4]

                # Compute number of blocks to cover full height and width
                num_blocks_h = tf.cast(tf.math.ceil(tf.cast(h, tf.float32) / block_size), tf.int32)
                num_blocks_w = tf.cast(tf.math.ceil(tf.cast(w, tf.float32) / block_size), tf.int32)
                total_blocks = num_blocks_h * num_blocks_w

                # Number of blocks to keep
                num_keep_blocks = tf.cast(
                tf.round((1.0 - masking_ratio) * tf.cast(total_blocks, tf.float32)),
                tf.int32
                )

                def single_channel_mask(_):
                        keep_indices = tf.random.shuffle(tf.range(total_blocks))[:num_keep_blocks]
                        flat_mask = tf.scatter_nd(
                                indices=tf.expand_dims(keep_indices, 1),
                                updates=tf.ones([num_keep_blocks], dtype=tf.float32),
                                shape=[total_blocks]
                        )
                        mask_2d = tf.reshape(flat_mask, [num_blocks_h, num_blocks_w])

                        # Repeat each block to full size
                        mask_2d = tf.repeat(mask_2d, repeats=block_size, axis=0)
                        mask_2d = tf.repeat(mask_2d, repeats=block_size, axis=1)

                        # Crop excess if needed (to match original shape)
                        mask_2d = mask_2d[:h, :w]

                        # Expand over depth and return
                        mask_3d = tf.expand_dims(mask_2d, axis=-1)  # (H, W, 1)
                        mask_3d = tf.tile(mask_3d, [1, 1, d])       # (H, W, D)
                        return mask_3d

                # Generate mask per channel
                channel_masks = tf.map_fn(
                single_channel_mask,
                elems=tf.range(c),
                fn_output_signature=tf.float32
                )  # (C, H, W, D)

                # Rearrange and tile over batch
                channel_masks = tf.transpose(channel_masks, [1, 2, 3, 0])  # (H, W, D, C)
                channel_masks = tf.expand_dims(channel_masks, axis=0)      # (1, H, W, D, C)
                mask = tf.tile(channel_masks, [batch_size, 1, 1, 1, 1])     # (B, H, W, D, C)

                return tensor * tf.cast(mask, tensor.dtype)

        return tf.keras.layers.Lambda(mask_fn, output_shape=lambda s: s)(x)
    def block_masking_per_channel(self, x, block_size=17, masking_ratio=0.5):
        """
        Applies random block-wise masking per channel on a 3D tensor (H, W, C),
        where the same block mask spans height/width but is different per channel.

        Args:
                x: Tensor of shape (B, H, W, C)
                block_size: Size of the square blocks (e.g., 17)
                masking_ratio: Fraction of blocks to mask out (e.g., 0.6 means keep 40%)

        Returns:
                Tensor of the same shape as input with masked values.
        """

        def mask_fn(tensor):
                h = tf.shape(tensor)[0]
                w = tf.shape(tensor)[1]
                c = tf.shape(tensor)[2]

                # Number of blocks along height and width, covering entire image without padding
                num_blocks_h = tf.math.floordiv(h + block_size - 1, block_size)
                num_blocks_w = tf.math.floordiv(w + block_size - 1, block_size)
                total_blocks = num_blocks_h * num_blocks_w

                num_keep_blocks = tf.cast(
                (1.0 - masking_ratio) * tf.cast(total_blocks, tf.float32),
                tf.int32
                )

                def mask_one_channel(_):
                        # Randomly pick blocks to keep
                        keep_idx = tf.random.shuffle(tf.range(total_blocks))[:num_keep_blocks]
                        flat_mask = tf.scatter_nd(
                                indices=tf.expand_dims(keep_idx, 1),
                                updates=tf.ones([num_keep_blocks], dtype=tf.float32),
                                shape=[total_blocks]
                        )
                        mask_grid = tf.reshape(flat_mask, [num_blocks_h, num_blocks_w])
                        # Repeat each block to full size
                        mask_grid = tf.repeat(mask_grid, block_size, axis=0)
                        mask_grid = tf.repeat(mask_grid, block_size, axis=1)
                        # Crop to original size (h, w)
                        return mask_grid[:h, :w]

                # Create masks for each channel independently
                masks = tf.map_fn(mask_one_channel, tf.range(c), dtype=tf.float32)
                mask_stack = tf.transpose(masks, perm=[1, 2, 0])  # (H, W, C)
                return tensor * tf.cast(mask_stack, tensor.dtype)

        return tf.keras.layers.Lambda(
                lambda batch_x: tf.map_fn(mask_fn, batch_x),
                output_shape=lambda input_shape: input_shape
        )(x)
    def block_masking_per_depth_channel(self, x, block_size=17, masking_ratio=0.5):
        """
        Applies independent 2D block-wise masking to each (H, W) slice of every (D, C) combination
        in a 5D tensor of shape (B, H, W, D, C).

        Args:
                x: Tensor of shape (B, H, W, D, C)
                block_size: Size of square masking block over H/W
                masking_ratio: Fraction of blocks to mask out per (H, W) slice

        Returns:
                Masked tensor of shape (B, H, W, D, C)
        """

        def sample_mask_fn(sample):  # sample shape: (H, W, D, C)
                h = tf.shape(sample)[0]
                w = tf.shape(sample)[1]
                d = tf.shape(sample)[2]
                c = tf.shape(sample)[3]

                num_blocks_h = tf.math.floordiv(h, block_size)
                num_blocks_w = tf.math.floordiv(w, block_size)
                total_blocks = num_blocks_h * num_blocks_w

                num_keep_blocks = tf.cast(
                tf.math.round((1.0 - masking_ratio) * tf.cast(total_blocks, tf.float32)),
                tf.int32
                )

                def mask_one(_):
                        # Generate per-slice mask
                        keep_idx = tf.random.shuffle(tf.range(total_blocks))[:num_keep_blocks]
                        flat_mask = tf.scatter_nd(
                                indices=tf.expand_dims(keep_idx, 1),
                                updates=tf.ones([num_keep_blocks], dtype=tf.float32),
                                shape=[total_blocks]
                        )
                        mask_2d = tf.reshape(flat_mask, [num_blocks_h, num_blocks_w])
                        mask_2d = tf.repeat(mask_2d, block_size, axis=0)
                        mask_2d = tf.repeat(mask_2d, block_size, axis=1)
                        return mask_2d[:h, :w]  # crop to original (H, W)

                # Create (D × C) masks of shape (H, W)
                masks = tf.map_fn(mask_one, tf.range(d * c), dtype=tf.float32)
                masks = tf.reshape(masks, [d, c, h, w])
                masks = tf.transpose(masks, [2, 3, 0, 1])  # (H, W, D, C)

                return sample * tf.cast(masks, sample.dtype)

        return tf.keras.layers.Lambda(
                lambda batch_x: tf.map_fn(sample_mask_fn, batch_x),
                output_shape=lambda input_shape: input_shape
        )(x)





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
        inOP = self.block_masking_per_channel(inOP)

        inOP = Conv2D(filters=int(self.params['nFilters2D']/2), kernel_size=self.params['kernelConv2D'], strides=self.params['strideConv2D'], 
                padding='same', activation=self.params['activation'], data_format="channels_last")(inOP)
        #inOP = Dropout(0.5)(inOP)
        #inOP = self.random_masking(inOP, 0.5)
        inOP = self.block_masking_per_channel(inOP)

        inOP = Conv2D(filters=int(self.params['nFilters2D']/2), kernel_size=self.params['kernelConv2D'], strides=self.params['strideConv2D'], 
                padding='same', activation=self.params['activation'], data_format="channels_last")(inOP)
        #inOP = Dropout(0.5)(inOP)  
        #inOP = self.random_masking(inOP, 0.5)
        inOP = self.block_masking_per_channel(inOP)

        ## Fluorescence Input Branch ##
        #inFL = Reshape((inFL_beg.shape[1], inFL_beg.shape[2], 1,inFL_beg.shape[3]))(inFL_beg)
        input_shape = inFL_beg.shape

        inFL = Conv3D(filters=self.params['nFilters3D']//2, kernel_size=self.params['kernelConv3D'], strides=self.params['strideConv3D'], 
                padding='same', activation=self.params['activation'], input_shape=input_shape[1:], data_format="channels_last")(inFL_beg)
        #inFL = Dropout(0.5)(inFL)
        #inFL = self.random_masking(inFL, 0.5)
        #inFL = self.block_masking(inFL)
        inFL = self.block_masking_per_depth_channel(inFL)

        inFL = Conv3D(filters=int(self.params['nFilters3D']/2), kernel_size=self.params['kernelConv3D'], strides=self.params['strideConv3D'], 
                padding='same', activation=self.params['activation'], data_format="channels_last")(inFL)
        #inFL = Dropout(0.5)(inFL)
        #inFL = self.random_masking(inFL, 0.5)
        #inFL = self.block_masking(inFL)
        inFL = self.block_masking_per_depth_channel(inFL)

        inFL = Conv3D(filters=int(self.params['nFilters3D']/2), kernel_size=self.params['kernelConv3D'], strides=self.params['strideConv3D'], 
                padding='same', activation=self.params['activation'], data_format="channels_last")(inFL)
        #inFL = Dropout(0.5)(inFL)
        #inFL = self.random_masking(inFL, 0.5)
        #inFL = self.block_masking(inFL)
        inFL = self.block_masking_per_depth_channel(inFL)

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
