
from keras.models import Model, load_model
from keras.layers import BatchNormalization, Input, concatenate, Conv2D, add, Conv3D, Reshape, SeparableConv2D, Dropout, MaxPool2D,MaxPool3D, UpSampling2D, ZeroPadding2D, Activation, ReLU, Lambda
from keras.preprocessing import image
import keras
import importlib
import numpy as np 
import tensorflow as tf 
class ModelInit():  

        def __init__(self, model_dir="Models_tensorflow"):
                self.model_dir = model_dir
        
        
        def Model_tf(self, model_name, class_name):
                """
                Dynamically load a model class and return the built model.

                Args:
                model_name (str): Python file name in models_tensorflow (without .py)
                class_name (str): Optional class name if not same as file name

                Returns:
                tf.keras.Model: Compiled Keras model
                """
                try:
                        module_path = f"{self.model_dir}.{model_name}"
                        module = importlib.import_module(module_path)

                        if class_name is None:
                                # Convert model_name to PascalCase if class not specified
                                class_name = ''.join(word.title() for word in model_name.split('_'))

                        model_class = getattr(module, class_name)
                        model_instance = model_class(self.params)
                        return model_instance.build_model()

                except AttributeError as e:
                        raise ImportError(f"Class '{class_name}' not found in module '{model_name}': {e}")