import os
import sys

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 0 = all logs, 1 = filter INFO, 2 = filter INFO+WARNING, 3 = only ERRORs

# project_root = '/home/victorh/projects/gtx'
# os.chdir(project_root)
# sys.path.insert(0, project_root)

import logging
logging.basicConfig(level=logging.INFO) 
logger = logging.getLogger(__name__)

import argparse
import numpy as np
from model.model import ModelInit
from tqdm import tqdm
import tensorflow as tf
from datetime import datetime

from preprocess import load_data

np.random.seed(1024)
tf.random.set_seed(1024)

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        print(gpus)
        tf.config.set_visible_devices(gpus[0], 'GPU')
        tf.config.experimental.set_memory_growth(gpus[0], True)
    except RuntimeError as e:
        print(e)

def get_arg_parser():
    parser = argparse.ArgumentParser(description="Hyperparameter configuration for fluorescence imaging model.")

    parser.add_argument('--sagemaker', type=bool, default=False, help='SageMaker mode')

    # General hyperparameters
    parser.add_argument('--activation', type=str, default='relu', help='Activation function')
    parser.add_argument('--optimizer', type=str, default='Adam', help='Optimizer name')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--nF', type=int, default=6, help='Number of fluroescent spatial frequencies (fluorescent images)')
    parser.add_argument('--learningRate', type=float, default=5e-4, help='Learning rate')
    parser.add_argument('--batch', type=int, default=32, help='Batch size')
    parser.add_argument('--xX', type=int, default=101, help='Image width')
    parser.add_argument('--yY', type=int, default=101, help='Image height')
    parser.add_argument('--decayRate', type=float, default=0.3, help='Learning rate decay factor')
    parser.add_argument('--patience', type=int, default=20, help='Early stopping patience')

    # Scaling parameters
    parser.add_argument('--scaleFL', type=float, default=10e4, help='Scaling factor for fluorescence')
    parser.add_argument('--scaleOP0', type=float, default=10, help='Scaling for absorption coefficient (μa)')
    parser.add_argument('--scaleOP1', type=float, default=1, help='Scaling for scattering coefficient (μs)')
    parser.add_argument('--scaleDF', type=float, default=1, help='Scaling for depth')
    parser.add_argument('--scaleQF', type=float, default=1, help='Scaling for fluorophore concentration')
    parser.add_argument('--scaleRE', type=float, default=1, help='Scaling for reflectance (optional)')

    # 3D Conv parameters
    parser.add_argument('--nFilters3D', type=int, default=128)
    parser.add_argument('--kernelConv3D', type=int, nargs=3, default=[3,3,3])
    parser.add_argument('--strideConv3D', type=int, nargs=3, default=[1,1,1])

    # 2D Conv parameters
    parser.add_argument('--nFilters2D', type=int, default=128)
    parser.add_argument('--kernelConv2D', type=int, nargs=2, default=[3,3])
    parser.add_argument('--strideConv2D', type=int, nargs=2, default=[1,1])

    # Data path
    parser.add_argument('--data_path', type=str, default='../data/20241118_data_splited.mat')
    parser.add_argument('--model_dir', type=str, default=f'../code_tf/ckpt/{datetime.now().strftime("%Y%m%d_%H%M%S")}/')
    return parser

class BatchLogger(tf.keras.callbacks.Callback):
    def __init__(self, log_interval=50, num_samples=None, batch_size=None):
        super().__init__()
        self.log_interval = log_interval
        self.total_batches = num_samples // batch_size

    def on_train_batch_end(self, batch, logs=None):
        if batch % self.log_interval == 0:
            loss = logs.get('loss')
            logging.info(f"[Batch {batch}/{self.total_batches}] Loss: {loss:.4f}")

    def on_epoch_end(self, epoch, logs=None):
        logging.info(
            f"Epoch {epoch}: "
            + ", ".join([f"{k} = {v:.4f}" for k, v in logs.items() if isinstance(v, float)])
        )

class CustomModelCheckpoint(tf.keras.callbacks.Callback):
    def __init__(self, filepath, monitor='val_loss', verbose=1):
        super().__init__()
        self.filepath = filepath
        self.monitor = monitor
        self.best = float('inf')
        self.verbose = verbose

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        current = logs.get(self.monitor)
        if current is not None and current < self.best:
            if self.verbose:
                print(f"Epoch {epoch}: {self.monitor} improved from {self.best:.5f} to {current:.5f}, saving model to {self.filepath}")
            self.best = current
            self.model.save(self.filepath)

def train(params):

    # Load data
    scale_params = {
        'fluorescence': params['scaleFL'],
        'mu_a': params['scaleOP0'],
        'mu_s': params['scaleOP1'],
        'depth': params['scaleDF'],
        'concentration_fluor': params['scaleQF'],
        'reflectance': params['scaleRE']
    }

    if params['sagemaker']:
        filepath = os.path.join('/opt/ml/input/data/training', '20241118_data_splited.mat')
        data = load_data(filepath, scale_params)
    else:
        data = load_data(params['data_path'], scale_params)

    train_data = data['train']
    train_fluorescence = train_data['fluorescence']
    # train_fluorescence = np.transpose(train_fluorescence, (0, 3, 1, 2))
    train_fluorescence = np.expand_dims(train_fluorescence, axis=-1)
    train_op = np.stack([train_data['mu_a'], train_data['mu_s']], axis=1)#.transpose(0, 2, 3, 1)
    train_depth = train_data['depth']
    train_concentration_fluor = train_data['concentration_fluor']
    train_reflectance = train_data['reflectance']

    print(train_fluorescence.shape)
    print(train_op.shape)
    print(train_depth.shape)
    print(train_concentration_fluor.shape)
    print(train_reflectance.shape)

    train_dataset = tf.data.Dataset.from_tensor_slices((
        (train_op, train_fluorescence),  # tuple of inputs
        {'outQF': train_concentration_fluor, 'outDF': train_depth}  # dict of outputs
    ))
    # train_dataset = tf.data.Dataset.from_tensor_slices((
    #     (train_op, train_fluorescence),  # tuple of inputs
    #     {'outQF': train_concentration_fluor, 'outDF': train_depth, 'outReflect': train_reflectance}  # dict of outputs
    # ))
    train_dataset = train_dataset.shuffle(buffer_size=1000, seed=1024, reshuffle_each_iteration=False).batch(params['batch'])

    val_data = data['val']
    val_fluorescence = val_data['fluorescence']
    # val_fluorescence = np.transpose(val_fluorescence, (0, 3, 1, 2))
    val_fluorescence = np.expand_dims(val_fluorescence, axis=-1)
    val_op = np.stack([val_data['mu_a'], val_data['mu_s']], axis=1).transpose(0, 2, 3, 1)
    val_depth = val_data['depth']
    val_concentration_fluor = val_data['concentration_fluor']
    val_reflectance = val_data['reflectance']

    val_dataset = tf.data.Dataset.from_tensor_slices((
        (val_op, val_fluorescence),  # tuple of inputs
        {'outQF': val_concentration_fluor, 'outDF': val_depth}  # dict of outputs
    ))
    # val_dataset = tf.data.Dataset.from_tensor_slices((
    #     (val_op, val_fluorescence),  # tuple of inputs
    #     {'outQF': val_concentration_fluor, 'outDF': val_depth, 'outReflect': val_reflectance}  # dict of outputs
    # ))
    val_dataset = val_dataset.batch(params['batch'])

    test_data = data['test']
    test_fluorescence = test_data['fluorescence']
    # test_fluorescence = np.transpose(test_fluorescence, (0, 3, 1, 2))
    test_fluorescence = np.expand_dims(test_fluorescence, axis=-1)
    test_op = np.stack([test_data['mu_a'], test_data['mu_s']], axis=1).transpose(0, 2, 3, 1)
    test_depth = test_data['depth']
    test_concentration_fluor = test_data['concentration_fluor']
    test_reflectance = test_data['reflectance']

    test_dataset = tf.data.Dataset.from_tensor_slices((
        (test_op, test_fluorescence),  # tuple of inputs
        {'outQF': test_concentration_fluor, 'outDF': test_depth}  # dict of outputs
    ))
    # test_dataset = tf.data.Dataset.from_tensor_slices((
    #     (test_op, test_fluorescence),  # tuple of inputs
    #     {'outQF': test_concentration_fluor, 'outDF': test_depth, 'outReflect': test_reflectance}  # dict of outputs
    # ))
    test_dataset = test_dataset.batch(params['batch'])

    # Initialize model
    model = ModelInit(params)
    model.build_model()

    # Initialize optimizer
    lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=params['decayRate'], patience=5, verbose=1, min_lr=5e-5)
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=5e-5, patience=params['patience'], verbose=1, mode='auto')
    batch_logger = BatchLogger(log_interval=20, num_samples= int(train_fluorescence.shape[0]), batch_size=params['batch'])
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if params['sagemaker']:
        model_dir = '/opt/ml/model'
    else:
        model_dir = params['model_dir']
        os.makedirs(model_dir, exist_ok=True)

    checkpoint = CustomModelCheckpoint(
        filepath=os.path.join(model_dir, f'model_ckpt_{timestamp}.keras'),
        monitor='val_loss',
        verbose=1
    )
    csv_logger = tf.keras.callbacks.CSVLogger(os.path.join(model_dir, f'loss_{timestamp}.log'), append=True)
    callbacks = [lr_scheduler, early_stopping, batch_logger, checkpoint, csv_logger]
    

    # Train model
    model.model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=params['epochs'],
        verbose=0,  
        callbacks=callbacks
    )
    

    model.load_model(os.path.join(model_dir, f'model_ckpt_{timestamp}.keras'))
    model.model.evaluate(
        test_dataset,
        verbose=1
    )

        
if __name__ == "__main__":
    parser = get_arg_parser()
    args = parser.parse_args()
    params = vars(args)

    np.random.seed(1024)
    tf.random.set_seed(1024)

    train(params)





