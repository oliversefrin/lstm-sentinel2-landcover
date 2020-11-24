"""
Neural Network test with new Ground Truth.

Information:
- 10 timesteps for training
- sequence length: 6
- pretrained model baseline_cnn
"""

# packages
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
# os.nice(20)
import sys
import time

from glob import glob
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, TimeDistributed, Bidirectional, ConvLSTM2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard

SM_FRAMEWORK=tf.keras
from segmentation_models.losses import CategoricalCELoss

sys.path.append('../..')
import src.helpers.image_functions as img_func
from src.predict.metrics import get_metrics_and_cm

model_name = sys.argv[0][:-3]
test_nr = len(glob(f'../../models/{model_name}*')) + 1

test_name = f'{model_name}_{test_nr:02d}'

# set variables
timesteps = 6
n_features = 13
image_size = 32
n_classes = 8
batch_size = 16

# class weights
pixel_per_class = [
    13247,
    581782,
    866153,
    90385,
    715608,
    11759,
    41714
]

class_weights = np.array([1/nr for nr in pixel_per_class] + [0])
class_weights = class_weights/np.sum(class_weights)*1000

# load pretrained cnn model
dummy_model = load_model(f'../../models/baseline_cnn_{test_nr:02d}.h5')

# define new model without last activation layer
base_model = Model(dummy_model.inputs, dummy_model.layers[-2].output)
base_model.set_weights(dummy_model.get_weights())

dates = [
    '20151231',
    '20160403',
    '20160522',
    '20160828',
    '20160929',
    '20161118',
    '20161206',
    '20170328',
    '20170424',
    '20170527'
]


# define LSTM model
inp = Input(shape=(timesteps, image_size, image_size, n_features))

cnn_time_distributed = TimeDistributed(
    base_model,
    input_shape=(timesteps, image_size, image_size, n_features)
)(inp)

out = Bidirectional(
    ConvLSTM2D(n_classes,
               kernel_size=(3, 3),
               padding='same',
               data_format='channels_last',
               activation='softmax'),
    merge_mode='ave'
)(cnn_time_distributed)

lstm_model = Model(inp, out)

loss = CategoricalCELoss(class_weights=class_weights)
adam = Adam(clipnorm=1.)
lstm_model.compile(optimizer=adam,
                   loss=loss,
                   metrics=['categorical_accuracy'])

# initialize data generators
path_to_data = '../../data/processed/training_data/'

train_sequence_generator = img_func.random_sequence_gen(
    path_to_data,
    dates,
    timesteps,
    n_classes=n_classes,
    split='train',
    batch_size=batch_size,
    rotation_range=45,
    horizontal_flip=True,
    vertical_flip=True
)

validation_sequence_generator = img_func.random_sequence_gen(
    path_to_data,
    dates,
    timesteps,
    n_classes=n_classes,
    split='valid',
    batch_size=batch_size
)

# define callbacks
earlystopper = EarlyStopping(
    monitor='val_categorical_accuracy',
    min_delta=0.001,
    patience=20,
    verbose=1,
    mode='auto',
    baseline=0.6,
    restore_best_weights=True
)

tensorboard = TensorBoard(
    log_dir=f'../../reports/tensorboard/{test_name}',
    write_graph=True,
    write_images=True,
    update_freq='epoch'
)


# This function keeps the learning rate at 0.001 for the first ten epochs
# and decreases it exponentially after that.
def scheduler(epoch):
    if epoch < 10:
        return 0.001
    else:
        return 0.001 * tf.math.exp(0.1 * (10 - epoch))


lr_schedule = tf.keras.callbacks.LearningRateScheduler(scheduler)

start_time = time.time()
# train model
lstm_model.fit(
    train_sequence_generator,
    steps_per_epoch=100,
    epochs=1000,
    verbose=1,
    callbacks=[earlystopper, tensorboard, lr_schedule],
    validation_data=validation_sequence_generator,
    validation_steps=5
)
end_time = time.time()

if not os.path.exists(f'../../reports/logs_and_plots/{test_name}/'):
    os.makedirs(f'../../reports/logs_and_plots/{test_name}/')

with open(f'../../reports/logs_and_plots/{test_name}/{test_name}_log.txt', 'w+') as file:
    file.write(f'Training completed in {end_time-start_time:0.1f} seconds.\n')


# save trained model
lstm_model.save(
    f'../../models/{test_name}.h5'
)

# get metrics on full test set
get_metrics_and_cm(
    model_name=test_name,
    split='test',
    model=lstm_model
)
