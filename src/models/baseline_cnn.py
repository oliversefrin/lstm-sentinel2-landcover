"""
Test of a CNN without following ConvLSTM2D layer.

Output model can be used as pretrained model for CNN+LSTM tests.

Test parameters:
- backbone:     vgg19
- kernel_size:  3
- dates: only dates of 2016
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
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard

SM_FRAMEWORK = tf.keras
from segmentation_models import Unet
from segmentation_models.losses import CategoricalCELoss

sys.path.append('../..')
import src.helpers.image_functions as img_func
from src.predict.metrics import get_metrics_and_cm

# allow GPU memory growth
gpu = tf.config.experimental.list_physical_devices('GPU')[0]
tf.config.experimental.set_memory_growth(gpu, True)

model_name = sys.argv[0][:-3]
test_nr = len(glob(f'../../models/{model_name}*')) + 1

test_name = f'{model_name}_{test_nr:02d}'

# set variables
dates = [
    '20160403',
    '20160522',
    '20160828',
    '20160929',
    '20161118',
    '20161206',
]

n_features = 13
image_size = 32
n_classes = 8
batch_size = 64

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

class_weights = np.array([1/nr for nr in pixel_per_class]+[0])
class_weights = class_weights/np.sum(class_weights)*1000


# define model
model = Unet(
    backbone_name='vgg19',
    encoder_weights=None,
    input_shape=(None, None, n_features),
    classes=n_classes,
    activation='softmax'
)

loss = CategoricalCELoss(class_weights=class_weights)
adam = Adam(clipnorm=1.)

model.compile(
    optimizer=adam,
    loss=loss,
    metrics=['categorical_accuracy']
)


# initialize data generators
path_to_data = '../../data/processed/training_data/'

train_sequence_generator = img_func.simple_image_generator(
    path_to_data,
    dates,
    n_classes=n_classes,
    split='train',
    batch_size=batch_size,
    rotation_range=45,
    horizontal_flip=True,
    vertical_flip=True
)


validation_sequence_generator = img_func.simple_image_generator(
    path_to_data,
    dates,
    n_classes=n_classes,
    split='valid',
    batch_size=batch_size,
)


# define callbacks
earlystopper = EarlyStopping(
    monitor='val_categorical_accuracy',
    min_delta=0.01,
    patience=20,
    verbose=1,
    mode='auto',
    baseline=0.5,
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
model.fit(
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
model.save(f'../../models/{test_name}.h5')

# get metrics on full test set
get_metrics_and_cm(
    model_name=test_name,
    split='test',
    model=model
)
