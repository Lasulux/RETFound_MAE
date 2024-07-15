import sys
import argparse
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_addons as tfa
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger
import atexit
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os
import glob
import random
from PIL import Image
from scheduler import WarmUpCosine
import tfimm
from models_vit import *
import datetime

def parse_option():
    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--batch_size', default=1, type=int, help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=2, type=int)
    parser.add_argument('--input_size', default=224, type=int, help='images input size')
    parser.add_argument('--lr', type=float, default=1e-4, metavar='LR', help='learning rate (absolute lr)')
    parser.add_argument('--warmup_epoch_percentage', type=int, default=.1, metavar='N', help='epochs to warmup LR')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay (default: 1e-4)')
    parser.add_argument('--data_path', default='./data/single_image', type=str, help='dataset path')
    parser.add_argument('--nb_classes', default=5, type=int, help='number of the classification types')
    parser.add_argument('--output_dir', default='./output_dir/', help='path where to save, empty for no saving')
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument('--cutmix', type=float, default=0., help='cutmix alpha, cutmix enabled if > 0.')
    parser.add_argument('--finetune', default='', type=str, help='finetune from checkpoint')
    parser.add_argument('--task', default='', type=str, help='finetune from checkpoint')
    parser.add_argument('--global_pool', action='store_true')
    parser.set_defaults(global_pool=True)
    parser.add_argument('--cls_token', action='store_false', dest='global_pool', help='Use class token instead of global pool for classification')
    opt = parser.parse_args()
    return opt

def load_single_image(image_path, image_size):
    img = Image.open(image_path)
    img = img.resize((image_size, image_size))
    img = np.array(img)
    img = img / 255.0  # Normalize to [0, 1]
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

def main(opt):

    
    strategy = tf.distribute.MirroredStrategy()
    print('Number of devices: {}'.format(strategy.num_replicas_in_sync))

    image_path = "./data/IDRiD_data/train/anoDR/IDRiD_118.png"
    image = load_single_image(image_path, opt.input_size)

    with strategy.scope():
        if opt.global_pool:
            keras_model = tfimm.create_model("vit_large_patch16_224_mae", nb_classes=opt.nb_classes)
        else:
            keras_model = tfimm.create_model("vit_large_patch16_224", nb_classes=opt.nb_classes)

        if not opt.eval and opt.finetune:
            keras_model.load_weights(opt.finetune, skip_mismatch=True, by_name=True)
            print("Load pre-trained checkpoint from: %s" % opt.finetune)

        if opt.cutmix == 0.:
            data_augmentation = keras.Sequential([
                tf.keras.layers.RandomFlip("horizontal"),
                tf.keras.layers.RandomRotation(0.2),
                tf.keras.layers.RandomCrop(opt.input_size, opt.input_size),
            ])
            inputs = tf.keras.Input(shape=(opt.input_size, opt.input_size, 3))
            x = data_augmentation(inputs)
            outputs = keras_model(x)
            keras_model = keras.Model(inputs, outputs)

        optimizer = tfa.optimizers.AdamW(learning_rate=opt.lr, weight_decay=opt.weight_decay)
        loss_func = keras.losses.CategoricalCrossentropy(from_logits=True)

        if not opt.eval:
            keras_model.summary()
            keras_model.compile(optimizer=optimizer, loss=loss_func, metrics=["accuracy"])
        else:
            keras_model.compile(optimizer=optimizer, loss=loss_func, metrics=["categorical_accuracy"])

    if not opt.eval:
        keras_model.fit(image, np.array([[1, 0, 0, 0, 0]]), epochs=opt.epochs)  # Dummy label for single image training
    else:
        prediction = keras_model.predict(image)
        print(f"Prediction: {prediction}")

    atexit.register(strategy._extended._collective_ops._pool.close)

if __name__ == '__main__':
    opt = parse_option()

    SEED = 42
    tf.keras.utils.set_random_seed(SEED)

    main(opt)
