"""The two models: a 1D CNN baseline (decision 3.15) and EEGNet (decision 3.16).

Both take raw windows in microvolts, shaped (18 channels, 512 samples), and
output one seizure probability. Each model starts with a fixed normalisation
layer holding the training patients' channel mean and std, so a saved model
needs no separate preprocessing step, here or on a wearable.
"""
from functools import partial

import keras
from keras import layers, regularizers
from keras.constraints import max_norm


def _input(mean, std, n_ch, n_samples):
    inp = keras.Input((n_ch, n_samples), name="eeg_uV")
    # Mean and variance are set after the layer is built: passing them to the
    # constructor needs differently shaped arrays in different Keras versions.
    norm = layers.Normalization(axis=1, name="channel_zscore")
    x = norm(inp)
    count = norm.get_weights()[2]
    norm.set_weights([mean.reshape(n_ch), (std ** 2).reshape(n_ch), count])
    norm.finalize_state()
    return inp, x


def baseline_cnn(mean, std, n_ch=18, n_samples=512, l2=0.0, spatial_dropout=0.0):
    """Four convolution blocks (two conv layers each) and global average pooling, about 330k weights.

    l2 adds weight decay to every convolution and the output layer.
    spatial_dropout drops whole feature maps after each block during training.
    Both are off for "cnn" and on for "cnn-reg" (decision 3.23).
    """
    reg = regularizers.L2(l2) if l2 else None
    inp, x = _input(mean, std, n_ch, n_samples)
    x = layers.Permute((2, 1))(x)                         # time x channels for Conv1D
    for filters, pool in [(32, 4), (64, 4), (128, 4), (128, None)]:
        for _ in range(2):
            x = layers.Conv1D(filters, 5, padding="same", use_bias=False, kernel_regularizer=reg)(x)
            x = layers.BatchNormalization()(x)
            x = layers.ReLU()(x)
        if spatial_dropout:
            x = layers.SpatialDropout1D(spatial_dropout)(x)
        if pool:
            x = layers.MaxPooling1D(pool)(x)              # 512 -> 128 -> 32 -> 8 time steps
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(0.5)(x)
    out = layers.Dense(1, activation="sigmoid", kernel_regularizer=reg)(x)
    return keras.Model(inp, out, name="baseline_cnn")


def eegnet(mean, std, n_ch=18, n_samples=512, fs=128, F1=8, D=2, F2=16, dropout=0.25):
    """EEGNet-8,2 (Lawhern et al., 2018) with one sigmoid output, about 1.7k weights.

    Kernel length fs/2 (half a second), dropout 0.25 as the authors suggest for
    training across subjects.
    """
    inp, x = _input(mean, std, n_ch, n_samples)
    x = layers.Reshape((n_ch, n_samples, 1))(x)
    # temporal filters: learn frequency bands
    x = layers.Conv2D(F1, (1, fs // 2), padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    # depthwise spatial filters: learn which channels to combine for each band
    x = layers.DepthwiseConv2D((n_ch, 1), depth_multiplier=D, use_bias=False,
                               depthwise_constraint=max_norm(1.0))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("elu")(x)
    x = layers.AveragePooling2D((1, 4))(x)
    x = layers.Dropout(dropout)(x)
    # separable convolution: summarise each feature map over time, then mix them
    x = layers.SeparableConv2D(F2, (1, 16), padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("elu")(x)
    x = layers.AveragePooling2D((1, 8))(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Flatten()(x)
    out = layers.Dense(1, activation="sigmoid", kernel_constraint=max_norm(0.25))(x)
    return keras.Model(inp, out, name="eegnet")


MODELS = {"cnn": baseline_cnn,
          "cnn-reg": partial(baseline_cnn, l2=1e-4, spatial_dropout=0.2),
          "eegnet": eegnet}
