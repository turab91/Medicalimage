from keras.models import Model
from keras.layers import Input
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate
from keras.layers import BatchNormalization, Activation, Dropout


# Build U-Net model

def Unet(activation='relu', kernel_init='he_uniform', scale_=False):
    F1 = 16
    F2 = 32
    F3 = 64
    F4 = 128
    F5 = 256

    # Model
    inputs = Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))

    # Block1
    c1 = Conv2D(F1, (3, 3), kernel_initializer=kernel_init, padding='same')(inputs)
    c1 = BatchNormalization(scale=scale_)(c1)
    c1 = Activation(activation)(c1)
    c1 = Dropout(0.1)(c1)

    c1 = Conv2D(F1, (3, 3), kernel_initializer=kernel_init, padding='same')(c1)
    c1 = BatchNormalization(scale=scale_)(c1)
    c1 = Activation(activation)(c1)
    p1 = MaxPooling2D((2, 2))(c1)

    # Block2
    c2 = Conv2D(F2, (3, 3), kernel_initializer=kernel_init, padding='same')(p1)
    c2 = BatchNormalization(scale=scale_)(c2)
    c2 = Activation(activation)(c2)
    c2 = Dropout(0.2)(c2)

    c2 = Conv2D(F2, (3, 3), kernel_initializer=kernel_init, padding='same')(c2)
    c2 = BatchNormalization(scale=scale_)(c2)
    c2 = Activation(activation)(c2)
    p2 = MaxPooling2D((2, 2))(c2)

    # Block3
    c3 = Conv2D(F3, (3, 3), kernel_initializer=kernel_init, padding='same')(p2)
    c3 = BatchNormalization(scale=scale_)(c3)
    c3 = Activation(activation)(c3)
    c3 = Dropout(0.2)(c3)

    c3 = Conv2D(F3, (3, 3), kernel_initializer=kernel_init, padding='same')(c3)
    c3 = BatchNormalization(scale=scale_)(c3)
    c3 = Activation(activation)(c3)
    p3 = MaxPooling2D((2, 2))(c3)

    # Block4
    c4 = Conv2D(F4, (3, 3), kernel_initializer=kernel_init, padding='same')(p3)
    c4 = BatchNormalization(scale=scale_)(c4)
    c4 = Activation(activation)(c4)
    c4 = Dropout(0.2)(c4)

    c4 = Conv2D(F4, (3, 3), kernel_initializer=kernel_init, padding='same')(c4)
    c4 = BatchNormalization(scale=scale_)(c4)
    c4 = Activation(activation)(c4)
    p4 = MaxPooling2D(pool_size=(2, 2))(c4)

    # Block5
    c5 = Conv2D(F5, (3, 3), kernel_initializer=kernel_init, padding='same')(p4)
    c5 = BatchNormalization(scale=scale_)(c5)
    c5 = Activation(activation)(c5)
    c5 = Dropout(0.3)(c5)

    c5 = Conv2D(F5, (3, 3), kernel_initializer=kernel_init, padding='same')(c5)
    c5 = BatchNormalization(scale=scale_)(c5)
    c5 = Activation(activation)(c5)

    # Block6
    u6 = Conv2DTranspose(F4, (2, 2), strides=(2, 2), padding='same')(c5)
    u6 = concatenate([u6, c4])

    c6 = Conv2D(F4, (3, 3), kernel_initializer=kernel_init, padding='same')(u6)
    c6 = BatchNormalization(scale=scale_)(c6)
    c6 = Activation(activation)(c6)
    c6 = Dropout(0.2)(c6)

    c6 = Conv2D(F4, (3, 3), kernel_initializer=kernel_init, padding='same')(c6)
    c6 = BatchNormalization(scale=scale_)(c6)
    c6 = Activation(activation)(c6)

    # Block7
    u7 = Conv2DTranspose(F3, (2, 2), strides=(2, 2), padding='same')(c6)
    u7 = concatenate([u7, c3])

    c7 = Conv2D(F3, (3, 3), kernel_initializer=kernel_init, padding='same')(u7)
    c7 = BatchNormalization(scale=scale_)(c7)
    c7 = Activation(activation)(c7)
    c7 = Dropout(0.2)(c7)

    c7 = Conv2D(F3, (3, 3), kernel_initializer=kernel_init, padding='same')(c7)
    c7 = BatchNormalization(scale=scale_)(c7)
    c7 = Activation(activation)(c7)

    # Block8
    u8 = Conv2DTranspose(F2, (2, 2), strides=(2, 2), padding='same')(c7)
    u8 = concatenate([u8, c2])

    c8 = Conv2D(F2, (3, 3), kernel_initializer=kernel_init, padding='same')(u8)
    c8 = BatchNormalization(scale=scale_)(c8)
    c8 = Activation(activation)(c8)
    c8 = Dropout(0.2)(c8)

    c8 = Conv2D(F2, (3, 3), kernel_initializer=kernel_init, padding='same')(c8)
    c8 = BatchNormalization(scale=scale_)(c8)
    c8 = Activation(activation)(c8)

    # Block9
    u9 = Conv2DTranspose(F1, (2, 2), strides=(2, 2), padding='same')(c8)
    u9 = concatenate([u9, c1], axis=3)

    c9 = Conv2D(F1, (3, 3), kernel_initializer=kernel_init, padding='same')(u9)
    c9 = BatchNormalization(scale=scale_)(c9)
    c9 = Activation(activation)(c9)
    c9 = Dropout(0.1)(c9)

    c9 = Conv2D(F1, (3, 3), kernel_initializer=kernel_init, padding='same')(c9)
    c9 = BatchNormalization(scale=scale_)(c9)
    c9 = Activation(activation)(c9)

    outputs = Conv2D(1, (1, 1), activation='sigmoid')(c9)

    model = Model(inputs=inputs, outputs=outputs)

    return model
