import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, ReLU, Conv2D, Input, MaxPooling2D, BatchNormalization, AveragePooling2D, Reshape, ZeroPadding2D, Dropout, Concatenate
from tensorflow.keras.models import Model

# CNN2
tf.keras.backend.clear_session()

Inp = Input(shape=(2,128))
x = Reshape((1,2,128))(Inp)

x = ZeroPadding2D((0,2))(x)
x = Conv2D(filters=256,kernel_size=(1,3),padding='valid',activation='relu',kernel_initializer='glorot_uniform',data_format='channels_first')(x)
x = Dropout(0.5)(x)

x = ZeroPadding2D((0,2))(x)
x = Conv2D(filters=80,kernel_size=(2,3),padding='valid',activation='relu',kernel_initializer='glorot_uniform',data_format='channels_first')(x)
x = Dropout(0.5)(x)

x = Flatten()(x)
x = Dense(256,activation='relu',kernel_initializer='he_normal')(x)
x = Dropout(0.5)(x)

Out = Dense(11,activation='softmax',kernel_initializer='he_normal')(x)

CNNModel = Model(Inp,Out)


# CNN4
tf.keras.backend.clear_session()

Inp = Input(shape=(2,128))
x = Reshape((1,2,128))(Inp)

x = Conv2D(filters=256,kernel_size=(1,3),padding='same',activation='relu',kernel_initializer='glorot_uniform',data_format='channels_first')(x)
x = Dropout(0.5)(x)

x = Conv2D(filters=256,kernel_size=(2,3),padding='same',activation='relu',kernel_initializer='glorot_uniform',data_format='channels_first')(x)
x = Dropout(0.5)(x)

x = Conv2D(filters=80,kernel_size=(1,3),padding='same',activation='relu',kernel_initializer='glorot_uniform',data_format='channels_first')(x)
x = Dropout(0.5)(x)

x = Conv2D(filters=80,kernel_size=(1,3),padding='same',activation='relu',kernel_initializer='glorot_uniform',data_format='channels_first')(x)
x = Dropout(0.5)(x)

x = Flatten()(x)
x = Dense(128,activation='relu',kernel_initializer='he_normal')(x)
x = Dropout(0.5)(x)

Out = Dense(11,activation='softmax',kernel_initializer='he_normal')(x)

CNNModel = Model(Inp,Out)


# DenseNet
tf.keras.backend.clear_session()

Inp = Input(shape=(2,128))
x = Reshape((1,2,128))(Inp)

x = Conv2D(filters=256,kernel_size=(1,3),padding='same',activation='relu',kernel_initializer='glorot_uniform',data_format='channels_first')(x)
node1 = Dropout(0.5)(x)

x = Conv2D(filters=256,kernel_size=(2,3),padding='same',activation='relu',kernel_initializer='glorot_uniform',data_format='channels_first')(node1)
node2 = Dropout(0.5)(x)
m1 = Concatenate(axis=1)([node1, node2])

x = Conv2D(filters=80,kernel_size=(1,3),padding='same',activation='relu',kernel_initializer='glorot_uniform',data_format='channels_first')(m1)
node3 = Dropout(0.5)(x)
m2 = Concatenate(axis=1)([node1, node2, node3])

x = Conv2D(filters=80,kernel_size=(1,3),padding='same',activation='relu',kernel_initializer='glorot_uniform',data_format='channels_first')(m2)
x = Dropout(0.5)(x)

x = Flatten()(x)
x = Dense(128,activation='relu',kernel_initializer='he_normal')(x)
x = Dropout(0.5)(x)

Out = Dense(11,activation='softmax',kernel_initializer='he_normal')(x)

CNNModel = Model(Inp,Out)
CNNModel.summary()
