import tensorflow as tf
from tensorflow.keras import layers, models

class DenseNet:
    def __init__(self, input_shape, num_classes, weight_decay=1e-4):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.weight_decay = weight_decay
        self.model = self._build_model()

    def dense_block(self, x, blocks, name):
        for i in range(blocks):
            x = self.conv_block(x, 32, name=name + '_block' + str(i + 1))
        return x

    def transition_block(self, x, reduction, name):
        x = layers.BatchNormalization(axis=-1, epsilon=1.001e-5, name=name + '_bn')(x)
        x = layers.Activation('relu', name=name + '_relu')(x)
        x = layers.Conv2D(int(tf.keras.backend.int_shape(x)[-1] * reduction), 1, use_bias=False, name=name + '_conv')(x)
        x = layers.AveragePooling2D(2, strides=1, padding='same', name=name + '_pool')(x)  # Adjusted pooling to fit the input size
        return x

    def conv_block(self, x, growth_rate, name):
        x1 = layers.BatchNormalization(axis=-1, epsilon=1.001e-5, name=name + '_0_bn')(x)
        x1 = layers.Activation('relu', name=name + '_0_relu')(x1)
        x1 = layers.Conv2D(4 * growth_rate, 1, use_bias=False, name=name + '_1_conv')(x1)
        x1 = layers.BatchNormalization(axis=-1, epsilon=1.001e-5, name=name + '_1_bn')(x1)
        x1 = layers.Activation('relu', name=name + '_1_relu')(x1)
        x1 = layers.Conv2D(growth_rate, 3, padding='same', use_bias=False, name=name + '_2_conv')(x1)
        x = layers.Concatenate(axis=-1, name=name + '_concat')([x, x1])
        return x

    def _build_model(self):
        img_input = layers.Input(shape=self.input_shape)
        x = layers.Conv2D(64, 3, padding='same', strides=1, use_bias=False, name='conv1_conv')(img_input)
        x = layers.BatchNormalization(axis=-1, epsilon=1.001e-5, name='conv1_bn')(x)
        x = layers.Activation('relu', name='conv1_relu')(x)
        x = layers.MaxPooling2D(3, strides=1, padding='same', name='pool1')(x)  # Adjusted pooling to fit the input size

        x = self.dense_block(x, blocks=2, name='conv2')  # Reduced blocks to fit the input size
        x = self.transition_block(x, 0.5, name='pool2')
        x = self.dense_block(x, blocks=2, name='conv3')  # Reduced blocks to fit the input size
        x = self.transition_block(x, 0.5, name='pool3')
        x = self.dense_block(x, blocks=2, name='conv4')  # Reduced blocks to fit the input size
        x = self.transition_block(x, 0.5, name='pool4')
        x = self.dense_block(x, blocks=2, name='conv5')  # Reduced blocks to fit the input size

        x = layers.BatchNormalization(axis=-1, epsilon=1.001e-5, name='bn')(x)
        x = layers.Activation('relu', name='relu')(x)
        x = layers.GlobalAveragePooling2D(name='avg_pool')(x)
        x = layers.Dense(self.num_classes, activation='softmax', name='fc')(x)

        model = models.Model(img_input, x, name='densenet')
        return model

    def get_model(self):
        return self.model