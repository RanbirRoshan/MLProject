import keras
from keras import regularizers
from keras.layers.core import Dense
from keras.constraints import UnitNorm,MinMaxNorm


class SingleLayerNN(keras.models.Sequential):
    def __init__(self, input_size, layer_count, node_count):
        super(SingleLayerNN, self).__init__(name="Final_Model")
        if layer_count == 0:
            self.add(Dense(units=1,
                       input_shape=(input_size,),
                           activation='softmax',
                           #kernel_initializer='RandomUniform',
                           # kernel_regularizer=regularizers.l1_l2(l1=0.01, l2=0.01),
                           use_bias=True,
                           #bias_initializer='RandomUniform',
                           # bias_regularizer=regularizers.l1(0.01),
                           # activity_regularizer = regularizers.l1_l2(0.0),
                           # kernel_constraint=UnitNorm(axis=0)
                           )
                     )
        else:
            self.add(Dense(units=node_count,
                           input_shape=(input_size,),
                           activation='relu',
                           #kernel_initializer='RandomUniform',
                           #kernel_regularizer=regularizers.l1_l2(l1=0.01, l2=0.01),
                           use_bias=True,
                           #bias_initializer='RandomUniform',
                           #bias_regularizer=regularizers.l1(0.01),
                           #activity_regularizer = regularizers.l1_l2(0.0),
                           #kernel_constraint=UnitNorm(axis=0)
                           )
                     )
            for i in range (1, layer_count):
                self.add(Dense(units=node_count,
                               activation='relu',
                               #kernel_initializer='RandomUniform',
                               #kernel_regularizer=regularizers.l1_l2(l1=0.01, l2=0.01),
                               use_bias=True,
                               #bias_initializer='RandomUniform',
                               #bias_regularizer=regularizers.l1(0.01),
                               #activity_regularizer = regularizers.l1_l2(0.0),
                               #kernel_constraint=UnitNorm(axis=0)
                               )
                         )
            self.add(Dense(units=1,
                           activation='softmax',
                           #kernel_initializer='RandomUniform',
                           #kernel_regularizer=regularizers.l1_l2(l1=0.01, l2=0.01),
                           use_bias=True,
                           #bias_initializer='RandomUniform',
                           #bias_regularizer=regularizers.l1(0.01),
                           #activity_regularizer = regularizers.l1_l2(0.0),
                           #kernel_constraint=UnitNorm(axis=0)
                           )
                     )
        self.compile(optimizer='SGD', loss='binary_crossentropy', metrics=['accuracy'])
