import cvtoolkit.tensorflow.classification._config as CONFIG
from tensorflow import keras

class CustomClassifier(keras.Model):
    
    def __init__(self, base, *args, **kwargs):
        super().__init__()
        
        # add customisations : sample below
        self.base = CONFIG.BASE_MODELS[base](
            weights=CONFIG.IMAGENET_WEIGHTS[base],
            **kwargs,
        )
        self.base.trainable = False
        
        self.flatten = keras.layers.Flatten()
        self.dropout_1 = keras.layers.Dropout(rate=0.1)
        
        self.hidden_dense = keras.layers.Dense(
            units=512, activation='relu',
        )
        self.dropout_2 = keras.layers.Dropout(rate=0.2)
        self.head = keras.layers.Dense(
            units=2, activation='sigmoid'
        )         
        
    def call(self, inputs, training=False):
        
        # update functional API base on custom model
        x = self.base(inputs)
        x = self.flatten(x)
        if training:
            x = self.dropout_1(x, training=training)
        x = self.hidden_dense(x)
        if training:
            x = self.dropout_2(x, training=training)
        return self.head(x)