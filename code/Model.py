import tensorflow as tf
import numpy as np
from .Generator import Generator
from .Discriminator import Discriminator
from .Loss import wasserstein_loss, gradient_penalty_loss, GRADIENT_PENALTY_WEIGHT

learning_rate = 0.00002

class Model():
    def __init__(self):
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, beta_1=0.5)
        self.discriminator = Discriminator()
        # compile ?

        self.colorizationModel = self.colorization_model()
        # compile ?
    
    def colorization_model(self):
        pass
