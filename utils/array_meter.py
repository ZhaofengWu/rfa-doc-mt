from fairseq.logging.meters import Meter
import numpy as np


class ArrayMeter(Meter):
    def __init__(self):
        self.reset()

    def reset(self):
        self.array = np.zeros(0)

    def update(self, val):
        self.array = np.concatenate((self.array, val), axis=0)

    def state_dict(self):
        return {"array": self.array}

    def load_state_dict(self, state_dict):
        self.array = state_dict["array"]


# Inject our custom meter into fairseq so that it has access to it when loading a trained model
from fairseq.logging import meters

setattr(meters, "ArrayMeter", ArrayMeter)
