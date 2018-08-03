import keras
import numpy as np
from model import baseline
from generate_train import run_time_batch_generate


def train():
    generator = run_time_batch_generate.get_generator()
    model = baseline.get_model()

    print(model.summary())
    for idx, ()
        model.fit(train_x, train_y, epochs=10, batch_size=128, validation_split=0.2)


def run_test():
    model = baseline.get_model()
    print(model.summary())


if __name__ == '__main__':
    run_test()