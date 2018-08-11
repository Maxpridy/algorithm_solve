import keras
import numpy as np
from lib import baseline_keras
from db_loader import SamsungDB

def train():
    db = SamsungDB()
    model = baseline_keras.get_model()
    batch_size = 64
    steps = 10
    print(model.summary())
    for step in range(steps):
        for idx, (x, y) in enumerate(db.train_generator(batch_size)):
            model.train_on_batch(x, y)
        for idx, (x, y) in enumerate(db.val_generator(batch_size)):
            model.test_on_batch(x, y)


def run_test():
    model = baseline_keras.get_model()
    print(model.summary())


if __name__ == '__main__':
    train()