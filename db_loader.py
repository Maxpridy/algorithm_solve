import json
import numpy as np

class SamsungDB:
    def __init__(self, mode='train', db_path='./data/samsung_db.npy', test_db_path='./data/samsung_test_db.npy', val_rate=0.2):
        self.mode = mode
        if mode == 'train':
            db = np.load(db_path)
            db_dict = db.item()

            val_cutline = int(len(db_dict['data']) * (1 - val_rate))
            self.train_data = np.array(db_dict['data'][:val_cutline])
            self.train_hole = np.array(db_dict['hole'][:val_cutline])
            self.val_data = np.array(db_dict['data'][val_cutline:])
            self.val_hole = np.array(db_dict['hole'][val_cutline:])
        else:
            test_db = np.load(test_db_path)
            test_db_dict = test_db.item()

            self.test_data = np.array(test_db_dict['data'])
            self.test_hole = np.array(test_db_dict['hole'])

    def train_len(self):
        if not self.mode == 'train':
            raise Exception("Test mode")
        return len(self.train_data)

    def val_len(self):
        if not self.mode == 'train':
            raise Exception("Test mode")
        return len(self.val_data)

    def test_len(self):
        if not self.mode == 'test':
            raise Exception("Test mode")
        return len(self.test_data)

    def train_generator(self, batch_size=64):
        if not self.mode == 'train':
            raise Exception("Test mode")

        length = self.train_len()
        for n_idx in range(0, length, batch_size):
            data = self.train_data[n_idx:min(n_idx + batch_size, length)]
            hole = self.train_hole[n_idx:min(n_idx + batch_size, length)]
            x = data.copy()
            for i in range(len(data)):
                x[i, hole[i]] = -1
            hole_matrix = []
            for h in hole:
                hole_matrix.append([1 if i in h else 0 for i in range(16)])

            yield x, data, hole_matrix

    def val_generator(self, batch_size=64):
        if not self.mode == 'train':
            raise Exception("Test mode")

        length = self.val_len()
        for n_idx in range(0, length, batch_size):
            data = self.val_data[n_idx:min(n_idx + batch_size, length)]
            hole = self.val_hole[n_idx:min(n_idx + batch_size, length)]
            x = data.copy()
            for i in range(len(data)):
                x[i, hole[i]] = -1
            hole_matrix = []
            for h in hole:
                hole_matrix.append([1 if i in h else 0 for i in range(16)])

            yield x, data, hole_matrix

    def test_generator(self, batch_size=64):
        if not self.mode == 'test':
            raise Exception("Train mode")

        length = self.test_len()
        for n_idx in range(0, length, batch_size):
            data = self.test_data[n_idx:min(n_idx + batch_size, length)]
            hole = self.test_hole[n_idx:min(n_idx + batch_size, length)]
            x = data.copy()
            for i in range(len(data)):
                x[i, hole[i]] = -1
            hole_matrix = []
            for h in hole:
                hole_matrix.append([1 if i in h else 0 for i in range(16)])

            yield x, None, hole_matrix