import keras
import keras.losses as K_loss
import keras.backend as K
import tensorflow as tf


def base_loss():
    losses = {
        "label_0": "sparse_categorical_crossentropy",
        "label_1": "sparse_categorical_crossentropy",
        "label_2": "mean_squared_error",
        "label_3": "mean_squared_error",
        "label_4": "mean_squared_error",
        "label_5": "mean_squared_error",
        "label_6": "mean_squared_error",
        "label_7": "sparse_categorical_crossentropy",
        "label_8": "sparse_categorical_crossentropy",
        "label_9": "sparse_categorical_crossentropy",
        "label_10": "sparse_categorical_crossentropy",
        "label_11": "sparse_categorical_crossentropy",
        "label_12": "sparse_categorical_crossentropy",
        "label_13": "sparse_categorical_crossentropy",
        "label_14": "sparse_categorical_crossentropy",
        "label_15": "sparse_categorical_crossentropy",
    }
    return losses


def base_loss2():
    reg_idx = [2, 3, 4, 5, 6]  # regression에 해당하는 라벨
    cls_idx = list(set(range(16)) - set(reg_idx))  # 나머지 라벨 (classification에 해당)

    def _base_loss(y_true, y_pred):
        print(y_true.shape)
        print(y_pred.shape)
        reg_true, reg_pred = y_true[:, reg_idx], y_pred[:, reg_idx]
        cls_true, cls_pred = y_true[:, cls_idx], y_pred[:, cls_idx]

        reg_losses = K_loss.mean_squared_error(reg_true, reg_pred)
        cls_losses = K_loss.categorical_crossentropy(cls_true, cls_pred)

        return reg_losses + cls_losses

    return _base_loss

