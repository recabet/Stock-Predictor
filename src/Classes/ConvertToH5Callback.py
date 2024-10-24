import os
import keras


class _ConvertToH5Callback(keras.callbacks.Callback):
    """Callback to convert the saved `.keras` model to `.h5` format."""

    def __init__(self, keras_filepath, h5_filepath):
        super().__init__()
        self.keras_filepath = keras_filepath
        self.h5_filepath = h5_filepath

    def on_epoch_end(self, epoch, logs=None):
        if os.path.exists(self.keras_filepath):
            model = keras.models.load_model(self.keras_filepath)
            model.save(self.h5_filepath)
            print(f"Model saved as {self.h5_filepath}")
