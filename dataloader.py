from pathlib import Path, PosixPath
import random
import cv2
import numpy as np
import tensorflow as tf
from typing import Generator
from customs.augmentor import Augmentor

class DataLoader:
    def __init__(
        self,
        path: str|PosixPath,
        shape: tuple[int, int, int] | list[int, int, int],
        seed: int | None = None,
    ):

        self._set_shape(shape)

        # path section
        self._dataset_path = Path(path)
        if self._dataset_path.exists() == False:
            raise ValueError("dataset path not exist")

        self.update_seed(seed)
        self.reload_datasets()

    def update_seed(self, seed: int | None = None):
        self._seed = seed or random.randint(0, 999999)
        random.seed(self._seed)

    def reload_datasets(self, update_seed: bool = False):

        if update_seed:
            self.update_seed()

        self._dataset_files, self._dataset_files_count = self._preload_datasets()

    def load_dataset(
        self, load_as: str = "numpy", use_augmentor = False,
    ) -> Generator[tuple[np.ndarray, np.ndarray] | tuple[tf.Tensor, tf.Tensor], None, None]:

        if load_as != "numpy" and load_as != "tf":
            raise ValueError(
                f"load_as parameter must be 'numpy' or 'tf', received: '{load_as}'"
            )

        augmentor = Augmentor() if use_augmentor else None

        for src_path, lbl_path in self._dataset_files:
            src_img = cv2.imread(src_path, self._imread_flag)
            lbl_img = cv2.imread(lbl_path, cv2.IMREAD_GRAYSCALE)

            src_img = cv2.resize(src_img, self._shape[:-1], interpolation=cv2.INTER_CUBIC)
            lbl_img = cv2.resize(lbl_img, self._shape[:-1], interpolation=cv2.INTER_CUBIC)

            if self._imread_flag == cv2.IMREAD_GRAYSCALE:
                src_img = np.expand_dims(src_img, -1)

            lbl_img = np.expand_dims(lbl_img, -1)

            if use_augmentor:
                src_img, lbl_img = augmentor.augment(src_img, lbl_img)

            src_img = np.astype(src_img, np.float32)
            lbl_img = np.astype(lbl_img, np.float32)

            src_img /= 255.0
            lbl_img /= 255.0

            np.where(lbl_img >= 0.5, 1.0, 0.0)

            if load_as == "numpy":
                yield src_img, lbl_img

            if load_as == "tf":
                yield tf.convert_to_tensor(src_img, tf.float32), tf.convert_to_tensor(
                    lbl_img, tf.float32
                )

    def load_as_tf_dataset(self, batch_size=16, augment = False) -> tf.data.Dataset:
        def load_dataset():
            return self.load_dataset('tf', use_augmentor=augment)
        
        return tf.data.Dataset.from_generator(load_dataset, output_types=(tf.float32, tf.float32),
                                        output_shapes=(self._shape, (self._shape[0], self._shape[1], 1))) \
                             .apply(tf.data.experimental.assert_cardinality(self._dataset_files_count)) \
                             .batch(batch_size) \
                             .prefetch(tf.data.AUTOTUNE)

    def get_count(self):
        return self._dataset_files_count

    def _preload_datasets(self):
        files = []

        source_path = self._dataset_path / "source"
        label_path = self._dataset_path / "label"

        if not source_path.exists() or not label_path.exists():
            raise ValueError('source directory does not exist')

        if not source_path.is_dir() or not label_path.is_dir():
            raise ValueError('Label directory does not exist')

        for image in source_path.iterdir():
            label = label_path / image.name
            if not label.exists():
                continue

            files.append((str(image.absolute()), str(label.absolute())))
            
        random.shuffle(files)
        return files, len(files)

    def _set_shape(self, shape: tuple[int, int, 1 | 3] | list[int, int, 1 | 3]):

        self._shape = shape

        channel = shape[-1]
        if channel == 1:
            self._imread_flag = cv2.IMREAD_GRAYSCALE
        elif channel == 3:
            self._imread_flag = cv2.IMREAD_COLOR_RGB
