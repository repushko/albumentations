from __future__ import division
import math
import numpy as np
import warnings

from albumentations.core.utils import DataProcessor

__all__ = [
    "convert_gaze_vector_to_albumentations",
    "convert_gaze_vector_from_albumentations",
    "gaze_vector_rotation_matrix",
    "GazeVectorProcessor",
]

gaze_vectors_formats = {"xyz"}


class GazeVectorProcessor(DataProcessor):
    @property
    def default_data_name(self):
        return "gaze_vector"

    def ensure_data_valid(self, data):
        if self.params.label_fields:
            if not all(i in data.keys() for i in self.params.label_fields):
                raise ValueError(
                    "Your 'label_fields' are not valid - them must have same "
                    "names as params in "
                    "'gaze_vector' dict"
                )

    def filter(self, data, rows, cols):
        return data

    def check(self, data, rows, cols):
        return data

    def ensure_transforms_valid(self, transforms):
        from albumentations.imgaug.transforms import DualIAATransform

        if self.params.format is not None and self.params.format != "xyz":
            for transform in transforms:
                if isinstance(transform, DualIAATransform):
                    warnings.warn(
                        "{} transformation supports only 'xyz' keypoints "
                        "augmentation. You have '{}' keypoints format. Scale "
                        "and angle WILL NOT BE transformed.".format(transform.__class__.__name__, self.params.format)
                    )
                    break

    def convert_from_albumentations(self, data, rows, cols):
        return convert_gaze_vector_from_albumentations(
            data,
            self.params.format,
            rows,
            cols
        )

    def convert_to_albumentations(self, data, rows, cols):
        return convert_gaze_vector_to_albumentations(
            data,
            self.params.format,
            rows,
            cols
        )


def convert_gaze_vector_to_albumentations(
        gaze_vector, source_format, rows, cols, check_validity=False
):
    if source_format not in gaze_vectors_formats:
        raise ValueError("Unknown target_format {}. Supported formats are: {}".format(source_format, gaze_vectors_formats))

    if source_format == "xyz":
        (x, y, z) = gaze_vector[:3]

    gaze_vector = (x, y, z)
    if check_validity:
        raise Warning("We don't validate gaze_vector in this version of library.")
    return gaze_vector


def convert_gaze_vector_from_albumentations(
        gaze_vector, target_format, rows, cols, check_validity=False
):
    # type (tuple, str, int, int, bool, bool) -> tuple
    if target_format not in gaze_vectors_formats:
        raise ValueError("Unknown target_format {}. Supported formats are: {}".format(target_format, gaze_vectors_formats))
    if check_validity:
        raise Warning("We don't validate gaze_vector in this version of library.")

    (x, y, z) = gaze_vector[:3]

    if target_format == "xyz":
        gaze_vector = (x, y, z)

    return gaze_vector


def gaze_vector_rotation_matrix(x_angle, y_angle):
    x_angle_rad = math.radians(x_angle)
    cos_x = math.cos(x_angle_rad)
    sin_x = math.sin(x_angle_rad)

    y_angle_rad = math.radians(y_angle)
    cos_y = math.cos(y_angle_rad)
    sin_y = math.sin(y_angle_rad)

    x_rotation_matrix = np.array([[1, 0, 0], [0, cos_x, -1*sin_x], [0, sin_x, cos_x]])
    y_rotation_matrix = np.array([[cos_y, 0, sin_y], [0, 1, 0], [-sin_y, 0, cos_y]])

    return x_rotation_matrix.dot(y_rotation_matrix)