import os
import numpy as np
from sklearn.utils.class_weight import compute_class_weight


def get_class_weights(**kwargs):
    """Return class weights as dict of class index and weights"""
    class_weights = compute_class_weight(**kwargs)
    num_classes = len(kwargs['classes'])
    return dict(zip(range(num_classes), class_weights))


# fname parser : ""<root>/<class>/<case_key>_<frame_key>.jpg"
def path_parser(fpath):
    """Parse metadata from image file path."""
    class_ = fpath.split("/")[-2]
    fname = fpath.split("/")[-1]
    case_id = fname.split("_")[0]
    img_path = os.path.join(class_, fname)

    # compound class and fname address multilabel collisions
    img_id = "_".join([class_, fname])

    return {
        "image_id": img_id,
        "fname": fname,
        "case_id": case_id,
        "target": class_,
        "image_path": img_path,
    }


def inspect_trainable_layers(model, return_counts=False):
    """Inspect and return number of trainable layers."""
    num_trainable = len(model.trainable_weights)
    num_non_trainable = len(model.non_trainable_weights)

    if return_counts:
        return num_trainable, num_non_trainable
    print("Trainable layers: ", num_trainable)
    print("Non-trainable layers: ", num_non_trainable)
