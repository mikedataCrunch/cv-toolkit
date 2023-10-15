REPO_ROOT = "/home/mdorosan/2023/cv-toolkit"

import sys
sys.path.append(REPO_ROOT)

from tensorflow.keras import metrics, losses, optimizers
from tensorflow.keras.applications import (ResNet50, DenseNet121, VGG16, NASNetMobile, NASNetLarge, Xception, MobileNet, InceptionResNetV2, InceptionV3)

from tensorflow.keras.applications import (mobilenet_v2, xception, densenet, efficientnet, resnet, nasnet, vgg16, inception_v3, mobilenet)

import cvtoolkit.tensorflow.classification._paths as PATHS

# define metric dictionary
BINARY_METRICS = {
    "AUC_PR": metrics.AUC(curve='PR', name="AUC_PR"),
    "AUC_ROC": metrics.AUC(curve='ROC', name="AUC_ROC"),
}

COMPILE_PARAMS = {
    'loss' : losses.BinaryCrossentropy(),
    'metrics' : list(BINARY_METRICS.values()),
}
    
MULTICLASS_METRICS = {
    "Top-1-ACC":  metrics.CategoricalAccuracy(
        name="Top-1-ACC"),
    "Top-2-ACC": metrics.TopKCategoricalAccuracy(
        k=2, name="Top-2-ACC"),
    "Top-3-ACC": metrics.TopKCategoricalAccuracy(
        k=3, name="Top-3-ACC"),
}
    
MULTICLASS_COMPILE_PARAMS = {
    'loss' : losses.CategoricalCrossentropy(),
    'metrics' : list(MULTICLASS_METRICS.values()),
}

OPTIMIZER_DICT = {
    'SGD': optimizers.legacy.SGD(learning_rate=0.001),
    'Adam': optimizers.legacy.Adam(learning_rate=0.001),
}

BASE_MODELS = {
    "VGG16" : VGG16, # 138.4M, 4.2 ms per inference step (GPU)
    "ResNet50" : ResNet50, # 25.6M, 4.6 ms (GPU)
    "DenseNet121" : DenseNet121, # 8.1M, 5.4 ms (GPU)
    "NASNetMobile": NASNetMobile, # 5.3M, 6.7 ms (GPU)
    "MobileNet" : MobileNet, # 4.3 M, 3.4 ms (GPU)
    "Xception" : Xception, # 22.9M, 8.1 ms per inference step (GPU)
    "InceptionV3" : InceptionV3,   # 23.9 M, 6.9 ms per inference step (GPU)
}


BASE_PREPROCESSOR = {
    "VGG16" : vgg16.preprocess_input, 
    "ResNet50" : resnet.preprocess_input, 
    "DenseNet121" : densenet.preprocess_input, 
    "NASNetMobile": nasnet.preprocess_input, 
    "MobileNet" : mobilenet.preprocess_input, 
    "Xception" : xception.preprocess_input, 
    "InceptionV3" : inception_v3.preprocess_input,
}

IMAGENET_WEIGHTS = {
    key: PATHS.imagenet_weights[key] \
    for key in BASE_MODELS.keys()
}

