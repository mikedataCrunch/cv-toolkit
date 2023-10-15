import os

singhealth_dataroot = "/home/dr_jiang/Desktop/Capsule project/CE_Data"
kvasir_dataroot = "/home/michaeldorsan/Desktop/datasets/kvasir-capsule-dataset"

REPO_ROOT = "/home/mdorosan/2023/pillcam-bp-and-detection"

# imagenet_weights_root = "/home/michaeldorsan/Desktop/repositories/keras-model-downloads/models"
# radimagenet_weights_root = "/home/michaeldorsan/Desktop/repositories/pillcam-bp-and-detection/data/radimagenet-weights"

imagenet_weights_root = "/home/mdorosan/2023/keras-model-downloads/models"
radimagenet_weights_root = os.path.join(REPO_ROOT, "data/radimagenet-weights")


imagenet_weights = {
    "VGG16" : os.path.join(
        imagenet_weights_root, 
        "vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5"), 
    "ResNet50" : os.path.join(
        imagenet_weights_root,
        "resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5"), 
    "DenseNet121" : os.path.join(
        imagenet_weights_root,
        "densenet121_weights_tf_dim_ordering_tf_kernels_notop.h5"), 
    "NASNetMobile": os.path.join(
        imagenet_weights_root, 
        "nasnet_mobile_no_top.h5"), 
    "MobileNet" : os.path.join(
        imagenet_weights_root, 
        "mobilenet_1_0_224_tf_no_top.h5"), 
    "Xception" : os.path.join(
        imagenet_weights_root, 
        "xception_weights_tf_dim_ordering_tf_kernels_notop.h5"), 
    "InceptionV3" : os.path.join(
        imagenet_weights_root, 
        "inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5"),
    "EfficientNetB0" : os.path.join(
        imagenet_weights_root, 
        "efficientnetb0_notop.h5"),
    "MobileNetV2" : os.path.join(
        imagenet_weights_root, 
        "mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.0_224_no_top.h5"),
}

radimagenet_weights = {
    "ResNet50" : os.path.join(
        radimagenet_weights_root,
        "RadImageNet-ResNet50_notop.h5"), 
    "DenseNet121" : os.path.join(
        radimagenet_weights_root,
        "RadImageNet-DenseNet121_notop.h5"),
    "InceptionV3" : os.path.join(
        radimagenet_weights_root, 
        "RadImageNet-InceptionV3_notop.h5"),
    "InceptionResNetV2" : os.path.join(
        radimagenet_weights_root,
        "RadImageNet-IRV2_notop.h5",)
}


