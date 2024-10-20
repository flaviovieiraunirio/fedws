# Fedws

This is the source code for the FedWS implementation, built on top of the https://github.com/c-gabri/Federated-Learning-PyTorch implementation.

FedWS added implementation of Lenet5 versions using Layer Normalization, Batch Normalization, Group Normalization and Weight Standardization layers.

model, optimizer and scheduler arguments:
  --model {cnn_cifar10,cnn_mnist,efficientnet,ghostnet,lenet5,lenet5_orig,mlp_mnist,mnasnet,mobilenet_v3}
                        model, place yours in models.py (default: lenet5)
  --model_args MODEL_ARGS
                        model arguments (default: ghost=True,norm=None)
                        norm {batch,layer,group,groupws}
