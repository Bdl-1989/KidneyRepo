from common import *


###############################################################################

IMAGE_RGB_MEAN = [0.485, 0.456, 0.406]
IMAGE_RGB_STD  = [0.229, 0.224, 0.225]

PRETRAIN_FILE = '/root/share/data/pretrain_model/resnet34-333f7ec4.pth'
CONVERSION=[
 'block0.0.weight',	(64, 3, 7, 7),	 'conv1.weight',	(64, 3, 7, 7),
 'block0.1.weight',	(64,),	 'bn1.weight',	(64,),
 'block0.1.bias',	(64,),	 'bn1.bias',	(64,),
 'block0.1.running_mean',	(64,),	 'bn1.running_mean',	(64,),
 'block0.1.running_var',	(64,),	 'bn1.running_var',	(64,),
 'block1.0.conv_bn1.conv.weight',	(64, 64, 3, 3),	 'layer1.0.conv1.weight',	(64, 64, 3, 3),
 'block1.0.conv_bn1.bn.weight',	(64,),	 'layer1.0.bn1.weight',	(64,),
 'block1.0.conv_bn1.bn.bias',	(64,),	 'layer1.0.bn1.bias',	(64,),
 'block1.0.conv_bn1.bn.running_mean',	(64,),	 'layer1.0.bn1.running_mean',	(64,),
 'block1.0.conv_bn1.bn.running_var',	(64,),	 'layer1.0.bn1.running_var',	(64,),
 'block1.0.conv_bn2.conv.weight',	(64, 64, 3, 3),	 'layer1.0.conv2.weight',	(64, 64, 3, 3),
 'block1.0.conv_bn2.bn.weight',	(64,),	 'layer1.0.bn2.weight',	(64,),
 'block1.0.conv_bn2.bn.bias',	(64,),	 'layer1.0.bn2.bias',	(64,),
 'block1.0.conv_bn2.bn.running_mean',	(64,),	 'layer1.0.bn2.running_mean',	(64,),
 'block1.0.conv_bn2.bn.running_var',	(64,),	 'layer1.0.bn2.running_var',	(64,),
 'block1.1.conv_bn1.conv.weight',	(64, 64, 3, 3),	 'layer1.1.conv1.weight',	(64, 64, 3, 3),
 'block1.1.conv_bn1.bn.weight',	(64,),	 'layer1.1.bn1.weight',	(64,),
 'block1.1.conv_bn1.bn.bias',	(64,),	 'layer1.1.bn1.bias',	(64,),
 'block1.1.conv_bn1.bn.running_mean',	(64,),	 'layer1.1.bn1.running_mean',	(64,),
 'block1.1.conv_bn1.bn.running_var',	(64,),	 'layer1.1.bn1.running_var',	(64,),
 'block1.1.conv_bn2.conv.weight',	(64, 64, 3, 3),	 'layer1.1.conv2.weight',	(64, 64, 3, 3),
 'block1.1.conv_bn2.bn.weight',	(64,),	 'layer1.1.bn2.weight',	(64,),
 'block1.1.conv_bn2.bn.bias',	(64,),	 'layer1.1.bn2.bias',	(64,),
 'block1.1.conv_bn2.bn.running_mean',	(64,),	 'layer1.1.bn2.running_mean',	(64,),
 'block1.1.conv_bn2.bn.running_var',	(64,),	 'layer1.1.bn2.running_var',	(64,),
 'block1.2.conv_bn1.conv.weight',	(64, 64, 3, 3),	 'layer1.2.conv1.weight',	(64, 64, 3, 3),
 'block1.2.conv_bn1.bn.weight',	(64,),	 'layer1.2.bn1.weight',	(64,),
 'block1.2.conv_bn1.bn.bias',	(64,),	 'layer1.2.bn1.bias',	(64,),
 'block1.2.conv_bn1.bn.running_mean',	(64,),	 'layer1.2.bn1.running_mean',	(64,),
 'block1.2.conv_bn1.bn.running_var',	(64,),	 'layer1.2.bn1.running_var',	(64,),
 'block1.2.conv_bn2.conv.weight',	(64, 64, 3, 3),	 'layer1.2.conv2.weight',	(64, 64, 3, 3),
 'block1.2.conv_bn2.bn.weight',	(64,),	 'layer1.2.bn2.weight',	(64,),
 'block1.2.conv_bn2.bn.bias',	(64,),	 'layer1.2.bn2.bias',	(64,),
 'block1.2.conv_bn2.bn.running_mean',	(64,),	 'layer1.2.bn2.running_mean',	(64,),
 'block1.2.conv_bn2.bn.running_var',	(64,),	 'layer1.2.bn2.running_var',	(64,),
 'block2.0.conv_bn1.conv.weight',	(128, 64, 3, 3),	 'layer2.0.conv1.weight',	(128, 64, 3, 3),
 'block2.0.conv_bn1.bn.weight',	(128,),	 'layer2.0.bn1.weight',	(128,),
 'block2.0.conv_bn1.bn.bias',	(128,),	 'layer2.0.bn1.bias',	(128,),
 'block2.0.conv_bn1.bn.running_mean',	(128,),	 'layer2.0.bn1.running_mean',	(128,),
 'block2.0.conv_bn1.bn.running_var',	(128,),	 'layer2.0.bn1.running_var',	(128,),
 'block2.0.conv_bn2.conv.weight',	(128, 128, 3, 3),	 'layer2.0.conv2.weight',	(128, 128, 3, 3),
 'block2.0.conv_bn2.bn.weight',	(128,),	 'layer2.0.bn2.weight',	(128,),
 'block2.0.conv_bn2.bn.bias',	(128,),	 'layer2.0.bn2.bias',	(128,),
 'block2.0.conv_bn2.bn.running_mean',	(128,),	 'layer2.0.bn2.running_mean',	(128,),
 'block2.0.conv_bn2.bn.running_var',	(128,),	 'layer2.0.bn2.running_var',	(128,),
 'block2.0.shortcut.conv.weight',	(128, 64, 1, 1),	 'layer2.0.downsample.0.weight',	(128, 64, 1, 1),
 'block2.0.shortcut.bn.weight',	(128,),	 'layer2.0.downsample.1.weight',	(128,),
 'block2.0.shortcut.bn.bias',	(128,),	 'layer2.0.downsample.1.bias',	(128,),
 'block2.0.shortcut.bn.running_mean',	(128,),	 'layer2.0.downsample.1.running_mean',	(128,),
 'block2.0.shortcut.bn.running_var',	(128,),	 'layer2.0.downsample.1.running_var',	(128,),
 'block2.1.conv_bn1.conv.weight',	(128, 128, 3, 3),	 'layer2.1.conv1.weight',	(128, 128, 3, 3),
 'block2.1.conv_bn1.bn.weight',	(128,),	 'layer2.1.bn1.weight',	(128,),
 'block2.1.conv_bn1.bn.bias',	(128,),	 'layer2.1.bn1.bias',	(128,),
 'block2.1.conv_bn1.bn.running_mean',	(128,),	 'layer2.1.bn1.running_mean',	(128,),
 'block2.1.conv_bn1.bn.running_var',	(128,),	 'layer2.1.bn1.running_var',	(128,),
 'block2.1.conv_bn2.conv.weight',	(128, 128, 3, 3),	 'layer2.1.conv2.weight',	(128, 128, 3, 3),
 'block2.1.conv_bn2.bn.weight',	(128,),	 'layer2.1.bn2.weight',	(128,),
 'block2.1.conv_bn2.bn.bias',	(128,),	 'layer2.1.bn2.bias',	(128,),
 'block2.1.conv_bn2.bn.running_mean',	(128,),	 'layer2.1.bn2.running_mean',	(128,),
 'block2.1.conv_bn2.bn.running_var',	(128,),	 'layer2.1.bn2.running_var',	(128,),
 'block2.2.conv_bn1.conv.weight',	(128, 128, 3, 3),	 'layer2.2.conv1.weight',	(128, 128, 3, 3),
 'block2.2.conv_bn1.bn.weight',	(128,),	 'layer2.2.bn1.weight',	(128,),
 'block2.2.conv_bn1.bn.bias',	(128,),	 'layer2.2.bn1.bias',	(128,),
 'block2.2.conv_bn1.bn.running_mean',	(128,),	 'layer2.2.bn1.running_mean',	(128,),
 'block2.2.conv_bn1.bn.running_var',	(128,),	 'layer2.2.bn1.running_var',	(128,),
 'block2.2.conv_bn2.conv.weight',	(128, 128, 3, 3),	 'layer2.2.conv2.weight',	(128, 128, 3, 3),
 'block2.2.conv_bn2.bn.weight',	(128,),	 'layer2.2.bn2.weight',	(128,),
 'block2.2.conv_bn2.bn.bias',	(128,),	 'layer2.2.bn2.bias',	(128,),
 'block2.2.conv_bn2.bn.running_mean',	(128,),	 'layer2.2.bn2.running_mean',	(128,),
 'block2.2.conv_bn2.bn.running_var',	(128,),	 'layer2.2.bn2.running_var',	(128,),
 'block2.3.conv_bn1.conv.weight',	(128, 128, 3, 3),	 'layer2.3.conv1.weight',	(128, 128, 3, 3),
 'block2.3.conv_bn1.bn.weight',	(128,),	 'layer2.3.bn1.weight',	(128,),
 'block2.3.conv_bn1.bn.bias',	(128,),	 'layer2.3.bn1.bias',	(128,),
 'block2.3.conv_bn1.bn.running_mean',	(128,),	 'layer2.3.bn1.running_mean',	(128,),
 'block2.3.conv_bn1.bn.running_var',	(128,),	 'layer2.3.bn1.running_var',	(128,),
 'block2.3.conv_bn2.conv.weight',	(128, 128, 3, 3),	 'layer2.3.conv2.weight',	(128, 128, 3, 3),
 'block2.3.conv_bn2.bn.weight',	(128,),	 'layer2.3.bn2.weight',	(128,),
 'block2.3.conv_bn2.bn.bias',	(128,),	 'layer2.3.bn2.bias',	(128,),
 'block2.3.conv_bn2.bn.running_mean',	(128,),	 'layer2.3.bn2.running_mean',	(128,),
 'block2.3.conv_bn2.bn.running_var',	(128,),	 'layer2.3.bn2.running_var',	(128,),
 'block3.0.conv_bn1.conv.weight',	(256, 128, 3, 3),	 'layer3.0.conv1.weight',	(256, 128, 3, 3),
 'block3.0.conv_bn1.bn.weight',	(256,),	 'layer3.0.bn1.weight',	(256,),
 'block3.0.conv_bn1.bn.bias',	(256,),	 'layer3.0.bn1.bias',	(256,),
 'block3.0.conv_bn1.bn.running_mean',	(256,),	 'layer3.0.bn1.running_mean',	(256,),
 'block3.0.conv_bn1.bn.running_var',	(256,),	 'layer3.0.bn1.running_var',	(256,),
 'block3.0.conv_bn2.conv.weight',	(256, 256, 3, 3),	 'layer3.0.conv2.weight',	(256, 256, 3, 3),
 'block3.0.conv_bn2.bn.weight',	(256,),	 'layer3.0.bn2.weight',	(256,),
 'block3.0.conv_bn2.bn.bias',	(256,),	 'layer3.0.bn2.bias',	(256,),
 'block3.0.conv_bn2.bn.running_mean',	(256,),	 'layer3.0.bn2.running_mean',	(256,),
 'block3.0.conv_bn2.bn.running_var',	(256,),	 'layer3.0.bn2.running_var',	(256,),
 'block3.0.shortcut.conv.weight',	(256, 128, 1, 1),	 'layer3.0.downsample.0.weight',	(256, 128, 1, 1),
 'block3.0.shortcut.bn.weight',	(256,),	 'layer3.0.downsample.1.weight',	(256,),
 'block3.0.shortcut.bn.bias',	(256,),	 'layer3.0.downsample.1.bias',	(256,),
 'block3.0.shortcut.bn.running_mean',	(256,),	 'layer3.0.downsample.1.running_mean',	(256,),
 'block3.0.shortcut.bn.running_var',	(256,),	 'layer3.0.downsample.1.running_var',	(256,),
 'block3.1.conv_bn1.conv.weight',	(256, 256, 3, 3),	 'layer3.1.conv1.weight',	(256, 256, 3, 3),
 'block3.1.conv_bn1.bn.weight',	(256,),	 'layer3.1.bn1.weight',	(256,),
 'block3.1.conv_bn1.bn.bias',	(256,),	 'layer3.1.bn1.bias',	(256,),
 'block3.1.conv_bn1.bn.running_mean',	(256,),	 'layer3.1.bn1.running_mean',	(256,),
 'block3.1.conv_bn1.bn.running_var',	(256,),	 'layer3.1.bn1.running_var',	(256,),
 'block3.1.conv_bn2.conv.weight',	(256, 256, 3, 3),	 'layer3.1.conv2.weight',	(256, 256, 3, 3),
 'block3.1.conv_bn2.bn.weight',	(256,),	 'layer3.1.bn2.weight',	(256,),
 'block3.1.conv_bn2.bn.bias',	(256,),	 'layer3.1.bn2.bias',	(256,),
 'block3.1.conv_bn2.bn.running_mean',	(256,),	 'layer3.1.bn2.running_mean',	(256,),
 'block3.1.conv_bn2.bn.running_var',	(256,),	 'layer3.1.bn2.running_var',	(256,),
 'block3.2.conv_bn1.conv.weight',	(256, 256, 3, 3),	 'layer3.2.conv1.weight',	(256, 256, 3, 3),
 'block3.2.conv_bn1.bn.weight',	(256,),	 'layer3.2.bn1.weight',	(256,),
 'block3.2.conv_bn1.bn.bias',	(256,),	 'layer3.2.bn1.bias',	(256,),
 'block3.2.conv_bn1.bn.running_mean',	(256,),	 'layer3.2.bn1.running_mean',	(256,),
 'block3.2.conv_bn1.bn.running_var',	(256,),	 'layer3.2.bn1.running_var',	(256,),
 'block3.2.conv_bn2.conv.weight',	(256, 256, 3, 3),	 'layer3.2.conv2.weight',	(256, 256, 3, 3),
 'block3.2.conv_bn2.bn.weight',	(256,),	 'layer3.2.bn2.weight',	(256,),
 'block3.2.conv_bn2.bn.bias',	(256,),	 'layer3.2.bn2.bias',	(256,),
 'block3.2.conv_bn2.bn.running_mean',	(256,),	 'layer3.2.bn2.running_mean',	(256,),
 'block3.2.conv_bn2.bn.running_var',	(256,),	 'layer3.2.bn2.running_var',	(256,),
 'block3.3.conv_bn1.conv.weight',	(256, 256, 3, 3),	 'layer3.3.conv1.weight',	(256, 256, 3, 3),
 'block3.3.conv_bn1.bn.weight',	(256,),	 'layer3.3.bn1.weight',	(256,),
 'block3.3.conv_bn1.bn.bias',	(256,),	 'layer3.3.bn1.bias',	(256,),
 'block3.3.conv_bn1.bn.running_mean',	(256,),	 'layer3.3.bn1.running_mean',	(256,),
 'block3.3.conv_bn1.bn.running_var',	(256,),	 'layer3.3.bn1.running_var',	(256,),
 'block3.3.conv_bn2.conv.weight',	(256, 256, 3, 3),	 'layer3.3.conv2.weight',	(256, 256, 3, 3),
 'block3.3.conv_bn2.bn.weight',	(256,),	 'layer3.3.bn2.weight',	(256,),
 'block3.3.conv_bn2.bn.bias',	(256,),	 'layer3.3.bn2.bias',	(256,),
 'block3.3.conv_bn2.bn.running_mean',	(256,),	 'layer3.3.bn2.running_mean',	(256,),
 'block3.3.conv_bn2.bn.running_var',	(256,),	 'layer3.3.bn2.running_var',	(256,),
 'block3.4.conv_bn1.conv.weight',	(256, 256, 3, 3),	 'layer3.4.conv1.weight',	(256, 256, 3, 3),
 'block3.4.conv_bn1.bn.weight',	(256,),	 'layer3.4.bn1.weight',	(256,),
 'block3.4.conv_bn1.bn.bias',	(256,),	 'layer3.4.bn1.bias',	(256,),
 'block3.4.conv_bn1.bn.running_mean',	(256,),	 'layer3.4.bn1.running_mean',	(256,),
 'block3.4.conv_bn1.bn.running_var',	(256,),	 'layer3.4.bn1.running_var',	(256,),
 'block3.4.conv_bn2.conv.weight',	(256, 256, 3, 3),	 'layer3.4.conv2.weight',	(256, 256, 3, 3),
 'block3.4.conv_bn2.bn.weight',	(256,),	 'layer3.4.bn2.weight',	(256,),
 'block3.4.conv_bn2.bn.bias',	(256,),	 'layer3.4.bn2.bias',	(256,),
 'block3.4.conv_bn2.bn.running_mean',	(256,),	 'layer3.4.bn2.running_mean',	(256,),
 'block3.4.conv_bn2.bn.running_var',	(256,),	 'layer3.4.bn2.running_var',	(256,),
 'block3.5.conv_bn1.conv.weight',	(256, 256, 3, 3),	 'layer3.5.conv1.weight',	(256, 256, 3, 3),
 'block3.5.conv_bn1.bn.weight',	(256,),	 'layer3.5.bn1.weight',	(256,),
 'block3.5.conv_bn1.bn.bias',	(256,),	 'layer3.5.bn1.bias',	(256,),
 'block3.5.conv_bn1.bn.running_mean',	(256,),	 'layer3.5.bn1.running_mean',	(256,),
 'block3.5.conv_bn1.bn.running_var',	(256,),	 'layer3.5.bn1.running_var',	(256,),
 'block3.5.conv_bn2.conv.weight',	(256, 256, 3, 3),	 'layer3.5.conv2.weight',	(256, 256, 3, 3),
 'block3.5.conv_bn2.bn.weight',	(256,),	 'layer3.5.bn2.weight',	(256,),
 'block3.5.conv_bn2.bn.bias',	(256,),	 'layer3.5.bn2.bias',	(256,),
 'block3.5.conv_bn2.bn.running_mean',	(256,),	 'layer3.5.bn2.running_mean',	(256,),
 'block3.5.conv_bn2.bn.running_var',	(256,),	 'layer3.5.bn2.running_var',	(256,),
 'block4.0.conv_bn1.conv.weight',	(512, 256, 3, 3),	 'layer4.0.conv1.weight',	(512, 256, 3, 3),
 'block4.0.conv_bn1.bn.weight',	(512,),	 'layer4.0.bn1.weight',	(512,),
 'block4.0.conv_bn1.bn.bias',	(512,),	 'layer4.0.bn1.bias',	(512,),
 'block4.0.conv_bn1.bn.running_mean',	(512,),	 'layer4.0.bn1.running_mean',	(512,),
 'block4.0.conv_bn1.bn.running_var',	(512,),	 'layer4.0.bn1.running_var',	(512,),
 'block4.0.conv_bn2.conv.weight',	(512, 512, 3, 3),	 'layer4.0.conv2.weight',	(512, 512, 3, 3),
 'block4.0.conv_bn2.bn.weight',	(512,),	 'layer4.0.bn2.weight',	(512,),
 'block4.0.conv_bn2.bn.bias',	(512,),	 'layer4.0.bn2.bias',	(512,),
 'block4.0.conv_bn2.bn.running_mean',	(512,),	 'layer4.0.bn2.running_mean',	(512,),
 'block4.0.conv_bn2.bn.running_var',	(512,),	 'layer4.0.bn2.running_var',	(512,),
 'block4.0.shortcut.conv.weight',	(512, 256, 1, 1),	 'layer4.0.downsample.0.weight',	(512, 256, 1, 1),
 'block4.0.shortcut.bn.weight',	(512,),	 'layer4.0.downsample.1.weight',	(512,),
 'block4.0.shortcut.bn.bias',	(512,),	 'layer4.0.downsample.1.bias',	(512,),
 'block4.0.shortcut.bn.running_mean',	(512,),	 'layer4.0.downsample.1.running_mean',	(512,),
 'block4.0.shortcut.bn.running_var',	(512,),	 'layer4.0.downsample.1.running_var',	(512,),
 'block4.1.conv_bn1.conv.weight',	(512, 512, 3, 3),	 'layer4.1.conv1.weight',	(512, 512, 3, 3),
 'block4.1.conv_bn1.bn.weight',	(512,),	 'layer4.1.bn1.weight',	(512,),
 'block4.1.conv_bn1.bn.bias',	(512,),	 'layer4.1.bn1.bias',	(512,),
 'block4.1.conv_bn1.bn.running_mean',	(512,),	 'layer4.1.bn1.running_mean',	(512,),
 'block4.1.conv_bn1.bn.running_var',	(512,),	 'layer4.1.bn1.running_var',	(512,),
 'block4.1.conv_bn2.conv.weight',	(512, 512, 3, 3),	 'layer4.1.conv2.weight',	(512, 512, 3, 3),
 'block4.1.conv_bn2.bn.weight',	(512,),	 'layer4.1.bn2.weight',	(512,),
 'block4.1.conv_bn2.bn.bias',	(512,),	 'layer4.1.bn2.bias',	(512,),
 'block4.1.conv_bn2.bn.running_mean',	(512,),	 'layer4.1.bn2.running_mean',	(512,),
 'block4.1.conv_bn2.bn.running_var',	(512,),	 'layer4.1.bn2.running_var',	(512,),
 'block4.2.conv_bn1.conv.weight',	(512, 512, 3, 3),	 'layer4.2.conv1.weight',	(512, 512, 3, 3),
 'block4.2.conv_bn1.bn.weight',	(512,),	 'layer4.2.bn1.weight',	(512,),
 'block4.2.conv_bn1.bn.bias',	(512,),	 'layer4.2.bn1.bias',	(512,),
 'block4.2.conv_bn1.bn.running_mean',	(512,),	 'layer4.2.bn1.running_mean',	(512,),
 'block4.2.conv_bn1.bn.running_var',	(512,),	 'layer4.2.bn1.running_var',	(512,),
 'block4.2.conv_bn2.conv.weight',	(512, 512, 3, 3),	 'layer4.2.conv2.weight',	(512, 512, 3, 3),
 'block4.2.conv_bn2.bn.weight',	(512,),	 'layer4.2.bn2.weight',	(512,),
 'block4.2.conv_bn2.bn.bias',	(512,),	 'layer4.2.bn2.bias',	(512,),
 'block4.2.conv_bn2.bn.running_mean',	(512,),	 'layer4.2.bn2.running_mean',	(512,),
 'block4.2.conv_bn2.bn.running_var',	(512,),	 'layer4.2.bn2.running_var',	(512,),
 'logit.weight',	(1000, 512),	 'fc.weight',	(1000, 512),
 'logit.bias',	(1000,),	 'fc.bias',	(1000,),

]

def load_pretrain(net, skip=[], pretrain_file=PRETRAIN_FILE, conversion=CONVERSION, is_print=True):

    #raise NotImplementedError
    print('\tload pretrain_file: %s'%pretrain_file)

    #pretrain_state_dict = torch.load(pretrain_file)
    pretrain_state_dict = torch.load(pretrain_file, map_location=lambda storage, loc: storage)
    state_dict = net.state_dict()

    i = 0
    conversion = np.array(conversion).reshape(-1,4)
    for key,_,pretrain_key,_ in conversion:
        if any(s in key for s in
            ['.num_batches_tracked',]+skip):
            continue

        #print('\t\t',key)
        if is_print:
            print('\t\t','%-48s  %-24s  <---  %-32s  %-24s'%(
                key, str(state_dict[key].shape),
                pretrain_key, str(pretrain_state_dict[pretrain_key].shape),
            ))
        i = i+1

        state_dict[key] = pretrain_state_dict[pretrain_key]

    #---
    net.load_state_dict(state_dict)
    print('')
    print('len(pretrain_state_dict.keys()) = %d'%len(pretrain_state_dict.keys()))
    print('len(state_dict.keys())          = %d'%len(state_dict.keys()))
    print('loaded    = %d'%i)
    print('')



#---
class RGB(nn.Module):
    def __init__(self,):
        super(RGB, self).__init__()
        self.register_buffer('mean', torch.zeros(1,3,1,1))
        self.register_buffer('std', torch.ones(1,3,1,1))
        self.mean.data = torch.FloatTensor(IMAGE_RGB_MEAN).view(self.mean.shape)
        self.std.data = torch.FloatTensor(IMAGE_RGB_STD).view(self.std.shape)

    def forward(self, x):
        x = (x-self.mean)/self.std
        return x

###############################################################################
class ConvBn2d(nn.Module):

    def __init__(self, in_channel, out_channel, kernel_size=3, padding=1, stride=1):
        super(ConvBn2d, self).__init__()
        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, padding=padding, stride=stride, bias=False)
        self.bn   = nn.BatchNorm2d(out_channel, eps=1e-5)

    def forward(self,x):
        x = self.conv(x)
        x = self.bn(x)
        return x




# bottleneck type C
class Basic(nn.Module):
    def __init__(self, in_channel, out_channel, channel, kernel_size=3, stride=1, is_shortcut=False):
        super(Basic, self).__init__()
        self.is_shortcut = is_shortcut

        self.conv_bn1 = ConvBn2d(in_channel,    channel, kernel_size=kernel_size, padding=kernel_size//2, stride=stride)
        self.conv_bn2 = ConvBn2d(   channel,out_channel, kernel_size=kernel_size, padding=kernel_size//2, stride=1)

        if is_shortcut:
            self.shortcut = ConvBn2d(in_channel, out_channel, kernel_size=1, padding=0, stride=stride)


    def forward(self, x):
        z = F.relu(self.conv_bn1(x),inplace=True)
        z = self.conv_bn2(z)

        if self.is_shortcut:
            x = self.shortcut(x)

        z += x
        z = F.relu(z,inplace=True)
        return z




class ResNet34(nn.Module):

    def __init__(self, num_class=1000 ):
        super(ResNet34, self).__init__()

        self.block0  = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, padding=3, stride=2, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        self.block1  = nn.Sequential(
             Basic( 64, 64, 64, kernel_size=3, stride=1, is_shortcut=False,),
          * [Basic( 64, 64, 64, kernel_size=3, stride=1, is_shortcut=False,) for i in range(1,3)],
        )
        self.block2  = nn.Sequential(
             Basic( 64,128,128, kernel_size=3, stride=2, is_shortcut=True, ),
          * [Basic(128,128,128, kernel_size=3, stride=1, is_shortcut=False,) for i in range(1,4)],
        )
        self.block3  = nn.Sequential(
             Basic(128,256,256, kernel_size=3, stride=2, is_shortcut=True, ),
          * [Basic(256,256,256, kernel_size=3, stride=1, is_shortcut=False,) for i in range(1,6)],
        )
        self.block4 = nn.Sequential(
             Basic(256,512,512, kernel_size=3, stride=2, is_shortcut=True, ),
          * [Basic(512,512,512, kernel_size=3, stride=1, is_shortcut=False,) for i in range(1,3)],
        )
        self.logit = nn.Linear(512,num_class)
        self.rgb = RGB()

    def forward(self, x):
        batch_size = len(x)
        x = self.rgb(x)

        x = self.block0(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = F.adaptive_avg_pool2d(x,1).reshape(batch_size,-1)
        logit = self.logit(x)
        return logit

#########################################################################
def run_check_pretrain():

    net = ResNet34()

    #---
    if 0:
        print(net)
        print('')

        print('*** print key *** ')
        state_dict = net.state_dict()
        keys = list(state_dict.keys())
        #keys = sorted(keys)
        for k in keys:
            if any(s in k for s in [
                'num_batches_tracked'
                # '.kernel',
                # '.gamma',
                # '.beta',
                # '.running_mean',
                # '.running_var',
            ]):
                continue

            p = state_dict[k].data.cpu().numpy()
            print(' \'%s\',\t%s,'%(k,tuple(p.shape)))
        print('')
        exit(0)

    load_pretrain(net, is_print=True)


    #---
    if 1:
        net = net.cuda().eval()

        synset_file = '/root/share/data/imagenet/dummy/synset_words'
        synset = read_list_from_file(synset_file)
        synset = [s[10:].split(',')[0] for s in synset]

        image_dir ='/root/share/data/imagenet/dummy/256x256'
        fs = [ 'great_white_shark','screwdriver','ostrich','blad_eagle','english_foxhound','goldfish',]
        #fs = [f[len(image_dir)+1:-4] for f in glob.glob(image_dir+'/*.jpg')]

        fs = sorted(fs)
        for f in fs:
            image_file = image_dir +'/%s.jpg'%f
            image = cv2.imread(image_file, cv2.IMREAD_COLOR)
            #image = cv2.resize(image,dsize=(380,380))
            #image = image[16:16+224,16:16+224]

            image = image[:,:,::-1]
            image = image.astype(np.float32)/255
            #image = (image -IMAGE_RGB_MEAN)/IMAGE_RGB_STD
            input = image.transpose(2,0,1)
            input = np.ascontiguousarray(input)
            input = torch.from_numpy(input).float().cuda().unsqueeze(0)

            logit = net(input)
            proability = F.softmax(logit,-1)

            probability = proability.data.cpu().numpy().reshape(-1)
            argsort = np.argsort(-probability)

            print(f, image.shape)
            print(probability[:5])
            for t in range(5):
                print(t, '%24s'%synset[argsort[t]][:24], '%3d'%argsort[t], probability[argsort[t]])
            print('')

    print('\nsucess!')




# main #################################################################
if __name__ == '__main__':
    print( '%s: calling main function ... ' % os.path.basename(__file__))
    run_check_pretrain()

##############################################################################3


'''

blad_eagle (256, 256, 3)
[1.0544026e-09 1.0391688e-09 3.3691344e-10 2.5403493e-10 5.3890115e-09]
0               bald eagle  22 0.9996056
1                albatross 146 0.00015695988
2                     kite  21 0.00012857879
3                  vulture  23 7.818386e-05
4              white stork 127 8.167942e-06

english_foxhound (256, 256, 3)
[2.2772535e-06 1.6688922e-06 1.3474948e-06 1.4425866e-06 8.0607208e-07]
0         English foxhound 167 0.47306937
1             Walker hound 166 0.3022376
2                   beagle 162 0.1783484
3                 bluetick 164 0.008347741
4               bloodhound 163 0.006411056

goldfish (256, 256, 3)
[7.3288182e-05 9.9819857e-01 5.0156052e-07 1.3255216e-06 1.9134770e-06]
0                 goldfish   1 0.99819857
1                    macaw  88 0.0003081666
2                  axolotl  29 0.00020368141
3                 lorikeet  90 0.00014414966
4              rock beauty 392 9.927926e-05

great_white_shark (256, 256, 3)
[1.7603778e-09 4.4697251e-10 9.1863257e-01 8.0921926e-02 4.3487269e-04]
0        great white shark   2 0.91863257
1              tiger shark   3 0.080921926
2               hammerhead   4 0.0004348727
3             electric ray   5 2.7358656e-06
4                     coho 391 1.9878255e-06

ostrich (256, 256, 3)
[2.4325537e-08 1.1584138e-08 3.1243250e-08 5.4276814e-08 6.9122493e-08]
0                  ostrich   9 0.98532593
1                  bustard 138 0.012045887
2                    crane 134 0.0009492677
3                  vulture  23 0.00028382108
4                    goose  99 0.00022175076

screwdriver (256, 256, 3)
[1.6742955e-08 6.9743031e-08 2.0134753e-07 5.2668806e-08 1.1299680e-07]
0              screwdriver 784 0.6263062
1                  syringe 845 0.37200174
2                  spatula 813 0.00014389504
3               safety pin 772 0.000105540035
4            letter opener 623 9.461693e-05


sucess!

Process finished with exit code 0

'''






































