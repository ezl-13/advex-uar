import importlib
import os
import subprocess

from PIL import Image
import numpy as np

import torch
import torch.nn.functional as F
import torch.utils.data
from torchvision import datasets, transforms, models

from advex_uar.common.loader import StridedImageFolder
from advex_uar.eval.cifar10c import CIFAR10C
from advex_uar.train.trainer import Metric, accuracy, correct

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

def norm_to_pil_image(img):
    # Next three lines reverse normalization
    #img_new = torch.Tensor(img)
    img_new = reverse_normalization(img.clone().detach().cpu())
    img_new.mul_(255)
    np_img = np.rollaxis(np.uint8(img_new.numpy()), 0, 3)
    return Image.fromarray(np_img, mode='RGB')

def reverse_normalization(img):
    img_new = img.clone()
    for t, m, s in zip(img_new, IMAGENET_MEAN, IMAGENET_STD):
        t.mul_(s).add_(m)
    return img_new

class Accumulator(object):
    def __init__(self, name):
        self.name = name
        self.vals = []

    def update(self, val):
        self.vals.append(val)

    @property
    def avg(self):
        total_sum = sum([torch.sum(v) for v in self.vals])
        total_size = sum([v.size()[0] for v in self.vals])
        return total_sum / total_size

class BaseEvaluator():
    def __init__(self, **kwargs):
        default_attr = dict(
            # eval options
            model=None, batch_size=32, stride=10,
            dataset_path=None, # val dir for imagenet, base dir for CIFAR-10-C
            nb_classes=None,
            # attack options
            attack=None,
            # Communication options
            fp16_allreduce=False,
            # Logging options
            logger=None)
        default_attr.update(kwargs)
        for k in default_attr:
            setattr(self, k, default_attr[k])
        if self.dataset not in ['imagenet', 'imagenet-c', 'cifar-10', 'cifar-10-c']:
            raise NotImplementedError
        self.cuda = True
        if self.cuda:
            self.model.cuda()
        self.attack = self.attack()
        self._init_loaders()

    def _init_loaders(self):
        raise NotImplementedError

    def evaluate(self):
        self.model.eval()

        std_loss = Accumulator('std_loss')
        adv_loss = Accumulator('adv_loss')
        std_corr = Accumulator('std_corr')
        adv_corr = Accumulator('adv_corr')
        std_logits = Accumulator('std_logits')
        adv_logits = Accumulator('adv_logits')

        seen_classes = []
        adv_images = Accumulator('adv_images')
        first_batch_images = Accumulator('first_batch_images')

        from PIL import Image

        # for batch_idx, (data, target) in enumerate(self.val_loader[0]):
        #     if self.cuda:
        #         data, target = data.cuda(non_blocking=True), target.cuda(non_blocking=True)
        #     with torch.no_grad():
        #         #output = self.model(data)
        #         data_cpy = data.clone().detach()
        #         std_cpy = data.clone().detach() # std_cpy is used for finding the standard accuracy and has transforms applied as normal
        #         # data_cpy = torch.tensor([])
        #         # std_cpy = torch.tensor([])
        #         # for idx in range(len(data_cpy)):
        #         #     #print("Tensor is cuda?", data_cpy.is_cuda)
        #
        #         #     data_cpy = torch.cat((data_cpy, torch.tensor(transforms.functional.normalize(transforms.functional.to_tensor(data[idx, :]), IMAGENET_MEAN, IMAGENET_STD)      )))
        #         #     #std_cpy[idx] = transforms.functional.normalize(data[idx].clone().cpu(), IMAGENET_MEAN, IMAGENET_STD).cuda() # DELETE
        #         #     transformedTensor = applyTransforms(np.copy(data[idx, :]))
        #         #     std_cpy = torch.cat((std_cpy, torch.tensor(transforms.functional.normalize(transformedTensor.clone().cpu(), IMAGENET_MEAN, IMAGENET_STD))))
        #         #     #std_cpy[idx, :] = transforms.functional.normalize(transformedTensor.cpu(), IMAGENET_MEAN, IMAGENET_STD).cuda()
        #         #     transformedImage = norm_to_pil_image(np.array(std_cpy[idx, :].cpu()))
        #         #     transformedImage.save('sample_data/standard' + str(idx) + '.png')
        #         #     untransformedImage = norm_to_pil_image(np.array(data_cpy[idx, :].cpu()))
        #         #     untransformedImage.save('sample_data/data' + str(idx) + '.png')
        #         #     # print(np.array(data_cpy[idx].cpu()) - np.array(std_cpy[idx].cpu()))
        #         output = self.model(std_cpy)
        #         std_logits.update(output.cpu())
        #         loss = F.cross_entropy(output, target, reduction='none').cpu()
        #         std_loss.update(loss)
        #         corr = correct(output, target)
        #         corr = corr.view(corr.size()[0]).cpu()
        #         std_corr.update(corr)
        #
        #     run_output = {'std_loss':std_loss.avg,
        #                   'std_acc':std_corr.avg}
        #     print('Standard Batch', batch_idx)
        #     print(run_output)

        for batch_idx, (data, target) in enumerate(self.val_loader[1]):

            # data is normalized at this point

            if self.cuda:
                data, target = data.cuda(non_blocking=True), target.cuda(non_blocking=True)

            # for idx in range(len(data)):
            #     savedImage = norm_to_pil_image(data[idx])
            #     savedImage.save("sample_data/eric" + str(idx) + '.png')


            # with torch.no_grad():
            #     #output = self.model(data)
            #     data_cpy = data.clone().detach()
            #     std_cpy = data.clone().detach() # std_cpy is used for finding the standard accuracy and has transforms applied as normal
            #     # data_cpy = torch.tensor([])
            #     # std_cpy = torch.tensor([])
            #     # for idx in range(len(data_cpy)):
            #     #     #print("Tensor is cuda?", data_cpy.is_cuda)

            #     #     data_cpy = torch.cat((data_cpy, torch.tensor(transforms.functional.normalize(transforms.functional.to_tensor(data[idx, :]), IMAGENET_MEAN, IMAGENET_STD)      )))
            #     #     #std_cpy[idx] = transforms.functional.normalize(data[idx].clone().cpu(), IMAGENET_MEAN, IMAGENET_STD).cuda() # DELETE
            #     #     transformedTensor = applyTransforms(np.copy(data[idx, :]))
            #     #     std_cpy = torch.cat((std_cpy, torch.tensor(transforms.functional.normalize(transformedTensor.clone().cpu(), IMAGENET_MEAN, IMAGENET_STD))))
            #     #     #std_cpy[idx, :] = transforms.functional.normalize(transformedTensor.cpu(), IMAGENET_MEAN, IMAGENET_STD).cuda()
            #     #     transformedImage = norm_to_pil_image(np.array(std_cpy[idx, :].cpu()))
            #     #     transformedImage.save('sample_data/standard' + str(idx) + '.png')
            #     #     untransformedImage = norm_to_pil_image(np.array(data_cpy[idx, :].cpu()))
            #     #     untransformedImage.save('sample_data/data' + str(idx) + '.png')
            #     #     # print(np.array(data_cpy[idx].cpu()) - np.array(std_cpy[idx].cpu()))
            #     output_adv = self.model(data)
            #     adv_logits.update(output_adv.cpu())
            #     loss = F.cross_entropy(output_adv, target, reduction='none').cpu()
            #     adv_loss.update(loss)
            #     corr = correct(output_adv, target)
            #     corr = corr.view(corr.size()[0]).cpu()
            #     adv_corr.update(corr)

            rand_target = torch.randint(
                0, self.nb_classes - 1, target.size(),
                dtype=target.dtype, device='cuda')
            rand_target = torch.remainder(target + rand_target + 1, self.nb_classes)

            data_cpy = data.clone().detach()

            for idx in range(len(data_cpy)):
                # savedImage = norm_to_pil_image(data_adv[idx])
                # savedImage.save("sample_data/before_transforms" + str(idx) + '.png')
                unnormalized = reverse_normalization(data[idx])
                changed = np.swapaxes(np.array(unnormalized.cpu().detach()) * 255.0, 0, 2)

                transformed = applyTransforms(np.swapaxes(np.array(unnormalized.cpu().clone().detach()) * 255.0, 0, 2))
                data_cpy[idx] = transforms.functional.normalize(transformed.clone().cpu(), IMAGENET_MEAN, IMAGENET_STD).cuda()

            #from PIL import Image
            data_adv = self.attack(self.model, data_cpy, rand_target,
                                   avoid_target=False, scale_eps=False)

            # for idx in range(len(data)):
            #     savedImage = norm_to_pil_image(data_adv[idx])
            #     savedImage.save("sample_data/eric" + str(idx) + '.png')

            with torch.no_grad():
                output_adv = self.model(data_adv)
                adv_logits.update(output_adv.cpu())
                loss = F.cross_entropy(output_adv, target, reduction='none').cpu()
                adv_loss.update(loss)
                corr = correct(output_adv, target)
                corr = corr.view(corr.size()[0]).cpu()
                adv_corr.update(corr)

            run_output = {'adv_loss':adv_loss.avg,
                          'adv_acc':adv_corr.avg}
            print('Adv Batch', batch_idx)
            print(run_output)

        summary_dict = {'std_acc':std_corr.avg.item(),
                        'adv_acc':adv_corr.avg.item()}
        print(std_loss.avg, std_corr.avg, adv_loss.avg, adv_corr.avg)

class CIFAR10Evaluator(BaseEvaluator):
    def _init_loaders(self):
        normalize = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        self.val_dataset = datasets.CIFAR10(
                root='./', download=True, train=False,
                transform=transforms.Compose([
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.RandomApply([applyTransforms], p=1),
                    normalize]))
        self.val_adv_dataset = datasets.CIFAR10(
                root='./', download=True, train=False,
                transform=transforms.Compose([
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.ToTensor(),
                    normalize]))
        self.val_loader = [
            torch.utils.data.DataLoader(
                    self.val_dataset, batch_size=self.batch_size,
                    shuffle=False, num_workers=8, pin_memory=True),
            torch.utils.data.DataLoader(
                self.val_adv_dataset, batch_size=self.batch_size,
                shuffle=False, num_workers=8, pin_memory=True)
        ]

class CIFAR10CEvaluator(BaseEvaluator):
    def __init__(self, corruption_type=None, corruption_name=None, corruption_level=None, **kwargs):
        self.corruption_type = corruption_type
        self.corruption_name = corruption_name
        self.corruption_level = corruption_level
        super().__init__(**kwargs)

    def _init_loaders(self):
        valdir = os.path.join(self.dataset_path, 'CIFAR-10-C')
        transform = transforms.Compose(
                [transforms.ToTensor(),
                 transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                      std=[0.229, 0.224, 0.225])])
        self.val_dataset = CIFAR10C(valdir, transform=transform,
                                    corruption_name=self.corruption_name,
                                    corruption_level=self.corruption_level)
        self.val_sampler = torch.utils.data.SequentialSampler(self.val_dataset)
        self.val_loader = [torch.utils.data.DataLoader(
                self.val_dataset, batch_size=self.batch_size,
                sampler=self.val_sampler, num_workers=1, pin_memory=True,
                shuffle=False)]























import click
import importlib
import os

import numpy as np
import torch

from advex_uar.common.pyt_common import *
from advex_uar.common import FlagHolder

# The below code was obtained from the UAR-codebase /common/models/cifar10_resnet.py
'''
Properly implemented ResNet-s for CIFAR10 as described in paper [1].
The implementation and structure of this file is hugely influenced by [2]
which is implemented for ImageNet and doesn't have option A for identity.
Moreover, most of the implementations on the web is copy-paste from
torchvision's resnet and has wrong number of params.
Proper ResNet-s for CIFAR10 (for fair comparision and etc.) has following
number of layers and parameters:
name      | layers | params
ResNet20  |    20  | 0.27M
ResNet32  |    32  | 0.46M
ResNet44  |    44  | 0.66M
ResNet56  |    56  | 0.85M
ResNet110 |   110  |  1.7M
ResNet1202|  1202  | 19.4m
which this implementation indeed has.
Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
[2] https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
If you use this implementation in you work, please don't forget to mention the
author, Yerlan Idelbayev.
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from torch.autograd import Variable

__all__ = ['resnet50']

def _weights_init(m):
    classname = m.__class__.__name__
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal(m.weight)

class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)

# This ResNet50 archiecture was obtained from https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

def resnet50():
    print("Getting the 50!)")
    return ResNet(Bottleneck, [3, 4, 6, 3])


def test(net):
    import numpy as np
    total_params = 0

    for x in filter(lambda p: p.requires_grad, net.parameters()):
        total_params += np.prod(x.data.numpy().shape)
    print("Total number of params", total_params)
    print("Total layers", len(list(filter(lambda p: p.requires_grad and len(p.data.size())>1, net.parameters()))))


if __name__ == "__main__":
    for net_name in __all__:
        if net_name.startswith('resnet'):
            print(net_name)
            test(globals()[net_name]())
            print()

def get_ckpt(FLAGS):
    if FLAGS.ckpt_path is not None:
        print('Loading ckpt from {}'.format(FLAGS.ckpt_path))
        return torch.load(FLAGS.ckpt_path)
    elif FLAGS.use_wandb and FLAGS.wandb_run_id is not None:
        globals()['wandb'] = importlib.import_module('wandb')
        print('Loading ckpt from wandb run id {}'.format(FLAGS.wandb_run_id))
        api = wandb.Api()
        run = api.run("{}/{}/{}".format(
                FLAGS.wandb_username, FLAGS.wandb_ckpt_project, FLAGS.wandb_run_id))
        ckpt_file = run.file("ckpt.pth")
        ckpt_file.download(replace=False)
        os.rename('ckpt.pth', os.path.join(wandb.run.dir, 'ckpt.pth'))
        return torch.load(os.path.join(wandb.run.dir, 'ckpt.pth'))
    else:
        raise ValueError('You must specify a wandb_run_id or a ckpt_path.')

def run(**flag_kwargs):
    FLAGS = FlagHolder()
    FLAGS.initialize(**flag_kwargs)
    if FLAGS.wandb_ckpt_project is None:
        FLAGS._dict['wandb_ckpt_project'] = FLAGS.wandb_project
    if FLAGS.step_size is None:
        FLAGS.step_size = get_step_size(FLAGS.epsilon, FLAGS.n_iters, FLAGS.use_max_step)
        FLAGS._dict['step_size'] = FLAGS.step_size
    FLAGS.summary()

    #logger = init_logger(FLAGS.use_wandb, 'eval', FLAGS._dict)

    if FLAGS.dataset in ['cifar-10', 'cifar-10-c']:
        nb_classes = 10
    else:
        nb_classes = 1000 // FLAGS.class_downsample_factor

    model_dataset = FLAGS.dataset
    if model_dataset == 'imagenet-c':
        model_dataset = 'imagenet'
    model = resnet50()
    ckpt = get_ckpt(FLAGS)
    model.load_state_dict(ckpt['net'])

    attack = get_attack(FLAGS.dataset, FLAGS.attack, FLAGS.epsilon,
                        FLAGS.n_iters, FLAGS.step_size, False)

    if FLAGS.dataset == 'imagenet':
        Evaluator = ImagenetEvaluator
    elif FLAGS.dataset == 'imagenet-c':
        Evaluator = ImagenetCEvaluator
    elif FLAGS.dataset == 'cifar-10':
        Evaluator = CIFAR10Evaluator
    elif FLAGS.dataset == 'cifar-10-c':
        Evaluator = CIFAR10CEvaluator

    evaluator = Evaluator(model=model, attack=attack, dataset=FLAGS.dataset,
                          dataset_path=FLAGS.dataset_path, nb_classes=nb_classes,
                          corruption_type=FLAGS.corruption_type, corruption_name=FLAGS.corruption_name,
                          corruption_level=FLAGS.corruption_level,
                          batch_size=FLAGS.batch_size, stride=FLAGS.class_downsample_factor,
                          fp_all_reduce=FLAGS.use_fp16, tag=FLAGS.tag) # REMOVED LOGGER
    evaluator.evaluate()














# The below code was taken from the appendix of the paper "Barrage of Random Transforms for Adversarially Robust Defense"
# This paper can be found at this link: http://openaccess.thecvf.com/content_CVPR_2019/papers/Raff_Barrage_of_Random_Transforms_for_Adversarially_Robust_Defense_CVPR_2019_paper.pdf.

import numpy as np
import random
from skimage import color, transform, morphology
import skimage
from io import BytesIO
import PIL
from scipy import fftpack

# Helper functions for some transforms
def randUnifC(low, high, params=None):
  p = np.random.uniform()
  if params is not None:
    params.append(p)
  return (high-low)*p + low

def randUnifI(low, high, params=None):
  p = np.random.uniform()
  if params is not None:
    params.append(p)
  return round((high-low)*p + low)

def randLogUniform(low, high, base=np.exp(1)):
  div = np.log(base)
  return base**np.random.uniform(np.log(low)/div, np.log(high)/div)

##### TRANSFORMS BELOW #####
def colorPrecisionReduction(img):
  scales = [np.asscalar(np.random.random_integers(8, 200)) for x in range(3)]
  multi_channel = np.random.choice(2) == 0
  params = [multi_channel] + [s/200.0 for s in scales]

  if multi_channel:
    img = np.round(img*scales[0])/scales[0]
  else:
    for i in range(3):
      img[:,:,i] = np.round(img[:,:,i]*scales[i]) / scales[i]

  return img

def jpegNoise(img):
  quality = np.asscalar(np.random.random_integers(55, 95))
  params = [quality/100.0]
  pil_image = PIL.Image.fromarray((img*255.0).astype(np.uint8))
  f = BytesIO()
  pil_image.save(f, format='jpeg', quality=quality)
  jpeg_image = np.asarray(PIL.Image.open(f)).astype(np.float32) / 255.0
  return jpeg_image

def swirl(img):
  strength = (2.0-0.01)*np.random.random(1)[0] + 0.01
  c_x = np.random.random_integers(1, 256)
  c_y = np.random.random_integers(1, 256)
  radius = np.random.random_integers(10, 200)
  params = [strength/2.0, c_x/256.0, c_y/256.0, radius/200.0]
  img = skimage.transform.swirl(img, rotation=0, strength=strength, radius=radius, center=(c_x, c_y))
  return img

def fftPerturbation(img):
  r, c, _ = img.shape
  #Everyone gets the same factor to avoid too many weird artifacts
  point_factor = (1.02-0.98)*np.random.random((r,c)) + 0.98
  randomized_mask = [np.random.choice(2)==0 for x in range(3)]
  keep_fraction = [(0.95-0.0)*np.random.random(1)[0] + 0.0 for x in range(3)]
  params = randomized_mask + keep_fraction
  for i in range(3):
    im_fft = fftpack.fft2(img[:,:,i])
    # Set r and c to be the number of rows and columns of the array.
    r, c = im_fft.shape
    if randomized_mask[i]:
      mask = np.ones(im_fft.shape[:2]) > 0
      im_fft[int(r*keep_fraction[i]):int(r*(1-keep_fraction[i]))] = 0
      im_fft[:, int(c*keep_fraction[i]):int(c*(1-keep_fraction[i]))] = 0
      mask = ~mask
      #Now things to keep = 0, things to remove = 1
      mask = mask * ~(np.random.uniform(size=im_fft.shape[:2] ) < keep_fraction[i])
      #Now switch back
      mask = ~mask
      im_fft = np.multiply(im_fft, mask)
    else:
      im_fft[int(r*keep_fraction[i]):int(r*(1-keep_fraction[i]))] = 0
      im_fft[:, int(c*keep_fraction[i]):int(c*(1-keep_fraction[i]))] = 0
      #Now, lets perturb all the rest of the non-zero values by a relative factor
      im_fft = np.multiply(im_fft, point_factor)
      im_new = fftpack.ifft2(im_fft).real
      #FFT inverse may no longer produce exact same range, so clip it back
      im_new = np.clip(im_new, 0, 1)
      img[:,:,i] = im_new
  return img

## Color Space Group Below
def alterHSV(img):
  img = color.rgb2hsv(img)
  params = []
  #Hue
  img[:,:,0] += randUnifC(-0.05, 0.05, params=params)
  #Saturation
  img[:,:,1] += randUnifC(-0.25, 0.25, params=params)
  #Value
  img[:,:,2] += randUnifC(-0.25, 0.25, params=params)
  img = np.clip(img, 0, 1.0)
  img = color.hsv2rgb(img)
  img = np.clip(img, 0, 1.0)
  return img

def alterXYZ(img):
  img = color.rgb2xyz(img)
  params = []
  #X
  img[:,:,0] += randUnifC(-0.05, 0.05, params=params)
  #Y
  img[:,:,1] += randUnifC(-0.05, 0.05, params=params)
  #Z
  img[:,:,2] += randUnifC(-0.05, 0.05, params=params)
  img = np.clip(img, 0, 1.0)
  img = color.xyz2rgb(img)
  img = np.clip(img, 0, 1.0)
  return img

def alterLAB(img):
  img = color.rgb2lab(img)
  params = []
  #L
  img[:,:,0] += randUnifC(-5.0, 5.0, params=params)
  #a
  img[:,:,1] += randUnifC(-2.0, 2.0, params=params)
  #b
  img[:,:,2] += randUnifC(-2.0, 2.0, params=params)
  # L 2 [0,100] so clip it; a & b channels can have,! negative values.
  img[:,:,0] = np.clip(img[:,:,0], 0, 100.0)
  img = color.lab2rgb(img)
  img = np.clip(img, 0, 1.0)
  return img

def alterYUV(img):
  img = color.rgb2yuv(img)
  params = []
  #Y
  img[:,:,0] += randUnifC(-0.05, 0.05, params=params)
  #U
  img[:,:,1] += randUnifC(-0.02, 0.02, params=params)
  #V
  img[:,:,2] += randUnifC(-0.02, 0.02, params=params)
  # U & V channels can have negative values; clip only Y
  img[:,:,0] = np.clip(img[:,:,0], 0, 1.0)
  img = color.yuv2rgb(img)
  img = np.clip(img, 0, 1.0)
  return img

## Grey Scale Group Below
def greyScaleMix(img):
  # average of color channels, different contribution for each channel
  ratios = np.random.rand(3)
  ratios /= ratios.sum()
  params = [x for x in ratios]
  img_g = img[:,:,0] * ratios[0] + img[:,:,1] * ratios[1] + img[:,:,2] * ratios[2]
  for i in range(3):
    img[:,:,i] = img_g
  return img

def greyScalePartialMix(img):
  ratios = np.random.rand(3)
  ratios/=ratios.sum()
  prop_ratios = np.random.rand(3)
  params = [x for x in ratios] + [x for x in prop_ratios]
  img_g = img[:,:,0] * ratios[0] + img[:,:,1] * ratios[1] + img[:,:,2] * ratios[2]
  for i in range(3):
    p = max(prop_ratios[i], 0.2)
    img[:,:,i] = img[:,:,i]*p + img_g*(1.0-p)
  return img

def greyScaleMixTwoThirds(img):
  params = []
  # Pick a channel that will be left alone and remove it from the ones to be averaged
  channels = [0, 1, 2]
  remove_channel = np.random.choice(3)
  channels.remove( remove_channel)
  params.append( remove_channel )
  ratios = np.random.rand(2)
  ratios/=ratios.sum()
  params.append(ratios[0]) #They sum to one, so first item fully specifies the group
  img_g = img[:,:,channels[0]] * ratios[0] + img[:,:,channels[1]] * ratios[1]
  for i in channels:
    img[:,:,i] = img_g
  return img

def oneChannelPartialGrey(img):
  params = []
  # Pick a channel that will be altered and remove it from the ones to be averaged
  channels = [0, 1, 2]
  to_alter = np.random.choice(3)
  channels.remove(to_alter)
  params.append(to_alter)
  ratios = np.random.rand(2)
  ratios/=ratios.sum()
  params.append(ratios[0]) #They sum to one, so first item fully specifies the group
  img_g = img[:,:,channels[0]] * ratios[0] + img[:,:,channels[1]] * ratios[1]
  # Lets mix it back in with the original channel
  p = (0.9-0.1)*np.random.random(1)[0] + 0.1
  params.append( p )
  img[:,:,to_alter] = img_g*p + img[:,:,to_alter] *(1.0-p)
  return img

## Denoising Group
def gaussianBlur(img):
  if randUnifC(0, 1) > 0.5:
    sigma = [randUnifC(0.1, 0.8)]*3
  else:
    sigma = [randUnifC(0.1, 0.8), randUnifC(0.1, 0.8), randUnifC(0.1, 0.8)]
    img[:,:,0] = skimage.filters.gaussian(img[:,:,0], sigma=sigma[0])
    img[:,:,1] = skimage.filters.gaussian(img[:,:,1], sigma=sigma[1])
    img[:,:,2] = skimage.filters.gaussian(img[:,:,2], sigma=sigma[2])
  return img

def chambolleDenoising(img):
  params = []
  weight = (0.25-0.05)*np.random.random(1)[0] + 0.05
  params.append( weight )
  multi_channel = np.random.choice(2) == 0
  params.append( multi_channel )
  img = skimage.restoration.denoise_tv_chambolle( img, weight=weight, multichannel=multi_channel)
  return img

def nonlocalMeansDenoising(img):
  h_1 = randUnifC(0, 1)
  params = [h_1]
  sigma_est = np.mean(skimage.restoration.estimate_sigma(img,multichannel=True) )
  h = (1.15-0.6)*sigma_est*h_1 + 0.6*sigma_est
  #If false, it assumes some weird 3D stuff
  multi_channel = np.random.choice(2) == 0
  params.append( multi_channel )
  #Takes too long to run without fast mode.
  fast_mode = True
  patch_size = np.random.random_integers(5, 7)
  params.append(patch_size)
  patch_distance = np.random.random_integers(6, 11)
  params.append(patch_distance)
  if multi_channel:
    img = skimage.restoration.denoise_nl_means( img,h=h, patch_size=patch_size,patch_distance=patch_distance,fast_mode=fast_mode )
  else:
      for i in range(3):
          sigma_est = np.mean(skimage.restoration.estimate_sigma(img[:,:,i], multichannel=True ) )
          h = (1.15-0.6)*sigma_est*params[i] + 0.6*sigma_est
          img[:,:,i] = skimage.restoration.denoise_nl_means(img[:,:,i], h=h, patch_size=patch_size, patch_distance=patch_distance, fast_mode=fast_mode )
  return img

def applyTransforms(img):
  img = np.array(img)
  allTransforms = [[colorPrecisionReduction], [jpegNoise], [swirl], [fftPerturbation], [alterHSV, alterXYZ, alterLAB, alterYUV], [greyScaleMix, greyScalePartialMix, greyScaleMixTwoThirds, oneChannelPartialGrey], [gaussianBlur, chambolleDenoising, nonlocalMeansDenoising]]
  numTransforms = random.randint(0, 5)

  id = str(img[0][0])
  # savedImage = Image.fromarray(np.uint8(img), 'RGB')
  # savedImage.save("sample_data/transform" + str(id) + str(0) + ".png")

  #print("Original.")
  img = img / 255.0

  #for i in range(numTransforms):
  for i in range(numTransforms):
      transformGroup = random.choice(allTransforms)
      transform = random.choice(transformGroup)
      #transform = alterHSV
      # print(img)

      img = transform(img)

      # savedImage = Image.fromarray(np.uint8(img * 255.0), 'RGB')
      # savedImage.save("sample_data/transform" + str(id) + str(i + 1) + str(transform) + ".png")
      allTransforms.remove(transformGroup)

  return torch.from_numpy(np.swapaxes(img, 0, 2)).float()














@click.command()
# wandb options
@click.option("--use_wandb/--no_wandb", is_flag=True, default=True)
@click.option("--wandb_project", default=None, help="WandB project to log to")
@click.option("--tag", default='eval', help="Short tag for WandB")

# Dataset options
# Allowed values: ['imagenet', 'imagenet-c', 'cifar-10', 'cifar-10-c']
@click.option("--dataset", default='imagenet')
@click.option("--dataset_path", default=None)

# Model options
@click.option("--resnet_size", default=50)
@click.option("--class_downsample_factor", default=1, type=int)

# checkpoint options; if --ckpt_path is None, assumes that ckpt is pulled from WandB
@click.option("--ckpt_path", default=None, help="Path to the checkpoint for evaluation")
@click.option("--wandb_username", default=None, help="WandB username to pull ckpt from")
@click.option("--wandb_ckpt_project", default=None, help='WandB project to pull ckpt from')
@click.option("--wandb_run_id", default=None,
              help='If --use_wandb is set, WandB run_id to pull ckpt from.  Otherwise'\
              'a run_id which will be associated with --ckpt_path')

# Evaluation options
@click.option("--use_fp16/--no_fp16", is_flag=True, default=False)
@click.option("--batch_size", default=128)

# Options for ImageNet-C and CIFAR-10-C
@click.option("--corruption_type", default=None)
@click.option("--corruption_name", default=None)
@click.option("--corruption_level", default=None)

# Attack options
# Allowed values: ['pgd_linf', 'pgd_l2', 'fw_l1', 'jpeg_linf', 'jpeg_l2', 'jpeg_l1', 'elastic', 'fog', 'gabor', 'snow']
@click.option("--attack", default=None, type=str)
@click.option("--epsilon", default=16.0, type=float)
@click.option("--step_size", default=None, type=float)
@click.option("--use_max_step", is_flag=True, default=False)
@click.option("--n_iters", default=50, type=int)

def main(**flags):
    run(**flags)

if __name__ == '__main__':
    main()