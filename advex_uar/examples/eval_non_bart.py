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
        #         std_cpy = data.clone().detach()
        #         output = self.model(std_cpy)
        #         std_logits.update(output.cpu())
        #         loss = F.cross_entropy(output, target, reduction='none').cpu()
        #         std_loss.update(loss)
        #         corr = correct(output, target)
        #         corr = corr.view(corr.size()[0]).cpu()
        #         std_corr.update(corr)
        
        #     run_output = {'std_loss':std_loss.avg,
        #                   'std_acc':std_corr.avg}
        #     print('Standard Batch', batch_idx)
        #     print(run_output)

        for batch_idx, (data, target) in enumerate(self.val_loader[1]):

            # data is normalized at this point

            if self.cuda:
                data, target = data.cuda(non_blocking=True), target.cuda(non_blocking=True)

            rand_target = torch.randint(
                0, self.nb_classes - 1, target.size(),
                dtype=target.dtype, device='cuda')
            rand_target = torch.remainder(target + rand_target + 1, self.nb_classes)

            from PIL import Image
            data_adv = self.attack(self.model, data, rand_target,
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
                    transforms.ToTensor(),
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