# Copyright (C) 2021 * Ltd. All rights reserved.
# author : Sanghyun Jo <shjo.april@gmail.com>

import os
import cv2
import torch
import shutil

import numpy as np

from torch import nn
from torch.nn import functional as F

from torchvision import transforms
from torch.utils.data import DataLoader

from PIL import Image

from core import networks, datasets

from tools.ai import torch_utils, trainers
from tools.general import io_utils, log_utils

if __name__ == '__main__':
    ###################################################################################
    # 1. Arguments
    ###################################################################################
    parser = io_utils.Parser()

    # dataset
    parser.add('gpus', '0', str)

    parser.add('seed', 1, int)
    parser.add('num_workers', 8, int)
    
    parser.add('root_dir', 'H:/COCO2014/train/image/', str)
    parser.add('style_path', './data/vg_starry_night.jpg', str)
    
    # hyperparameters
    parser.add('image_size', 256, int)
    parser.add('batch_size', 4, int)
    
    parser.add('tag', 'Transformer', str)
    
    # training
    parser.add('reset', True, bool)
    parser.add('max_epochs', 2, int)

    parser.add('optimizer', 'Adam', str)

    parser.add('lr', 1e-3, float)
    
    # for debugging
    parser.add('debug', False, bool)

    # style transfer
    parser.add('content_weight', 1e5, float)
    parser.add('style_weight', 1e10, float)

    args = parser.get_args()
    
    ###################################################################################
    # 2. Set GPU option
    ###################################################################################
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    device = torch.device('cuda')
    
    ###################################################################################
    # 3. Make directories and pathes.
    ###################################################################################
    txt_dir = './experiments/txt/'
    model_dir = f'./experiments/models/{args.tag}/'
    tensorboard_dir = f'./experiments/tensorboards/{args.tag}/'

    txt_path = txt_dir + f'{args.tag}.txt'

    if args.reset and os.path.isdir(txt_dir):
        if os.path.isfile(txt_path):
            os.remove(txt_path)
        if os.path.isdir(model_dir):
            shutil.rmtree(model_dir)
        if os.path.isdir(tensorboard_dir):
            shutil.rmtree(tensorboard_dir)

    txt_dir = io_utils.create_directory(txt_dir)
    model_dir = io_utils.create_directory(model_dir)
    tensorboard_dir = io_utils.create_directory(tensorboard_dir)

    ###################################################################################
    # 4. Set the seed number and define log function. 
    ###################################################################################
    torch_utils.set_seed(args.seed)
    log_func = lambda string='': log_utils.log_print(string, txt_path)

    ###################################################################################
    # 5. Make transformation
    ###################################################################################
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]
    
    train_transform = transforms.Compose(
        [
            transforms.Resize(args.image_size),
            transforms.RandomCrop(args.image_size),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.mul(255))
        ]
    )

    test_transform = transforms.Compose(
        [
            transforms.Resize(args.image_size),
            transforms.CenterCrop(args.image_size),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.mul(255))
        ]
    )

    log_func('[i] train transform : {}'.format(train_transform))
    log_func('[i] test transform : {}'.format(test_transform))
    
    ###################################################################################
    # 6. Make dataset and loader
    ###################################################################################
    train_dataset = datasets.Unlabeled_Dataset(args.root_dir, train_transform)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True, drop_last=True, pin_memory=True)
    
    log_func('[i] The size of training set is {}'.format(len(train_dataset)))
    
    ###################################################################################
    # 7. Make Network
    ###################################################################################
    model = networks.Transformer(in_channels=3)
    
    model.to(device)
    model.train()

    # extract gram matrix
    pretrained_model = networks.VGG16(requires_grad=False)
    pretrained_model.to(device)
    pretrained_model.eval()
    
    style_image = Image.open(args.style_path)
    style_image = test_transform(style_image)
    style_image = style_image.repeat(args.batch_size, 1, 1, 1).to(device)

    ###################################################################################
    # 8. Define losses and optimizer
    ###################################################################################
    l2_loss_fn = nn.MSELoss().to(device)
    optimizer_option = {'params': model.parameters(), 'lr':args.lr, 'weight_decay': 0.}
    
    #################################################################################################
    # 9. Training
    #################################################################################################
    class Trainer(trainers.Trainer):
        def __init__(self, 
                content_weight, style_weight,
                style_image, pretrained_model,
                **kwargs
            ):
            super().__init__(**kwargs)

            self.content_weight = content_weight
            self.style_weight = style_weight

            self.style_image = style_image
            self.pretrained_model = pretrained_model

            self.style_features, self.style_grams = self.infer_vgg16(self.style_image)

            self.l2_loss = self.losses[0]

        def gram_matrix(self, y):
            (b, ch, h, w) = y.size()

            features = y.view(b, ch, w * h)
            features_t = features.transpose(1, 2)
            
            gram = features.bmm(features_t) / (ch * h * w)
            return gram

        def infer_vgg16(self, x, with_gram=True):
            features = pretrained_model(x)
            
            if with_gram:
                grams = [self.gram_matrix(features[key]) for key in features.keys()]
            else:
                grams = None
            
            return features, grams

        def preprocess(self, data):
            return {'images': data}

        def calculate_losses(self, data):
            images = data['images'].to(self.device)
            
            # inference
            stylized_images = self.model(images)

            # get feature maps from content images
            content_features, _ = self.infer_vgg16(images, with_gram=False)
            stylized_content_features, stylized_content_grams = \
                self.infer_vgg16(stylized_images, with_gram=True)

            # calculate losses
            content_loss = self.l2_loss(
                content_features['relu2_2'], 
                stylized_content_features['relu2_2']
            )

            style_loss = 0.
            for content_gram, style_gram in zip(stylized_content_grams, self.style_grams):
                style_loss += self.l2_loss(content_gram, style_gram)

            content_loss *= self.content_weight
            style_loss *= self.style_weight

            loss = content_loss + style_loss
            
            return [loss, content_loss, style_loss], {}
        
        def update_tensorboard(self, losses, _):
            self.writer.add_scalar('Training/Loss', losses[0], self.iteration)
            self.writer.add_scalar('Training/Content_Loss', losses[1], self.iteration)
            self.writer.add_scalar('Training/Style_Loss', losses[2], self.iteration)
            self.writer.add_scalar('Training/Learning_Rate', self.get_learning_rate(), self.iteration)
    
    trainer_params = {
        # parent variables
        'model' : model,
        'losses' : [l2_loss_fn],
        'log_names' : ['loss', 'content_loss', 'style_loss'],
        
        'loader' : train_loader,
        'max_epochs' : args.max_epochs,
        
        'optimizer' : args.optimizer, 'optimizer_option' : optimizer_option,
        'scheduler' : None, 'scheduler_option' : {},

        'amp' : False, 'ema':None, 'tensorboard_dir' : tensorboard_dir,
        'device' : device, 'RANK': 0, 'WORLD_SIZE' : 0,

        # main variables
        'content_weight': args.content_weight, 
        'style_weight': args.style_weight,

        'style_image': style_image,
        'pretrained_model': pretrained_model,
    }
    trainer = Trainer(**trainer_params)
    
    for epoch in range(args.max_epochs):
        # training
        (loss, content_loss, style_loss), train_time = trainer.step(debug=args.debug)

        # save weights
        torch_utils.save_model(model, model_dir + f'ep={epoch+1}.pth')
        
        # visualize log
        data = {
            'epoch' : epoch + 1,
            'loss' : float(loss),
            'content_loss' : float(content_loss),
            'style_loss' : float(style_loss),
            'time' : train_time
        }
        log_func('[i] epoch={epoch:,}, loss={loss:.4f}, content_loss={content_loss:.4f}, style_loss={style_loss:.4f}, time={time:.0f}sec'.format(**data))
