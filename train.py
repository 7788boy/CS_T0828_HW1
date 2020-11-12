import torch
from torch.utils.data.dataset import Dataset
from torch.optim import RMSprop
from torch import optim
from torch.cuda import amp
from torch.nn import CrossEntropyLoss
from torchvision.transforms import transforms
from torch.utils.data.dataloader import DataLoader
import torchvision.models as models
from torch import nn, hub
from torch.utils.tensorboard import SummaryWriter
from efficientnet_pytorch import EfficientNet
from tqdm import tqdm

import time
import numpy as np
import csv
from PIL import Image
import argparse

class TrainingData(Dataset):
    def __init__(self, transform, part=None) -> None:
        super().__init__()
        with open('cs-t0828-2020-hw1/training_labels.csv') as label_file:
            reader = csv.reader(label_file)
            next(reader)  # Pop out the title: ['id', 'label']
            label_dict = {}  # format {label: {'instance_num': int, 'label_id': int}}
            label_id_counter = 0
            data_list = []
            for img_id, label in reader:  # Each line is ['#imgnum', 'label']
                if label not in label_dict.keys():
                    label_dict[label] = {'instance_num': 0, 'label_id': label_id_counter}
                    label_id_counter += 1
                else:
                    label_dict[label]['instance_num'] += 1
                data_list.append([int(img_id), label_dict[label]['label_id']])
  
            if part is None:
                self.data_list = np.array(data_list)
            else:
                self.data_list = np.array(data_list)[part]

            self.transform = transform
            self.data_num = self.data_list.shape[0]
            self.id_2_label = list(label_dict.keys())

    def __getitem__(self, index):
        img_id, label = self.data_list[index]
        img: Image.Image = Image.open('cs-t0828-2020-hw1/training_data/training_data/' + '%06d.jpg' % img_id).convert('RGB')
        img = self.transform(img)
        return img, label

    def __len__(self):
        return self.data_num

class TestingData(Dataset):
    def __init__(self, transform) -> None:
        super().__init__()
        with open('cs-t0828-2020-hw1/test_data_list.txt') as list_file:
            self.test_list = []
            for line in list_file:
                self.test_list.append(line[:-1])
            self.transform = transform
            self.data_num = len(self.test_list)

    def __getitem__(self, index):
        img_path = self.test_list[index]
        img: Image.Image = Image.open('cs-t0828-2020-hw1/' + img_path).convert('RGB')
        img = self.transform(img)
        return img, img_path[-10:-4]

    def __len__(self):
        return self.data_num

class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.efficientnet = EfficientNet.from_pretrained('efficientnet-b4')
        self.efficientnet._fc.out_features = 196 # as same as class number

    def forward(self, img):
        return self.efficientnet(img)

def parser_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--mode', 
        choices=['train', 'test'], 
        default='train',
        help='choose mode for training or testing.')
    parser.add_argument(
        '--epochs', 
        default=300, 
        type=int,
        help='epochs count for training.')
    parser.add_argument(
        '--batch', 
        default=16, 
        type=int,
        help='Batch size for training.')
    parser.add_argument(
        '--ckptID', 
        default=1111, 
        type=int,
        help='Checkpoint ID for testing.(4 digit int)')
    args = parser.parse_args()
    return args

# Training or testing
if __name__ == '__main__':
    args = parser_args()
    # ----- Model ----- #
    if args.mode == "train":
        model = Classifier().cuda()
        # ----- Data Loader ----- #
        preprocess = transforms.Compose([
            transforms.Resize((300, 300)),
            transforms.ToTensor(),
        ])
        preprocess_val = transforms.Compose([
            transforms.Resize((300, 300)),
            transforms.ToTensor(),
        ])

        val_index = np.random.choice(11185, 1185, replace=False)
        val_mask = np.zeros(11185, dtype=np.bool)
        val_mask[val_index] = True
        train_mask = np.logical_not(val_mask)
        dataset = TrainingData(preprocess, train_mask)
        dataset_val = TrainingData(preprocess_val, val_mask)
        data_loader = DataLoader(dataset, batch_size=args.batch, num_workers=8, drop_last=True, shuffle=True)
        val_data_loader = DataLoader(dataset_val, batch_size=args.batch, num_workers=8, drop_last=True, shuffle=True)

        # ----- Optimizer ----- #
        optimizer = optim.RMSprop(params=model.parameters(), lr=0.00001, weight_decay=0.0001)
        loss_function = CrossEntropyLoss()
        scaler = amp.GradScaler()
        p = list(model.parameters())[-2]

        # ---- Training Loop ---- #
        writer = SummaryWriter()
        best_acc = 0

        for epoch in range(args.epochs):
            iterator = tqdm(data_loader)
            for step, (input_img, gt_label) in enumerate(iterator):
                optimizer.zero_grad()
                pred_labels = model(input_img.cuda())
                loss = loss_function(pred_labels, gt_label.cuda())
                loss.backward()
                optimizer.step()
                iterator.set_description(f"epoch:{epoch}, loss:{loss.detach().cpu():{5}.{6}}")


            correct_count = 0
            model.eval()
            for step, (val_img, val_label) in enumerate(tqdm(val_data_loader)):
                val_img = val_img.cuda()
                val_label = val_label.cuda()
                pred = model(val_img).detach()
                pred_label = torch.argmax(pred, dim=1)
                loss_val = loss_function(pred, val_label)
                writer.add_scalar('Val_Loss', loss_val, step)
                correct_count += torch.sum(pred_label == val_label)
            
            acc = correct_count / 1185.0
            print('val:', epoch, ', acc:', acc.detach().cpu(), ', best_acc:', best_acc)
            model.train()

            if acc > best_acc:
                print('***** save model *****')
                best_acc = acc
                torch.save(model, 'checkpoints/checkpoint_%04d.pth'%epoch)

    elif args.mode == "test":
        model = torch.load('checkpoints/checkpoint_%04d.pth' % args.ckptID).eval().cuda()
        preprocess_test = transforms.Compose([
            transforms.Resize((300, 300)),
            transforms.ToTensor(),
        ])
        DATA_ROOT = 'cs-t0828-2020-hw1/'
        dataset = TrainingData(None)
        dataset_test = TestingData(preprocess_test)
        data_loader = DataLoader(dataset_test, batch_size=args.batch, num_workers=8)

        with open('result.csv', 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['id','label'])
            for img, ids in data_loader:
                pred_labels: torch.Tensor = model(img.cuda())
                pred_labels = pred_labels.detach().cpu()

                for label, img_id in zip(pred_labels, ids):
                    writer.writerow([img_id, dataset.id_2_label[torch.argmax(label)]])
