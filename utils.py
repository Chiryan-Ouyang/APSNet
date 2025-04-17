import os
import torch
import scipy.io as sio
import numpy as np
from torch.autograd import Variable
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
import torch.nn.functional as F
import pdb
from torchvision import transforms
from PIL import Image
import time
from seg_losses import *
from matrics import *
from sklearn.metrics import roc_curve, auc
import csv


def iou_score_gpu(output, target):
    smooth = 1e-5
    if isinstance(output, list):
        output = output[-1]
    
    if output.dim() == 4:  
        # [B, C*H*W]
        B = output.size(0)
        output = output.view(B, -1)
        target = target.view(B, -1)
    elif output.dim() == 3:  #  [C, H, W]
        # [C*H*W]
        output = output.view(-1)
        target = target.view(-1)
    
    output = torch.sigmoid(output)
    
    output_ = (output > 0.5).float()
    target_ = (target > 0.5).float()
    
    if output.dim() == 2: 
        intersection = (output_ * target_).sum(dim=1)
        union = (output_ + target_ - output_ * target_).sum(dim=1)
        iou = (intersection + smooth) / (union + smooth)
        return iou.mean().item() 
    else:  
        intersection = (output_ * target_).sum()
        union = (output_ + target_ - output_ * target_).sum()
        return (intersection + smooth) / (union + smooth).item()


def dice_coef_gpu(output, target):
    smooth = 1e-5
    if isinstance(output, list):
        output = output[-1]
    
    if output.dim() == 4:  #  [B, C, H, W]
        # [B, C*H*W]
        B = output.size(0)
        output = output.view(B, -1)
        target = target.view(B, -1)
    elif output.dim() == 3:  #  [C, H, W]
        #  [C*H*W]
        output = output.view(-1)
        target = target.view(-1)
    
    output = torch.sigmoid(output)
    
    output = (output >= 0.5).float()
    
    if output.dim() == 2: 
        intersection = (output * target).sum(dim=1)
        dice = (2. * intersection + smooth) / (output.sum(dim=1) + target.sum(dim=1) + smooth)
        return dice.mean().item() 
    else: 
        intersection = (output * target).sum()
        return (2. * intersection + smooth) / (output.sum() + target.sum() + smooth).item()


class Trainer(object):
    def __init__(self, model, optimizer, save_dir=None, save_freq=1):
        self.model = model
        self.optimizer = optimizer
        self.save_dir = save_dir
        self.save_freq = save_freq
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.best_model_path = None
        self.best_accuracy = 0.0  # You can use this for best model selection based on accuracy
        
    def _loop(self, data_loader, ep, is_train=True):
        loop_loss_class, loop_loss_seg, loop_iou, loop_dice = [], [], [], []
        correct_samples = 0 
        mode = 'train' if is_train else 'test'
        all_targets, all_predictions = [], []
        
        total_samples = len(data_loader.dataset)
        total_iterations = len(data_loader)
        
        pbar = tqdm(
            enumerate(data_loader), 
            desc=f"{mode} Epoch {ep}", 
            unit="iter", 
            total=total_iterations, 
            position=0, 
            leave=True
        )
        
        total_loss_class = 0.0
        total_loss_seg = 0.0
        total_samples_seen = 0
        total_iou = 0.0
        total_dice = 0.0
        total_correct = 0
        
        with torch.set_grad_enabled(is_train):
            for iteration, (data, tar, label) in pbar:
                batch_size = data.size(0)
                total_samples_seen += batch_size
                
                data, tar, label = data.to(self.device), tar.to(self.device), label.to(self.device)
                
                out_class, out_seg = self.model(data)
                
                loss_class = F.cross_entropy(out_class, label)
                loss_seg = F.binary_cross_entropy(torch.sigmoid(out_seg.view(batch_size, -1)), tar.view(batch_size, -1)) + \
                          IOULoss(out_seg, tar)
                loss = loss_class + loss_seg

                total_loss_class += loss_class.item() * batch_size
                total_loss_seg += loss_seg.item() * batch_size
                
                pred_class = out_class.max(1)[1]
                correct_count = (pred_class == label).sum().item()
                total_correct += correct_count
                
                batch_iou = 0.0
                batch_dice = 0.0
                
                iou = iou_score_gpu(out_seg, tar)
                dice = dice_coef_gpu(out_seg, tar)
                total_iou += iou * batch_size
                total_dice += dice * batch_size
                batch_iou = iou
                batch_dice = dice
                
                for j in range(batch_size):
                    actual_class = label[j].item()
                    predicted_prob = torch.softmax(out_class[j], dim=0).detach().cpu().numpy()
                    actual_prob = np.zeros(out_class.shape[1])
                    actual_prob[actual_class] = 1.0
                    all_targets.append(actual_prob)
                    all_predictions.append(predicted_prob)
                
                if is_train:
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    
                pbar.set_postfix({
                    'loss_class': total_loss_class / total_samples_seen,
                    'loss_seg': total_loss_seg / total_samples_seen,
                    'acc': total_correct / total_samples_seen,
                    'iou': batch_iou,
                    'dice': batch_dice
                })

        avg_loss_class = total_loss_class / total_samples
        avg_loss_seg = total_loss_seg / total_samples
        avg_accuracy = total_correct / total_samples
        avg_iou = total_iou / total_samples
        avg_dice = total_dice / total_samples

        all_targets = np.array(all_targets)
        all_predictions = np.array(all_predictions)
        fpr, tpr, _ = roc_curve(all_targets.ravel(), all_predictions.ravel())
        fnr = 1 - tpr
        tnr = 1 - fpr
        auc_score = auc(fpr, tpr)
        
        threshold = 0.5
        binary_predictions = (all_predictions[:, 1] >= threshold).astype(int)

        # TP, FP, TN, FN
        TP = np.sum((binary_predictions == 1) & (all_targets[:, 1] == 1))
        FP = np.sum((binary_predictions == 1) & (all_targets[:, 1] == 0))
        TN = np.sum((binary_predictions == 0) & (all_targets[:, 1] == 0))
        FN = np.sum((binary_predictions == 0) & (all_targets[:, 1] == 1))

        # Sen, Spe, Ppv, Npv
        SEN = TP / (TP + FN + 1e-6)
        SPE = TN / (TN + FP + 1e-6)
        PPV = TP / (TP + FP + 1e-6)
        NPV = TN / (TN + FN + 1e-6)
        F1 = 2 * (PPV * SEN) / (PPV + SEN + 1e-6)
        
        print(f"{mode}_clas: loss_class: {avg_loss_class:.6f}, Acc: {avg_accuracy:.6%}, Sen: {SEN:.6f}, Spe: {SPE:.6f}, Ppv: {PPV:.6f}, Npv: {NPV:.6f}, AUC: {auc_score:.6f}, F1: {F1:.6f}")
        print(f"{mode}_seg: loss_seg: {avg_loss_seg:.6f}, iou: {avg_iou:.6%}, dice: {avg_dice:.6f}")

        return avg_loss_class, avg_accuracy, SEN, SPE, PPV, NPV, auc_score, F1

    def train(self, data_loader, ep):
        self.model.train() 
        results = self._loop(data_loader, ep, is_train=True)
        return results

    def test(self, data_loader, ep):
        self.model.eval() 
        results = self._loop(data_loader, ep, is_train=False)
        return results

    def loop(self, epochs, train_loader, test_loader, scheduler, save_freq, test_freq=1):
        f = open(self.save_dir + 'testlog.txt', 'w')
        f.close()
        for ep in range(1, epochs + 1):
            if scheduler is not None:
                scheduler.step()
            print('epoch {}'.format(ep))
            train_results = np.array(self.train(train_loader, ep))
            
            if ep == 1 or ep % test_freq == 0 or ep == epochs:
                test_results = np.array(self.test(test_loader, ep))
                
                # Save the results for each test to a log file
                with open(self.save_dir + 'testlog.txt', 'a') as f:
                    f.write(f"{ep}\t"
                            f"{test_results[0]:.6f}\t{test_results[1]:.6f}\t{test_results[2]:.6f}\t"
                            f"{test_results[3]:.6f}\t{test_results[4]:.6f}\t{test_results[5]:.6f}\t{test_results[6]:.6f}\t{test_results[7]:.6f}\n")
                
                if test_results[1] > self.best_accuracy:
                    self.best_accuracy = test_results[1]
                    self.best_model_path = self.save_dir + str(ep) + '_best'  +  '_models.pth'
                    torch.save(self.model.state_dict(), self.best_model_path)
                    # Save the predicted and true classification labels to CSV
                    self.save_predictions(test_loader)
            
            if not ep % save_freq:
                self.save(ep)  # Save model weights
                    
        print('Training finished!')

    def save(self, epoch, **kwargs):
        if self.save_dir:
            name = self.save_dir + 'train' + str(epoch) + 'models.pth'
            torch.save(self.model.state_dict(), name)
            # torch.save(self.model, name)
            
    def get_best_model_path(self):
            return self.best_model_path
        
    def save_predictions(self, data_loader):
        self.model.eval()
        predictions = []
        true_labels = []
        file_names = []

        total_iterations = len(data_loader)
        batch_idx_offset = 0
        
        pbar = tqdm(
            enumerate(data_loader), 
            desc="Saving predictions", 
            unit="iter", 
            total=total_iterations, 
            position=0, 
            leave=True
        )
        
        with torch.no_grad():
            for iteration, (data, tar, label) in pbar:
                data, tar, label = data.to(self.device), tar.to(self.device), label.to(self.device)
                out_class, _ = self.model(data)
                predicted_class = out_class.argmax(dim=1).cpu().numpy()
                true_labels.extend(label.cpu().numpy())
                predictions.extend(predicted_class.tolist())

                acc = (predicted_class == label.cpu().numpy()).mean()
                pbar.set_postfix({'acc': acc})

                batch_size = data.size(0)
                if hasattr(data_loader.dataset, 'imgs'):
                    for i in range(batch_size):
                        idx = batch_idx_offset + i
                        if idx < len(data_loader.dataset.imgs):
                            file_name = data_loader.dataset.imgs[idx][0]
                            file_names.append(file_name)
                
                batch_idx_offset += batch_size

        file_path = os.path.join(self.save_dir, 'predictions.csv')
        with open(file_path, 'w', newline='') as csvfile:
            fieldnames = ['File_Name', 'True_Label', 'Predicted_Label']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for name, true, pred in zip(file_names, true_labels, predictions):
                writer.writerow({'File_Name': name, 'True_Label': true, 'Predicted_Label': pred})
                