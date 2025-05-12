import argparse
import os
import random
import json
from datetime import datetime

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

from model import FBNETGEN, GNNPredictor

from datasets import DatasetMCI, DatasetASD, DatasetMDD
from utils import accuracy, sensitivity, specificity
from losses import intra_loss, inter_loss, sparse_loss

import warnings
from setproctitle import *

# 특정 UserWarning 메시지 무시
warnings.filterwarnings("ignore", category=UserWarning)

# torch.autograd.set_detect_anomaly(True)
setproctitle('project2')

def set_seed(seed):

        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
    
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.cuda.manual_seed(seed)  # pytorch (both CPU and CUDA)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.enabled = False

def main(params, device, log_path):

    acc, sen, spe, f1, roc = [], [], [], [], []
    lambda_gl1 = 1.0
    lambda_gl2 = 0.001
    lambda_sl = 0.0001

    loss_weight = [lambda_gl1, lambda_gl2, lambda_sl] 

    params["lambda_gl1"] = lambda_gl1
    params["lambda_gl2"] = lambda_gl2
    params["lambda_sl"] = lambda_sl

    with open(os.path.join(log_path, "config.json"),"w") as f:
        json.dump(params, f, indent = 4)

    print(f"\nArguments\n{params}\n")

    for k in range(1,6):
        set_seed(params['seed'])
        print("identify:",params['id'])
        
        train_dataset = DatasetMDD(type='train',k=k, threshold=15)
        test_dataset = DatasetMDD(type='test',k=k, threshold=15)

        label_weight = train_dataset.weight
        node_num = train_dataset.num_node
        time_length = train_dataset.time_length
        
        print("Label Weight", label_weight)
        print("Number of Nodes", node_num)
        print("Length of Bold Signal", time_length)

        train_dataloader = DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=True)
        test_dataloader = DataLoader(test_dataset, batch_size=params['batch_size'], shuffle=True)

        feature_dim = node_num

        if params['model'] == 'fbnetgen':
            model=FBNETGEN(num_node=node_num, 
                        node_feature_dim=feature_dim, 
                        embed_dim=params['feature_dim'], 
                        time_length=time_length,
                        extractor_type=params['extractor_type'],
                        device=device)
        elif params['model'] == 'cls':
            model = GNNPredictor(node_input_dim=node_num, roi_num=node_num)
        else:
            raise
        
        model.to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=params['learning_rate'], weight_decay=0.0001) 
        criterion = nn.CrossEntropyLoss(weight=torch.Tensor(label_weight).to(device))
        schedular = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.998)


        # Train
        print("\nStart Training for fold",k)
        model.train()

        for epoch in tqdm(range(int(params['epochs'])), desc=f"Training Fold {k}", ncols=80):
            epoch_losses = [0,0,0,0]
            epoch_loss = 0
            total_num=0
            true_pred=0

            for i, data in enumerate(train_dataloader):
                
                x = data['bold'].to(device)
                pcc = data['non_fisher'].to(device)
                adj = data['non_fisher_adj'].to(device)
                y = data['label'].to(device)

                if params['model'] == 'fbnetgen':
                    logit, edge = model(x, pcc)

                    ce_loss = criterion(logit, y)
                    gl1_loss = intra_loss(y, edge)
                    gl2_loss = inter_loss(y, edge)
                    sl_loss = sparse_loss(edge)/y.shape[0]

                    total_loss = ce_loss + lambda_gl1*gl1_loss + lambda_gl2*gl2_loss + lambda_sl*sl_loss

                    epoch_losses[0] += ce_loss.item()
                    epoch_losses[1] += gl1_loss.item()
                    epoch_losses[2] += gl2_loss.item()
                    epoch_losses[3] += sl_loss.item()

                else:
                    logit = model(pcc, adj)

                    total_loss = criterion(logit, y)

                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

                epoch_loss += total_loss.item()

                total_num += y.shape[0]
                pred = logit.detach().argmax(dim=1)
                true_pred += (pred==y).float().sum().item()
            schedular.step()
            
            if params['model'] == "fbnetgen":
                tqdm.write(f"Epoch{epoch} ACC {true_pred/total_num:.4f} Loss {epoch_loss/(i+1):.4f} CE {epoch_losses[0]/(i+1):.4f} GL1 {epoch_losses[1]/(i+1):.4f} GL2 {epoch_losses[2]/(i+1):.4f} SL {epoch_losses[3]/(i+1):.4f}")
            
            else:
                tqdm.write(f"Epoch{epoch} ACC {true_pred/total_num:.4f} Loss {epoch_loss/(i+1):.4f}")
            

        print("\nSaving Model for fold",k)
        torch.save(model.state_dict(), os.path.join(log_path,f"model_fold_{k}.pt"))
        if os.path.exists(os.path.join(log_path,f"model_fold_{k}.pt")):
            print("Save model!")


        # Test
        print("\nEvaluation for fold",k)
        model.eval()

        temp_loss = 0
        labels = []
        preds = []
        adj_list = []
        pcc_list = []
        
        cnt = 0
        for i, data in enumerate(test_dataloader):
            with torch.no_grad():
                x = data['bold'].to(device)
                pcc = data['non_fisher'].to(device)
                adj = data['non_fisher_adj'].to(device)
                y = data['label'].to(device)

                if params['model'] == 'fbnetgen':
                    logit, edge = model(x, pcc)

                    ce_loss = criterion(logit, y)
                    gl1_loss = intra_loss(y, edge)
                    gl2_loss = inter_loss(y, edge)
                    sl_loss = sparse_loss(edge)/y.shape[0]

                    total_loss = ce_loss + lambda_gl1*gl1_loss + lambda_gl2*gl2_loss + lambda_sl*sl_loss

                    epoch_losses[0] += ce_loss.item()
                    epoch_losses[1] += gl1_loss.item()
                    epoch_losses[2] += gl2_loss.item()
                    epoch_losses[3] += sl_loss.item()

                    
                    adj_list.append(edge)
                    pcc_list.append(pcc)
                else:
                    logit = model(pcc, adj)
                    
                    total_loss = criterion(logit, y)

            labels.append(y)
            preds.append(logit.detach().argmax(dim=1))
            temp_loss += total_loss.item()

        if params['model'] == 'fgnetgen':
            adj = torch.cat(adj_list).detach().cpu().numpy()
            pcc = torch.cat(pcc_list).detach().cpu().numpy()
            np.save(os.path.join(log_path, f"adj_{k}.npy"), adj)
            np.save(os.path.join(log_path, f"pcc_{k}.npy"), pcc)

        labels = torch.cat(labels).tolist()
        preds = torch.cat(preds).tolist()

        test_loss = temp_loss/(i+1)
        test_acc = accuracy(preds, labels)
        test_sen = sensitivity(preds, labels)
        test_spe = specificity(preds, labels)
        test_f1 = f1_score(labels, preds)
        test_roc = roc_auc_score(labels, preds)

        print(f"Fold{k} acc {test_acc:.4f} sen {test_sen:.4f} spe {test_spe:.4f} f1 {test_f1:.4f} roc {test_roc:.4f} loss {test_loss:.4f}")
        
        print("-"*30,"\n")
        acc.append(test_acc)
        sen.append(test_sen)
        spe.append(test_spe)
        f1.append(test_f1)
        roc.append(test_roc)


    print("Final Result")
    print(params["dataset"], params["model"], params['id'])
    temp_dict = {}
    for i, (ac, se, sp, f, r) in enumerate(zip(acc, sen, spe, f1, roc)):
        print(f"Fold {i+1}| ACC {ac:.4f} SEN {se:.4f} SPE {sp:.4f} F1 {f:.4f} ROC {r:.4f}")
        temp_dict[f'fold{i+1}'] = {"accuracy":ac, "sensitivity":se, "speciticity":sp, "f1-score":f, "roc-auc":r}
    print('-'*30)
    print("AVG| ACC {:0.4f}±{:0.4f} SEN {:0.4f}±{:0.4f} SPE {:0.4f}±{:0.4f} F1 {:0.4f}±{:0.4f} ROC {:0.4f}±{:0.4f}".format(
        np.mean(acc), np.std(acc), np.mean(sen), np.std(sen), np.mean(spe), np.std(spe), np.mean(f1), np.std(f1), np.mean(roc), np.std(roc)
    ))
    temp_dict["average"] = {"accuracy":np.mean(acc), "sensitivity":np.mean(sen), "speciticity":np.mean(spe), "f1-score":np.mean(f1), "roc-auc":np.mean(roc)}
    temp_dict["std"] = {"accuracy":np.std(acc), "sensitivity":np.std(sen), "speciticity":np.std(spe), "f1-score":np.std(f1), "roc-auc":np.std(roc)}

    with open(os.path.join(log_path, "result.json"),"w") as f:
        json.dump(temp_dict, f, indent = 4)
    if os.path.exists(os.path.join(log_path, "result.json")):
            print("Save Results!")



if __name__=="__main__":

    parser = argparse.ArgumentParser(description="This script processes train and test for FC.")
    parser.add_argument('-s', '--seed', default=100, type=int, help='Please give a value for seed')
    parser.add_argument('-g', '--gpu_id', default=2, type=int, help="Please give a value for gpu id")

    parser.add_argument('-d', '--dataset', default='MDD', type=str, help="Please give a value for dataset name")
    parser.add_argument('-m', '--model', default="fbnetgen", type=str, help="Please give a value for model name (fbnetgen or cls)")

    parser.add_argument('-e', '--epochs', default=500, type=int, help="Please give a value for epoch")
    parser.add_argument('-b', '--batch_size', default=200, type=int, help="Please give a value for mini-batch")
    parser.add_argument('-l', '--learning_rate', default=1e-04, type=float, help="Please give a value for learning rate")
    parser.add_argument('-f', '--feature_dim', default=8, type=int, help="Please give a value for feature dimension")
    parser.add_argument('-t', '--type', default="abd2", type=str, help="Please give a value for graph encoder")
    parser.add_argument('--extractor_type', default="cnn", type=str, help="Please give a value for extractor_type")

    args = parser.parse_args()

    out = "out/"
    if not os.path.exists(out):
        os.makedirs(out)

    ## GPU
    if args.gpu_id not in [2, 3]:
        device = torch.device('cpu')
        print("\nDevice: cpu")
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        current_gpu = torch.cuda.current_device()


        print(f"\nDevice: {device}, GPU index: {args.gpu_id}")
        print(f"GPU name: {torch.cuda.get_device_name(current_gpu)}")

    ## ID
    identify = datetime.now().strftime("%y%m%d%H%M%S")
    print("identify:",identify)

    ## Seed
    seed = args.seed
    os.environ["PYTHONHASHSEED"] = str(seed)  # set PYTHONHASHSEED env var at fixed value
    
    # log
    ## Path
    log_path = os.path.join(out,args.model, args.dataset, identify)
    if not os.path.exists(log_path):
        os.makedirs(log_path)

    ## config
    args_dict = vars(args)
    args_dict["id"] = str(identify)

    main(args_dict,device, log_path)