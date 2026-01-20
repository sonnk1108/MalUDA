import argparse
import os
import gc
import math
import random
import itertools
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np
from utils.feature_selection import continuous_domain_wd_and_mi
from utils.evaluation import evaluate_target_and_source
from utils.dataset import TabularImageDataset, TabularDataset
from models.feature_extractor import Classifier, SourceExtractor, SourceNExtractor
from models.swd import swd_loss
from utils.preprocessing import preprocess_data_balanced
import lr_schedule

def parse_args():
    parser = argparse.ArgumentParser(description="Train a deep learning model")
    parser.add_argument("--source", type=str, required=True, help="Source dataset")
    parser.add_argument("--target", type=str, required=True, help="Target dataset")
    parser.add_argument("--num_epochs", type=int, required=False,default=100)
    parser.add_argument("--batch_size", type=int, required=False,default=2048)
    return parser.parse_args()
def get_lambda(epoch, total_epochs, max_lambda=1.0):
    p = epoch / total_epochs
    return max_lambda * (2. / (1. + np.exp(-10 * p)) - 1)
def main():
    args = parse_args()
    print(f"Source: {args.source}, Target: {args.target}")
    batch_size = args.batch_size
    num_epochs= args.num_epochs
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # Load data

    # Initialize model
    df_src = pd.read_csv(f"data/{args.source}.csv").drop(
            columns=['Flow ID', 'Timestamp'], errors='ignore'
        )
    df_tgt = pd.read_csv(f"data/{args.target}.csv").drop(
            columns=['Flow ID', 'Timestamp'], errors='ignore'
        )

    X_a, y_a = preprocess_data_balanced(df_src)
    X_b, y_b = preprocess_data_balanced(df_tgt)

    # -------------------------
    # 2. Split TARGET (CRITICAL)
    # -------------------------
    X_b_train, X_b_test, y_b_train, y_b_test = train_test_split(
        X_b, y_b,
        test_size=0.3,
        random_state=42,
        stratify=y_b
    )
    # -------------------------
    # 3. Scaling (fit on SOURCE only)
    # -------------------------
    scaler = StandardScaler()
    X_a_scaled = scaler.fit_transform(X_a)
    X_b_train_scaled = scaler.transform(X_b_train)
    X_b_test_scaled = scaler.transform(X_b_test)
    X_a_scaled = pd.DataFrame(X_a_scaled, columns=X_a.columns)
    X_b_train_scaled = pd.DataFrame(X_b_train_scaled, columns=X_b.columns)
    X_b_test_scaled = pd.DataFrame(X_b_test_scaled, columns=X_b.columns)
    # -------------------------
    # 4. Feature Selection (KEPT, but SAFE)
    # -------------------------
    print("Step 1: Feature Selection")
    cont_features = X_a.select_dtypes(include=['int64', 'float64']).columns.tolist()

    print(cont_features)
    col_con = continuous_domain_wd_and_mi(
        X_a_scaled,
        X_b_train_scaled,   # ONLY target train
        cont_features,
        y_a, 
        r=0.9
    )
    selected_cols = col_con
    print(selected_cols)
    X_a_scaled = X_a_scaled[selected_cols]
    X_b_train_scaled = X_b_train_scaled[selected_cols]
    X_b_test_scaled = X_b_test_scaled[selected_cols]
    source_dataset = TabularDataset(
        X_a_scaled.values, y_a
    )
    target_train_dataset = TabularDataset(
        X_b_train_scaled.values, y_b_train
    )
    target_test_dataset = TabularDataset(
        X_b_test_scaled.values, y_b_test
    )
    source_loader = DataLoader(
        source_dataset, batch_size=batch_size, shuffle=True, num_workers=0
    )
    target_train_loader = DataLoader(
        target_train_dataset, batch_size=batch_size, shuffle=True, num_workers=0
    )
    target_test_loader = DataLoader(
        target_test_dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )



    # -------------------------
    # 6. Model setup
    # -------------------------
    print("Step 2: Model Configuration")

    latent_dim = 64
    num_classes = len(np.unique(y_a))
    input_dim=X_a_scaled.values.shape[1]
    source_extractor = SourceNExtractor(input_dim=input_dim).to(device)
    classifier = Classifier(
        input_dim=latent_dim,
        num_classes=num_classes
    ).to(device)

    optimizer = optim.SGD(
        [
            {'params': source_extractor.parameters(), 'lr': 1},
            {'params': classifier.parameters(), 'lr': 1},
        ],
        momentum=0.9,
        weight_decay=0.0005,
        nesterov=True
    )

    param_lr = []
    for param_group in optimizer.param_groups:
        param_lr.append(param_group["lr"])
    schedule_param = {"gamma": 0.0003, "power": 0.75}
    lr_scheduler = lr_schedule.schedule_dict["inv_swd"]

    cls_loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)


    # -------------------------
    # 7. Training (INDUCTIVE UDA)
    # -------------------------
    print("Step 3: Training")

    best_val_loss=1
    counter=0
    iter_num=0
    scaler = torch.cuda.amp.GradScaler()  # define once before the epoch loop
    best_acc=0

    for epoch in range(num_epochs):
    
        # ---- lambda & entropy schedule ----
        if epoch < 3:
            lambda_global = 0.0
            lambda_local = 0.0
            enp = 0.0
            confidence_threshold = 0.0  # not used yet

        else:
            lambda_global = get_lambda(epoch, num_epochs)
            lambda_local  =  0.1 # get_lambda(epoch, num_epochs)
            enp = 0.05
            if epoch < 15:
                confidence_threshold = 0.99  # easier early on
            else:
                confidence_threshold = 0.8
        tgt_iter = itertools.cycle(target_train_loader)
        source_extractor.train()
        classifier.train()
        epoch_cls, epoch_swd_g, epoch_swd_l, epoch_ent = 0., 0., 0., 0.
        batches = 0
    
        for src_x, src_y in source_loader:
    
            optimizer = lr_scheduler(param_lr, optimizer, iter_num, **schedule_param)
            optimizer.zero_grad()
    
            tgt_x, _ = next(tgt_iter)
            src_x, src_y = src_x.to(device), src_y.to(device)
            tgt_x = tgt_x.to(device)
    
            with torch.cuda.amp.autocast():
    
                # ---- forward ----
                src_feat = source_extractor(src_x)
                tgt_feat = source_extractor(tgt_x)
    
                src_pred = classifier(src_feat)
                tgt_pred = classifier(tgt_feat)
    
                # ---- losses ----
                cls_loss = cls_loss_fn(src_pred, src_y)
                if lambda_global > 0:
                    swd_global = swd_loss(src_feat, tgt_feat,device=device)
                else:
                    swd_global = torch.tensor(0.0, device=device)
    
                # entropy (target)
                if enp > 0:
                    p = torch.softmax(tgt_pred, dim=1)
                    entropy_loss = -(p * torch.log(p + 1e-6)).sum(dim=1).mean()
                else:
                    entropy_loss = torch.tensor(0.0, device=device)
            
                # ---- local swd (safe) ----
                swd_local = torch.tensor(0.0, device=device)
    
                if lambda_local > 0:
                    mask_s0 = (src_y == 0)
                    src_feat_n = src_feat[mask_s0]
    
                    probs_tgt = torch.softmax(tgt_pred, dim=1)
                    tgt_lbl = probs_tgt.argmax(dim=1)
                    mask_t0 = (tgt_lbl == 0) & (probs_tgt[:, 0] > confidence_threshold)
                    tgt_feat_n = tgt_feat[mask_t0]
    
                    if src_feat_n.size(0) > 5 and tgt_feat_n.size(0) > 5:
                        swd_local = swd_loss(src_feat_n, tgt_feat_n,device=device)
    
                # ---- total loss ----
                total_loss = (
                    cls_loss
                    + lambda_global * swd_global
                    + lambda_local  * swd_local
                    + enp * entropy_loss #entropy loss included to make the training more stable
                )
    
            # ---- backward ----
            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()
    
            # ---- logging ----
            epoch_cls   += cls_loss.item()
            epoch_swd_g += swd_global.item()
            epoch_swd_l += swd_local.item()
            epoch_ent   += entropy_loss.item()
    
            batches += 1
            iter_num += 1
    
        # ---- epoch log ----
        print(
            f"Epoch [{epoch+1}/{num_epochs}] | "
            f"Cls: {epoch_cls/batches:.4f} | "
            f"SWD_g: {epoch_swd_g/batches:.4f} | "
            f"SWD_l: {epoch_swd_l/batches:.4f} | "
            f"Ent: {epoch_ent/batches:.4f} | "
            f"λg: {lambda_global:.3f}, λl: {lambda_local:.3f}"
        )

    
        if epoch % 5==0:
            acc, prec, rec, f1, src_err = evaluate_target_and_source(
            target_data_loader=target_test_loader,
            target_extractor=source_extractor,
            classifier=classifier,
            source_data_loader=source_loader,
            source_extractor=source_extractor,
            average='binary',
            device=device)
            if acc > best_acc:
                best_acc=acc
                print(f"ACC={acc:.4f}, P={prec:.4f}, R={rec:.4f}, F1={f1:.4f},scr_err: {src_err}")

    print("Training finished. Best model saved with acc: ",best_acc)
    
if __name__ == "__main__":
    main()
