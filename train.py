import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from dataset import PokemonDataset
from models import get_model
import pandas as pd


def train_epoch(model, loader, criterion, optimizer, device):
    if len(loader.dataset) == 0:
        return 0.0, 0.0
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    for step, (x, y) in enumerate(loader, start=1):
        x = x.to(device)
        y = y.to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()*x.size(0)
        _,pred = out.max(1)
        correct += (pred==y).sum().item()
        total += x.size(0)
        if step % 20 == 0 or step == len(loader):
            print(f"  [train] step {step}/{len(loader)}", flush=True)
    return total_loss/total, correct/total


def eval_epoch(model, loader, criterion, device):
    if len(loader.dataset) == 0:
        return 0.0, 0.0
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for step, (x, y) in enumerate(loader, start=1):
            x = x.to(device)
            y = y.to(device)
            out = model(x)
            loss = criterion(out,y)
            total_loss += loss.item()*x.size(0)
            _,pred = out.max(1)
            correct += (pred==y).sum().item()
            total += x.size(0)
            if step % 20 == 0 or step == len(loader):
                print(f"  [val] step {step}/{len(loader)}", flush=True)
    return total_loss/total, correct/total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='data')
    parser.add_argument('--model', default='resnet50')
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--pretrained', type=lambda x: x.lower()=='true', default='True')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"[info] device={device}", flush=True)

    transform_train = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    transform_val = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
    ])

    print("[info] loading datasets...", flush=True)
    trainset = PokemonDataset(args.data_dir, 'train', transform_train)
    valset = PokemonDataset(args.data_dir, 'val', transform_val)
    print(f"[info] train samples={len(trainset)}, val samples={len(valset)}", flush=True)

    num_classes = len(trainset.classes) if len(trainset.classes)>0 else 2
    print(f"[info] num_classes={num_classes}", flush=True)

    train_loader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(valset, batch_size=args.batch_size, shuffle=False)
    print(
        f"[info] train batches={len(train_loader)}, val batches={len(val_loader)}",
        flush=True
    )

    if len(trainset) == 0 or len(valset) == 0:
        raise RuntimeError(
            "Dataset is empty. Run download_data.py first and check data/train, data/val."
        )

    print(f"[info] building model={args.model}, pretrained={args.pretrained}", flush=True)
    model = get_model(args.model, num_classes, pretrained=args.pretrained)
    model = model.to(device)
    print("[info] model ready", flush=True)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    os.makedirs('checkpoints', exist_ok=True)
    history = []
    best_acc = 0.0
    for ep in range(args.epochs):
        print(f"[info] starting epoch {ep+1}/{args.epochs}", flush=True)
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = eval_epoch(model, val_loader, criterion, device)
        print(
            f"Epoch {ep+1}/{args.epochs} "
            f"Train loss:{train_loss:.4f} Acc:{train_acc:.4f}  "
            f"Val loss:{val_loss:.4f} Acc:{val_acc:.4f}",
            flush=True
        )
        history.append({'epoch':ep+1,'train_loss':train_loss,'train_acc':train_acc,'val_loss':val_loss,'val_acc':val_acc})
        df = pd.DataFrame(history)
        df.to_csv('history.csv', index=False)
        if val_acc>best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), f'checkpoints/{args.model}_best.pth')

if __name__=='__main__':
    main()
