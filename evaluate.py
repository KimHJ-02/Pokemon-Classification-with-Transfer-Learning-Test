import argparse
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from dataset import PokemonDataset
from models import get_model
from sklearn.metrics import classification_report
import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint')
    parser.add_argument('--model', default='resnet50')
    parser.add_argument('--data_dir', default='data')
    args = parser.parse_args()

    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
    ])
    valset = PokemonDataset(args.data_dir, 'val', transform)
    loader = DataLoader(valset, batch_size=32, shuffle=False)
    num_classes = len(valset.classes) if len(valset.classes)>0 else 2
    model = get_model(args.model, num_classes, pretrained=False)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    if args.checkpoint:
        model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()
    ys = []
    yps = []
    with torch.no_grad():
        for x,y in loader:
            x = x.to(device)
            out = model(x)
            _,pred = out.max(1)
            ys.append(y.numpy())
            yps.append(pred.cpu().numpy())
    ys = np.concatenate(ys)
    yps = np.concatenate(yps)
    report = classification_report(ys, yps, output_dict=True)
    df = pd.DataFrame(report).transpose()
    df.to_csv('classification_report.csv')
    print('Saved classification_report.csv')

if __name__=='__main__':
    main()
