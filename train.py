"""
Tamil Sign Language Recognition
Module: train.py
Synthetic Hand Pose Generator, ViT Training Pipeline, & Evaluation Matrix
"""

import os
import sys
import math
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from sklearn.metrics import classification_report, confusion_matrix

if sys.platform.startswith('win'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

from model import TAMIL_ALPHABET, CLASS_NAMES, TamilSignViT

SKIN_COLORS = [(198, 134, 105), (165, 107, 79), (218, 160, 126), (140, 85, 58)]
BG_COLORS = [(240, 242, 245), (225, 230, 238), (245, 238, 230)]

def draw_tamil_gesture(code, size=64):
    """Draw synthetic hand gesture with distinct finger configurations."""
    img = Image.new('RGB', (size, size), BG_COLORS[np.random.randint(0, len(BG_COLORS))])
    draw = ImageDraw.Draw(img)
    skin = SKIN_COLORS[np.random.randint(0, len(SKIN_COLORS))]
    shadow = (max(0, skin[0]-40), max(0, skin[1]-40), max(0, skin[2]-40))

    cx, cy = size // 2, size // 2 + 4
    # Wrist & Palm
    draw.rectangle([cx - 9, cy + 10, cx + 9, size], fill=skin)
    draw.ellipse([cx - 13, cy - 12, cx + 13, cy + 14], fill=skin, outline=shadow)

    def finger(start, length, angle, w=4):
        rad = math.radians(angle)
        end = (start[0] + length * math.sin(rad), start[1] - length * math.cos(rad))
        draw.line([start, end], fill=shadow, width=w+2)
        draw.line([start, end], fill=skin, width=w)
        return end

    if code == 'A':
        finger((cx - 8, cy), 14, -40, w=5)
    elif code == 'AA':
        finger((cx - 8, cy), 14, -85, w=5)
    elif code == 'I':
        finger((cx + 8, cy - 6), 16, 5, w=4)
    elif code == 'EE':
        finger((cx - 7, cy - 6), 16, -10, w=4)
        finger((cx + 7, cy - 6), 16, 10, w=4)
    elif code == 'U':
        finger((cx - 3, cy - 6), 18, -2, w=4)
        finger((cx + 3, cy - 6), 18, 2, w=4)
    elif code == 'OO':
        finger((cx - 4, cy - 6), 18, -20, w=4)
        finger((cx + 4, cy - 6), 18, 20, w=4)
    elif code == 'E':
        for fx in [-6, -2, 2, 6]:
            p = finger((cx + fx, cy - 6), 8, 0, w=4)
            finger(p, 8, 90, w=4)
    elif code == 'AE':
        for idx, (fx, ang) in enumerate([(-8, -40), (-5, -15), (0, 0), (5, 15), (8, 35)]):
            finger((cx + fx, cy - 6), 16, ang, w=4)
    elif code == 'AI':
        finger((cx - 8, cy), 14, -50, w=5)
        finger((cx - 5, cy - 6), 16, -10, w=4)
        finger((cx + 8, cy - 6), 15, 15, w=4)
    elif code == 'O':
        draw.ellipse([cx - 8, cy - 12, cx + 8, cy + 4], outline=skin, width=5)
    elif code == 'OA':
        draw.ellipse([cx - 10, cy - 15, cx + 10, cy + 3], outline=skin, width=6)
    elif code == 'AU':
        finger((cx - 9, cy), 15, -60, w=5)
        finger((cx + 9, cy - 4), 15, 60, w=4)

    return img.filter(ImageFilter.SMOOTH)

def generate_tsl_dataset(base_dir='data', train_n=35, test_n=10):
    print("Generating Tamil Sign Language benchmark dataset...")
    demo_dir = os.path.join(base_dir, 'demo_samples')
    os.makedirs(demo_dir, exist_ok=True)
    splits = {'train': train_n, 'test': test_n}

    for split, count in splits.items():
        for item in TAMIL_ALPHABET:
            code = item['code']
            folder = os.path.join(base_dir, split, code)
            os.makedirs(folder, exist_ok=True)
            for i in range(count):
                im = draw_tamil_gesture(code)
                im.save(os.path.join(folder, f"{code.lower()}_{i+1:03d}.png"))
                if split == 'test' and i == 0:
                    im.save(os.path.join(demo_dir, f"{code.lower()}_sample.png"))
    print("Tamil Sign Language dataset ready!")

tf = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class SignDataset(Dataset):
    def __init__(self, fps, lbls):
        self.fps, self.lbls = fps, lbls
    def __len__(self):
        return len(self.fps)
    def __getitem__(self, i):
        return tf(Image.open(self.fps[i]).convert('RGB')), torch.tensor(self.lbls[i], dtype=torch.long)

def run_training(epochs=8, batch_size=16, lr=0.001):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if not os.path.exists('data/train'):
        generate_tsl_dataset()

    def get_split(s):
        fps, lbls = [], []
        for item in TAMIL_ALPHABET:
            folder = os.path.join('data', s, item['code'])
            for f in os.listdir(folder):
                if f.endswith('.png'):
                    fps.append(os.path.join(folder, f))
                    lbls.append(item['id'])
        return fps, lbls

    tr_x, tr_y = get_split('train')
    te_x, te_y = get_split('test')
    train_loader = DataLoader(SignDataset(tr_x, tr_y), batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(SignDataset(te_x, te_y), batch_size=batch_size, shuffle=False)

    model = TamilSignViT(num_classes=12).to(device)
    crit = nn.CrossEntropyLoss()
    opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    sch = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    best_acc = 0.0
    print("\nTraining TamilSignViT on:", device)
    for ep in range(1, epochs + 1):
        model.train()
        r_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            out = model(x)
            loss = crit(out, y)
            loss.backward()
            opt.step()
            r_loss += loss.item() * len(y)

        model.eval()
        c, tot = 0, 0
        with torch.no_grad():
            for x, y in test_loader:
                preds = model(x.to(device)).argmax(dim=1).cpu()
                c += (preds == y).sum().item()
                tot += len(y)
        acc = (c / tot) * 100.0
        sch.step()
        print(f"Epoch {ep:02d} | Train Loss: {r_loss/len(tr_x):.4f} | Val Accuracy: {acc:.2f}%")

        if acc >= best_acc:
            best_acc = acc
            torch.save({'state_dict': model.state_dict(), 'best_acc': acc}, 'tamil_sign_vit.pt')

    print(f"\nModel trained! Saved -> 'tamil_sign_vit.pt' (Peak Acc: {best_acc:.2f}%)")
    evaluate_vit()

def evaluate_vit():
    def get_test():
        fps, lbls = [], []
        for item in TAMIL_ALPHABET:
            folder = os.path.join('data/test', item['code'])
            for f in os.listdir(folder):
                if f.endswith('.png'):
                    fps.append(os.path.join(folder, f))
                    lbls.append(item['id'])
        return fps, lbls

    te_x, te_y = get_test()
    loader = DataLoader(SignDataset(te_x, te_y), batch_size=16, shuffle=False)
    m = TamilSignViT(num_classes=12)
    m.load_state_dict(torch.load('tamil_sign_vit.pt', map_location='cpu')['state_dict'])
    m.eval()

    preds, trues = [], []
    with torch.no_grad():
        for x, y in loader:
            preds.extend(m(x).argmax(dim=1).numpy())
            trues.extend(y.numpy())

    labels = [f"{x['code']} ({x['translit']})" for x in TAMIL_ALPHABET]
    print("\n" + "="*55)
    print("      TAMIL SIGN LANGUAGE ViT REPORT")
    print("="*55)
    print(classification_report(trues, preds, target_names=labels, digits=4))

    cm = confusion_matrix(trues, preds)
    plt.figure(figsize=(7, 6))
    plt.imshow(cm, cmap=plt.cm.Greens)
    plt.colorbar()
    plt.title('Tamil Sign Language ViT Confusion Matrix')
    plt.xticks(range(12), [x['code'] for x in TAMIL_ALPHABET], rotation=35)
    plt.yticks(range(12), [x['code'] for x in TAMIL_ALPHABET])
    for i in range(12):
        for j in range(12):
            plt.text(j, i, str(cm[i, j]), ha='center', va='center',
                     color='white' if cm[i, j] > cm.max()/2 else 'black')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=200)
    plt.close()
    print("Confusion Matrix Saved -> 'confusion_matrix.png'\n")

if __name__ == '__main__':
    run_training()
