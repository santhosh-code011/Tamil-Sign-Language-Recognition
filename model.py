"""
Tamil Sign Language Recognition
Module: model.py
Tamil Vowel Dictionary & Hybrid Vision Transformer (ViT) Architecture
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

# 12 Primary Tamil Vowels (உயிர் எழுத்துக்கள்)
TAMIL_ALPHABET = [
    {'id': 0,  'code': 'A',  'tamil': 'அ', 'translit': 'a',  'name': 'Akkaram (Short A)',  'desc': 'Fist with thumb upright'},
    {'id': 1,  'code': 'AA', 'tamil': 'ஆ', 'translit': 'aa', 'name': 'Aakaaram (Long AA)', 'desc': 'Fist with thumb extended horizontally'},
    {'id': 2,  'code': 'I',  'tamil': 'இ', 'translit': 'i',  'name': 'Ikkaram (Short I)',  'desc': 'Closed fist with pinky upright'},
    {'id': 3,  'code': 'EE', 'tamil': 'ஈ', 'translit': 'ee', 'name': 'Eekaaram (Long EE)', 'desc': 'Index and pinky upright (horns)'},
    {'id': 4,  'code': 'U',  'tamil': 'உ', 'translit': 'u',  'name': 'Ukkaram (Short U)',  'desc': 'Index and middle fingers together upright'},
    {'id': 5,  'code': 'OO', 'tamil': 'ஊ', 'translit': 'oo', 'name': 'Ookaaram (Long OO)', 'desc': 'Index and middle fingers split (V-sign)'},
    {'id': 6,  'code': 'E',  'tamil': 'எ', 'translit': 'e',  'name': 'Ekkaram (Short E)',  'desc': 'Hand flat with 4 fingers bent 90 deg'},
    {'id': 7,  'code': 'AE', 'tamil': 'ஏ', 'translit': 'ae', 'name': 'Ekaaram (Long AE)',  'desc': 'Open palm, all 5 fingers fully spread'},
    {'id': 8,  'code': 'AI', 'tamil': 'ஐ', 'translit': 'ai', 'name': 'Aikaaram (AI)',      'desc': 'Thumb, index, and pinky extended (I-L-Y)'},
    {'id': 9,  'code': 'O',  'tamil': 'ஒ', 'translit': 'o',  'name': 'Okkaram (Short O)',  'desc': 'Curled fingers touching thumb forming O-ring'},
    {'id': 10, 'code': 'OA', 'tamil': 'ஓ', 'translit': 'oa', 'name': 'Oakaaram (Long OA)', 'desc': 'All fingers arched forming wide oval dome'},
    {'id': 11, 'code': 'AU', 'tamil': 'ஔ', 'translit': 'au', 'name': 'Aukaaram (AU)',      'desc': 'Thumb and pinky extended (Shaka sign)'}
]

CLASS_NAMES = [x['code'] for x in TAMIL_ALPHABET]

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim=128, num_heads=4, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, return_attn=False):
        b, n, c = x.shape
        qkv = self.qkv(x).reshape(b, n, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = F.softmax(scores, dim=-1)
        out = (self.drop(attn) @ v).transpose(1, 2).reshape(b, n, c)
        out = self.drop(self.proj(out))
        return (out, attn) if return_attn else out

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim=128, num_heads=4, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadSelfAttention(embed_dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.Dropout(dropout)
        )
    def forward(self, x, return_attn=False):
        if return_attn:
            attn_out, weights = self.attn(self.norm1(x), return_attn=True)
            x = x + attn_out
            x = x + self.mlp(self.norm2(x))
            return x, weights
        else:
            x = x + self.attn(self.norm1(x))
            x = x + self.mlp(self.norm2(x))
            return x

class TamilSignViT(nn.Module):
    """
    Vision Transformer (ViT) for Real-Time Tamil Sign Language Recognition.
    Processes hand poses through a Residual Tokenizer Stem into Multi-Head Self-Attention layers.
    """
    def __init__(self, num_classes=12, embed_dim=128, num_heads=4, depth=3):
        super().__init__()
        res = models.resnet18(weights=None)
        self.stem = nn.Sequential(res.conv1, res.bn1, res.relu, res.maxpool, res.layer1, res.layer2)
        self.pos_embed = nn.Parameter(torch.randn(1, 64, embed_dim) * 0.02)
        self.blocks = nn.ModuleList([TransformerBlock(embed_dim, num_heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes)
        )

    def forward(self, x, return_attn=False):
        feat = self.stem(x)
        tokens = feat.flatten(2).transpose(1, 2) + self.pos_embed
        last_attn = None
        for b in self.blocks:
            if return_attn:
                tokens, last_attn = b(tokens, return_attn=True)
            else:
                tokens = b(tokens)
        pooled = self.norm(tokens.mean(dim=1))
        logits = self.head(pooled)
        return (logits, last_attn) if return_attn else logits
