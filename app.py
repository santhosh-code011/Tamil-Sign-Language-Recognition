"""
Tamil Sign Language Recognition
Module: app.py
Interactive Dark Web Dashboard, ViT Attention Heatmap, & CLI Predictor
"""

import os
import io
import sys
import base64
import argparse
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torchvision.transforms as transforms
from flask import Flask, request, jsonify, render_template_string

if sys.platform.startswith('win'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

from model import TAMIL_ALPHABET, CLASS_NAMES, TamilSignViT
from train import generate_tsl_dataset

app = Flask(__name__)
MODEL_PATH = 'tamil_sign_vit.pt'
DEMO_DIR = 'data/demo_samples'

if not os.path.exists(DEMO_DIR):
    generate_tsl_dataset()

model = TamilSignViT(num_classes=12)
if os.path.exists(MODEL_PATH):
    model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu')['state_dict'])
    model.eval()

tf = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

HTML_PAGE = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Tamil Sign Language ViT | Sign-to-Tamil Text</title>
    <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@400;600;700&family=Noto+Sans+Tamil:wght@700;800&display=swap" rel="stylesheet">
    <style>
        :root { --bg: #090d16; --card: rgba(18, 24, 38, 0.85); --border: rgba(255,255,255,0.08); --tamil: #f59e0b; }
        * { box-sizing: border-box; margin: 0; padding: 0; font-family: 'Outfit', sans-serif; }
        body { background: var(--bg); color: #f3f4f6; padding: 24px; min-height: 100vh; }
        .container { max-width: 1150px; margin: 0 auto; }
        header { display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid var(--border); padding-bottom: 20px; margin-bottom: 24px; }
        .grid { display: grid; grid-template-columns: 1fr 1.3fr; gap: 24px; }
        .card { background: var(--card); border: 1px solid var(--border); border-radius: 16px; padding: 22px; backdrop-filter: blur(12px); }
        .alpha-grid { display: grid; grid-template-columns: repeat(6, 1fr); gap: 8px; margin-bottom: 18px; }
        .btn-l { background: rgba(255,255,255,0.05); border: 1px solid var(--border); color: #fff; padding: 10px 4px; border-radius: 8px; cursor: pointer; text-align: center; }
        .btn-l:hover { background: rgba(245, 158, 11, 0.2); border-color: var(--tamil); }
        .t-char { font-family: 'Noto Sans Tamil', sans-serif; font-size: 20px; font-weight: 700; color: var(--tamil); }
        .t-code { font-size: 10px; opacity: 0.7; }
        .upload-box { border: 2px dashed rgba(255,255,255,0.15); border-radius: 12px; padding: 22px; text-align: center; cursor: pointer; }
        .hero { background: linear-gradient(135deg, rgba(245,158,11,0.15), rgba(6,182,212,0.08)); border: 1px solid rgba(245,158,11,0.3); border-radius: 14px; padding: 20px; display: flex; align-items: center; justify-content: space-between; margin-bottom: 18px; }
        .big-t { font-family: 'Noto Sans Tamil', sans-serif; font-size: 72px; font-weight: 800; color: #fbbf24; text-shadow: 0 0 20px rgba(245,158,11,0.4); line-height: 1; }
        .sentence-box { background: rgba(0,0,0,0.3); border: 1px solid var(--border); border-radius: 10px; padding: 12px 16px; display: flex; justify-content: space-between; align-items: center; }
        .sentence-text { font-family: 'Noto Sans Tamil', sans-serif; font-size: 24px; color: #67e8f9; min-height: 36px; }
        .vis-row { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; margin-top: 16px; }
        .vis-box { border: 1px solid var(--border); border-radius: 10px; overflow: hidden; background: #000; text-align: center; }
        .vis-box img { width: 100%; display: block; }
        .vis-cap { padding: 4px; font-size: 11px; opacity: 0.7; }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <div>
                <h2>🖐️ Tamil Sign Language Recognition</h2>
                <div style="font-size: 13px; color: #9ca3af;">Real-Time Sign-to-Tamil Text via Vision Transformer (ViT)</div>
            </div>
            <div style="background: rgba(6,182,212,0.15); color: #22d3ee; padding: 6px 14px; border-radius: 20px; font-size: 12px; font-weight: 600;">ViT MHSA ATTENTION</div>
        </header>

        <div class="grid">
            <div>
                <div class="card">
                    <h3 style="margin-bottom: 12px;">1-Click Tamil Vowels (உயிர் எழுத்துக்கள்)</h3>
                    <div class="alpha-grid">
                        {% for item in alpha %}
                        <button class="btn-l" onclick="runSample('{{ item.code }}')">
                            <div class="t-char">{{ item.tamil }}</div>
                            <div class="t-code">{{ item.code }}</div>
                        </button>
                        {% endfor %}
                    </div>

                    <h3 style="margin: 18px 0 10px 0;">Upload Custom Hand Gesture</h3>
                    <div class="upload-box" onclick="document.getElementById('f').click()">
                        <div>📷 Upload Photo (PNG/JPG)</div>
                    </div>
                    <input type="file" id="f" accept="image/*" style="display:none;" onchange="uploadImage(event)">
                </div>
            </div>

            <div>
                <div class="card">
                    <h3 style="margin-bottom: 14px;">Vision Transformer Translation Output</h3>
                    <div class="hero">
                        <div id="tamilChar" class="big-t">அ</div>
                        <div style="flex: 1; margin-left: 20px;">
                            <div id="tamilName" style="font-size: 20px; font-weight: 700;">Akkaram (Short A)</div>
                            <div id="tamilDesc" style="font-size: 13px; color: #06b6d4; margin-top: 4px;">Fist with thumb upright</div>
                            <div id="conf" style="font-size: 13px; color: #34d399; font-weight: 600; margin-top: 4px;">Confidence: 99.4%</div>
                        </div>
                    </div>

                    <h4>Tamil Sentence Builder (Sign Accumulator)</h4>
                    <div class="sentence-box">
                        <div id="sentenceDisplay" class="sentence-text">அ</div>
                        <div>
                            <button onclick="sentence+=' '; document.getElementById('sentenceDisplay').innerText=sentence" style="padding: 4px 8px; border-radius: 4px; cursor: pointer;">Space</button>
                            <button onclick="sentence=''; document.getElementById('sentenceDisplay').innerText=''" style="padding: 4px 8px; border-radius: 4px; cursor: pointer;">Clear</button>
                        </div>
                    </div>

                    <div class="vis-row">
                        <div class="vis-box">
                            <img id="inImg" src="" alt="Input">
                            <div class="vis-cap">Input Sign Pose</div>
                        </div>
                        <div class="vis-box">
                            <img id="attnImg" src="" alt="Attention">
                            <div class="vis-cap">ViT Patch Attention Heatmap</div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script>
        let sentence = 'அ';
        async function runSample(code) {
            const res = await fetch(`/api/sample/${code}`);
            const data = await res.json();
            render(data);
        }
        async function uploadImage(e) {
            const file = e.target.files[0];
            if (!file) return;
            const fd = new FormData();
            fd.append('image', file);
            const res = await fetch('/api/predict', { method: 'POST', body: fd });
            const data = await res.json();
            render(data);
        }
        function render(data) {
            document.getElementById('tamilChar').innerText = data.tamil;
            document.getElementById('tamilName').innerText = `${data.name} (${data.translit})`;
            document.getElementById('tamilDesc').innerText = data.desc;
            document.getElementById('conf').innerText = `Confidence: ${(data.confidence * 100).toFixed(1)}%`;
            sentence += data.tamil;
            document.getElementById('sentenceDisplay').innerText = sentence;
            document.getElementById('inImg').src = 'data:image/png;base64,' + data.in_b64;
            document.getElementById('attnImg').src = 'data:image/png;base64,' + data.attn_b64;
        }
        window.onload = () => runSample('A');
    </script>
</body>
</html>
"""

def im_to_b64(pil_im):
    buf = io.BytesIO()
    pil_im.save(buf, format='PNG')
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')

def attn_to_b64(attn):
    # Mean attention over heads, reshape to 8x8 grid
    m_attn = attn.mean(dim=0).cpu().numpy()[0, :64].reshape(8, 8)
    fig, ax = plt.subplots(figsize=(3, 3), dpi=120)
    fig.patch.set_facecolor('#090d16')
    ax.imshow(m_attn, cmap='inferno', interpolation='bicubic')
    ax.axis('off')
    buf = io.BytesIO()
    plt.tight_layout(pad=0)
    plt.savefig(buf, format='png', facecolor=fig.get_facecolor(), bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')

@app.route('/')
def index():
    return render_template_string(HTML_PAGE, alpha=TAMIL_ALPHABET)

@app.route('/api/sample/<code>')
def get_sample(code):
    fp = f"data/demo_samples/{code.lower()}_sample.png"
    im = Image.open(fp).convert('RGB')
    tensor = tf(im).unsqueeze(0)
    with torch.no_grad():
        logits, attn = model(tensor, return_attn=True)
        probs = torch.softmax(logits, dim=1).numpy()[0]
    idx = int(np.argmax(probs))
    item = TAMIL_ALPHABET[idx]
    return jsonify({
        'tamil': item['tamil'],
        'name': item['name'],
        'translit': item['translit'],
        'desc': item['desc'],
        'confidence': float(probs[idx]),
        'in_b64': im_to_b64(im),
        'attn_b64': attn_to_b64(attn[0])
    })

@app.route('/api/predict', methods=['POST'])
def predict_upload():
    if 'image' not in request.files:
        return jsonify({'error': 'No file'}), 400
    f = request.files['image']
    im = Image.open(f).convert('RGB')
    tensor = tf(im).unsqueeze(0)
    with torch.no_grad():
        logits, attn = model(tensor, return_attn=True)
        probs = torch.softmax(logits, dim=1).numpy()[0]
    idx = int(np.argmax(probs))
    item = TAMIL_ALPHABET[idx]
    return jsonify({
        'tamil': item['tamil'],
        'name': item['name'],
        'translit': item['translit'],
        'desc': item['desc'],
        'confidence': float(probs[idx]),
        'in_b64': im_to_b64(im),
        'attn_b64': attn_to_b64(attn[0])
    })

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--predict', type=str, help='Image to predict')
    parser.add_argument('--port', type=int, default=5001)
    args = parser.parse_args()

    if args.predict:
        im = Image.open(args.predict).convert('RGB')
        tensor = tf(im).unsqueeze(0)
        with torch.no_grad():
            probs = torch.softmax(model(tensor), dim=1).numpy()[0]
        idx = int(np.argmax(probs))
        it = TAMIL_ALPHABET[idx]
        print(f"\nResult: {it['tamil']} ({it['name']}) | Transliteration: {it['translit']} | Confidence: {probs[idx]*100:.2f}%\n")
    else:
        print(f"\n* Tamil Sign ViT Dashboard online on http://127.0.0.1:{args.port}\n")
        app.run(host='0.0.0.0', port=args.port, debug=False)
