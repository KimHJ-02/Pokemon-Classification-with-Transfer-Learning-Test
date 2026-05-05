import streamlit as st
from PIL import Image
import torch
from models import get_model
from torchvision import transforms
import os
from dataset import load_class_names

MODELS = ['resnet50', 'mobilenet_v2', 'efficientnet_b0', 'simplecnn']
DATA_DIR = 'data'
CHECKPOINT_DIR = 'checkpoints'
DEVICE = 'cpu'

st.title('Pokemon Classifier Demo')
st.caption('업로드한 이미지를 4개 모델이 각각 예측합니다.')


def _build_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])


def _predict_top1(model_name, image_tensor, class_names):
    ckpt = os.path.join(CHECKPOINT_DIR, f'{model_name}_best.pth')
    if not os.path.exists(ckpt):
        return f'{model_name}: checkpoint 없음 ({ckpt})'

    model = get_model(model_name, len(class_names), pretrained=False)
    model.load_state_dict(torch.load(ckpt, map_location=DEVICE))
    model.eval()

    with torch.no_grad():
        logits = model(image_tensor)
        probs = torch.softmax(logits, dim=1)
        conf, pred = probs.max(1)
    label = class_names[int(pred.item())]
    return f'{model_name}: {label} ({float(conf.item()):.2%})'


uploaded = st.file_uploader('Upload an image', type=['png', 'jpg', 'jpeg'])

if uploaded is not None:
    img = Image.open(uploaded).convert('RGB')
    st.image(img, caption='uploaded', width="stretch")
    class_names = load_class_names(DATA_DIR)
    if not class_names:
        st.error('class 정보가 없습니다. data/train/<class> 구조를 확인하세요.')
    else:
        x = _build_transform()(img).unsqueeze(0)
        st.subheader('모델별 예측 결과')
        for model_name in MODELS:
            st.write(_predict_top1(model_name, x, class_names))
