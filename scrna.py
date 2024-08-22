import gradio as gr
import os
import shutil
import requests
import torch
from timm import create_model
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform


IMAGENET_1k_URL = "https://storage.googleapis.com/bit_models/ilsvrc2012_wordnet_lemmas.txt"
LABELS = requests.get(IMAGENET_1k_URL).text.strip().split('\n')

model = create_model('resnet50', pretrained=True)

transform = create_transform(
    **resolve_data_config({}, model=model)
)
model.eval()


def upload_file(files):
    UPLOAD_FOLDER = './data'
    if not os.path.exists(UPLOAD_FOLDER):
        os.mkdir(UPLOAD_FOLDER)

    file_paths = [file.name for file in files]
    for file in files:
        shutil.copy(file, UPLOAD_FOLDER)

    return file_paths


def predict_fn(img):
    img = img.convert('RGB')
    img = transform(img).unsqueeze(0)

    with torch.no_grad():
        out = model(img)

    probabilites = torch.nn.functional.softmax(out[0], dim=0)

    values, indices = torch.topk(probabilites, k=5)

    return {LABELS[i]: v.item() for i, v in zip(indices, values)}

def analyze_fn(input):
    return input

# gr.Interface(predict_fn, gr.components.Image(type='pil'), outputs='label').launch()
# gr.Interface(predict_fn, upload_file, outputs='label').launch()

with gr.Blocks() as block:
    label = gr.Label()
    file_output = gr.File()

    upload_button = gr.UploadButton("Click to Upload a File", file_count="multiple")
    file_list = upload_button.upload(upload_file, upload_button, file_output)

    analyze_button = gr.Button("Analyze")

    analyze_button.click(analyze_fn, inputs=file_list, outputs=label)

block.launch()


