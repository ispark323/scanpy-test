import gradio as gr
import shutil
import os


def upload_file(files):
    UPLOAD_FOLDER = './data'
    if not os.path.exists(UPLOAD_FOLDER):
        os.mkdir(UPLOAD_FOLDER)

    file_paths = [file.name for file in files]
    for file in files:
        shutil.copy(file, UPLOAD_FOLDER)
    
    return file_paths

with gr.Blocks() as demo:
    file_output = gr.File()
    upload_button = gr.UploadButton("Click to Upload a File", file_count="multiple")
    upload_button.upload(upload_file, upload_button, file_output)

demo.launch()