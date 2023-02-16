import cv2
import numpy as np
from PIL import Image

import tensorflow as tf
import torch

#設定
model_id = "tintin-diffusion"
pfg_path = "manman-pfg.pt"
input_size = 768
cross_attention_dim = 1024
num_tokens = 5


#なんもいみわかっとらんけどこれしないとVRAMくわれる。対応するバージョンもよくわからない
physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    for device in physical_devices:
        tf.config.experimental.set_memory_growth(device, True)
        print('{} memory growth: {}'.format(device, tf.config.experimental.get_memory_growth(device)))
else:
    print("Not enough GPU hardware devices available")

from tensorflow.keras.models import load_model, Model
tagger_model = load_model("wd-v1-4-vit-tagger-v2")
tagger_model = Model(tagger_model.layers[0].input, tagger_model.layers[-3].output) #最終層手前のプーリング層の出力を使う

from preprocess import dbimutils
def infer(img:Image):
    img = dbimutils.smart_imread_pil(img)
    img = dbimutils.smart_24bit(img)
    img = dbimutils.make_square(img, 448)
    img = dbimutils.smart_resize(img, 448)
    img = img.astype(np.float32)
    probs = tagger_model(np.array([img]), training=False)
    return torch.tensor(probs.numpy()).unsqueeze(0)
  
#Stable diffusion モデルのロード
from diffusers import StableDiffusionPipeline
device = "cuda"

pipe = StableDiffusionPipeline.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
).to(device)

from pfg.pfg import PFGNetwork
pfg = PFGNetwork(pipe.unet, input_size, cross_attention_dim, num_tokens)
pfg.load_state_dict(torch.load(pfg_path))
pfg.requires_grad_(False)
pfg.to(device, dtype = torch.float16)
print("loaded")

import gradio as gr
from datetime import datetime


def generate(image,prompt,negative_prompt,image_scale,width,height,steps,guidance_scale):
    hidden_states = infer(image).to(device)
    pfg.set_input(hidden_states * image_scale)
    with torch.autocast("cuda"):
        images = pipe(prompt = prompt,
                      width = width,
                      height = height,
                      guidance_scale=guidance_scale,
                      num_inference_steps=steps,
                      negative_prompt = negative_prompt,
                      ).images
    return images[0]

demo = gr.Interface(
    generate,
    [gr.Image(type="pil"),
     gr.Textbox(value="illustration of"),
     gr.Textbox(value="worst quality, low quality, medium quality, deleted, lowres, comic, bad anatomy,bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, jpeg artifacts, signature, watermark, username, blurry"),
     gr.Slider(0,5,value=1,step=0.05),
     gr.Slider(256,896,value=640,step=64),
     gr.Slider(256,896,value=896,step=64),
     gr.Slider(0,50,value=50,step=1),
     gr.Slider(0,20,value=7,step=0.5)],
    ["image"],
    title = "タイトル",
    description = "説明",
    allow_flagging='never'
)

demo.launch(share=True)
