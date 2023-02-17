import cv2
import numpy as np
from PIL import Image

import tensorflow as tf
import torch

import argparse

###コマンドライン引数#########################################################################
parser = argparse.ArgumentParser(description='生成')
parser.add_argument('--model_id', '-m', type=str, required=True, help='モデルパス（diffusers）')
parser.add_argument('--pfg_path', '-p', type=str, required=True, help='pfgパス')
parser.add_argument('--input_size', '-i', type=int, default = 768, help='分類器の出力、wdtaggerは768')
parser.add_argument('--cross_attention_dim', '-c', type=int, default = 1024, help='v1は768, v2は1024')
parser.add_argument('--num_tokens', '-n', type=int, default= 10, help='pfgトークン数')
parser.add_argument('--share', '-s', action="store_true", help='gradio share機能を使う')
args = parser.parse_args()
############################################################################################

model_id = args.model_id
pfg_path = args.pfg_path
input_size = args.input_size
cross_attention_dim = args.cross_attention_dim
num_tokens = args.num_tokens


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
from diffusers import DiffusionPipeline
device = "cuda"

pipe = DiffusionPipeline.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    custom_pipeline="lpw_stable_diffusion", #なんか知らんがwebui風のプロンプト強調()[]ができるらすい
).to(device)

try:
    pipe.unet.set_use_memory_efficient_attention_xformers(True)
    print("apply xformers for unet !!!")
except:
    print("cant apply xformers. using normal unet !!!")

from pfg.pfg import PFGNetwork
pfg = PFGNetwork(pipe.unet, input_size, cross_attention_dim, num_tokens)
pfg.load_state_dict(torch.load(pfg_path))
pfg.requires_grad_(False)
pfg.to(device, dtype = torch.float16)
print("pfg loaded")

import gradio as gr
from datetime import datetime


def generate(image,prompt,negative_prompt,batch_size,image_scale,width,height,steps,guidance_scale):
    hidden_states = infer(image).to(device)
    pfg.set_input(hidden_states * image_scale)
    with torch.autocast("cuda"):
        images = pipe(prompt = [prompt]*batch_size,
                      width = width,
                      height = height,
                      guidance_scale=guidance_scale,
                      num_inference_steps=steps,
                      negative_prompt = [negative_prompt]*batch_size,
                      ).images
    return images

demo = gr.Interface(
    generate,
    [gr.Image(type="pil"),
     gr.Textbox(value="illustration of"),
     gr.Textbox(value="worst quality, low quality, medium quality, deleted, lowres, comic, bad anatomy,bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, jpeg artifacts, signature, watermark, username, blurry"),
     gr.Slider(1,32,value=1,step=1),
     gr.Slider(0,2,value=1,step=0.01,label="pfgの強弱：ちょっとあげるだけでいい結果にならない"),
     gr.Slider(256,896,value=640,step=64),
     gr.Slider(256,896,value=896,step=64),
     gr.Slider(0,50,value=50,step=1),
     gr.Slider(0,30,value=12,step=0.5)],
    [gr.Gallery(label='Output', show_label=False, elem_id="gallery").style(grid=[2], height="auto")],
    title = "タイトル",
    description = "説明",
    allow_flagging='never'
)

demo.launch(share = args.share)
