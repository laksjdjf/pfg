from pfg.prg_compvis import PFGNetwork
import preprocess.dbimutils

#PFG for webui

import torch
import numpy as np

import modules.scripts as scripts
import gradio as gr

import modules.processing
from modules.processing import StableDiffusionProcessing
from tensorflow.keras.models import load_model, Model

class OrgDDIMSampler(ddim.DDIMSampler):
    pass

class Script(scripts.Script):
    def __init__(self) -> None:
        super().__init__()
        self.pfg_path = None
        
        #なんもいみわかっとらんけどこれしないとVRAMくわれる。対応するバージョンもよくわからない
        physical_devices = tf.config.list_physical_devices('GPU')
        if len(physical_devices) > 0:
            for device in physical_devices:
                tf.config.experimental.set_memory_growth(device, True)
                print('{} memory growth: {}'.format(device, tf.config.experimental.get_memory_growth(device)))
        else:
            print("Not enough GPU hardware devices available")
        self.tagger_model = None

    def title(self):
        return "PFG for webui"

    def ui(self, is_img2img):
        with gr.Row():
            gr.Markdown(
                """
                Prompt Free Generation
                """)
        with gr.Row():
            enable_pfg = gr.Checkbox(label='Enable PFG', value=False)
        with gr.Row():
            tagger_path = gr.Textbox(label="wd14taggerパス")
            pfg_path = gr.Textbox(label="pfg重みパス")
        with gr.Row():
            input_size = gr.Textbox(value=768, label="wdtaggerは768固定なので今は設定不要")
            cross_attention_dim = gr.Textbox(value=1024, label="v1は768, v2は1024")
            num_tokens = gr.Textbox(value=5, label="トークン数")
        with gr.Row():
            scale = gr.Slider(0, 3, value=1.0,step=0.1,label='image scale (pfgの強さ)')
        with gr.Row():
            image  = gr.Image(type="pil")
        return [enable_pfg, tagger_path, pfg_path, input_size, cross_attention_dim, num_tokens, scale, image]

    def run(self, p: StableDiffusionProcessing, enable_pfg, tagger_path, pfg_path, input_size, cross_attention_dim, num_tokens, scale, image):
        if enable_pfg:
            if self.pfg_path is None:
                unet = p.sd_model.model.diffusion_model
                pfg = PFGNetwork(model.model.diffusion_model, input_size, cross_attention_dim, num_tokens)
                pfg.load_state_dict(torch.load(pfg_path))
                pfg.requires_grad_(False)
                pfg.cuda()
                self.pfg_path = pfg_path
       
            if self.tagger_model is None:
                self.tagger_model = load_model(tagger_path)
                self.tagger_model = Model(tagger_model.layers[0].input, tagger_model.layers[-3].output) #最終層手前のプーリング層の出力を使う
    
            hidden_states = infer(image).to(device)
            pfg.set_input(hidden_states * scale)
            result = modules.processing.process_images(p)
            #pfg.reset() とすべきだがめんどくさくなってきた。
            
        else:
            result = modules.processing.process_images(p)
            
        return result
      
  def infer(self, img:Image):
      img = dbimutils.smart_imread_pil(img)
      img = dbimutils.smart_24bit(img)
      img = dbimutils.make_square(img, 448)
      img = dbimutils.smart_resize(img, 448)
      img = img.astype(np.float32)
      probs = self.tagger_model(np.array([img]), training=False)
      return torch.tensor(probs.numpy()).unsqueeze(0)
