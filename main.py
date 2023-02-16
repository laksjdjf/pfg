import argparse
import torch
from torch.utils.data import DataLoader
import os
from tqdm import tqdm
import numpy as np
import time
import copy

from diffusers import AutoencoderKL, UNet2DConditionModel, DDPMScheduler, StableDiffusionPipeline ,DDIMScheduler
from diffusers.optimization import get_scheduler

from transformers import CLIPTextModel, CLIPTokenizer

from utils.dataset import AspectDataset
from pfg.pfg import PFGNetwork


#検証画像用のネガティブプロンプト
NEGATIVE_PROMPT = "worst quality, low quality, medium quality, deleted, lowres, comic, bad anatomy,bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, jpeg artifacts, signature, watermark, username, blurry"

NUMBER_OF_IMAGE_LOGS = 4

###コマンドライン引数#########################################################################
parser = argparse.ArgumentParser(description='StableDiffusionの訓練コード')
parser.add_argument('--model', type=str, required=True, help='学習済みモデルパス（diffusers）')
parser.add_argument('--dataset', type=str, required=True, help='データセットのパス')
parser.add_argument('--output', type=str, required=True, help='学習チェックポイントの出力先')
parser.add_argument('--image_log', type=str, required=True, help='検証画像の出力先')
parser.add_argument('--resolution', type=str, default="512,512", help='検証画像のサイズ。"幅,高さ"で選択、もしくは"長さ"で正方形になる')
parser.add_argument('--batch_size', type=int, default=4, help='バッチサイズ')
parser.add_argument('--lr', type=float, default="5e-6", help='学習率、"学習率')
parser.add_argument('--lr_scheduler', type=str,default = "constant", help='学習率スケジューラー',choices=["cosine", "linear", "constant"])
parser.add_argument('--epochs', type=int, default=10, help='エポック数')
parser.add_argument('--save_n_epochs', type=int, default=5, help='何エポックごとにセーブするか')
parser.add_argument('--amp', action='store_true', help='AMPを利用する')
parser.add_argument('--gradient_checkpointing', action='store_true', help='勾配チェックポイントを利用する（VRAM減計算時間増）')
parser.add_argument('--wandb', action='store_true', help='wandbによるログ管理')
parser.add_argument('--v_prediction', action='store_true', help='SDv2系（-baseではない）を使う場合に指定する')
parser.add_argument('--pfg_input_size', type=int, default=768, help='pfgの埋め込みベクトルサイズ wdtaggerは768')
parser.add_argument('--pfg_cross_attention_dim', type=int, default=768, help='v1は768, v2は1024')
parser.add_argument('--pfg_num_tokens', type=int, default=5, help='pfgのトークン数')
parser.add_argument('--pfg_prompt', type=str, default="illustration of 1girl", help='pfgの共通プロンプト')
parser.add_argument('--resume_pfg', type=str,default = None, help='pfgのresume パス')
parser.add_argument('--train_unet', action='store_true', help='UNetごと学習する')
############################################################################################

def collate_fn(x):
    return x[0]

#メイン処理
def main(args):
    ###学習準備##############################################################################
    #image log pathを作る。
    if not os.path.exists(args.image_log):
        os.makedirs(args.image_log)    
    
    #画像サイズを定義
    size = args.resolution.split(",") 
    size = (int(size[0]),int(size[-1])) #長さが1の場合同じ値になる
    
    #device,dtypeを定義
    device = torch.device('cuda')
    weight_dtype = torch.float16 if args.amp else torch.float32
    
    #モデルのロード、勾配無効や推論モードに移行
    tokenizer = CLIPTokenizer.from_pretrained(args.model, subfolder='tokenizer')
    
    text_encoder = CLIPTextModel.from_pretrained(args.model, subfolder='text_encoder')
    text_encoder.requires_grad_(False)
    text_encoder.eval()
    
    vae = AutoencoderKL.from_pretrained(args.model, subfolder='vae')
    vae.requires_grad_(False)
    vae.eval()

    unet = UNet2DConditionModel.from_pretrained(args.model, subfolder='unet')
    unet.requires_grad_(args.train_unet)
    
    try:
        unet.set_use_memory_efficient_attention_xformers(True)
        print("apply xformers for unet !!!")
    except:
        print("cant apply xformers. using normal unet !!!")
    
    #AMP用のスケーラー
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)
    
    #pfgの準備
    network = PFGNetwork(unet, args.pfg_input_size, args.pfg_cross_attention_dim, args.pfg_num_tokens)
    if args.resume_pfg is not None:
        network.load_state_dict(torch.load(args.resume_pfg))
    params = network.prepare_optimizer_params(args.lr)
        
    #パラメータ
    if args.train_unet:
        params.append({'params':unet.parameters(),'lr':args.lr})
        
    #最適化関数
    try:
        import bitsandbytes as bnb
        optimizer_cls = bnb.optim.AdamW8bit
        print('apply AdamW8bit optimizer !')
    except:
        print('cant import bitsandbytes, using regular Adam optimizer !')
        optimizer_cls = torch.optim.AdamW
    
    #最適化関数にパラメータを入れる
    optimizer = optimizer_cls(params)
    
    #勾配チェックポイントによるVRAM削減（計算時間増）
    if args.gradient_checkpointing:
        unet.conv_in.requires_grad_(True)
        unet.enable_gradient_checkpointing()
    
    #型の指定とGPUへの移動
    text_encoder.to(device,dtype=weight_dtype)
    vae.to(device,dtype=weight_dtype)
    unet.to(device,dtype=torch.float32)
    network.to(device,dtype=torch.float32)
    
    #ノイズスケジューラー
    noise_scheduler = DDPMScheduler.from_pretrained(
        args.model,
        subfolder='scheduler',
    )
    
    #sampling stepの範囲を指定
    step_range = (0, noise_scheduler.num_train_timesteps)
    
    #データローダー num_workersは適当。
    dataset = AspectDataset(args.dataset,tokenizer = tokenizer,batch_size = args.batch_size,mask = False ,controll = True, prompt = args.pfg_prompt) #batch sizeはデータセット側で処理する
    dataloader = DataLoader(dataset,batch_size=1,num_workers=2,shuffle=False,collate_fn = collate_fn) #shuffleはdataset側で処理する、Falseが必須。
    
    #トータルステップ
    total_steps = args.epochs * len(dataloader)
    
    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps= int(0.05 * total_steps), #そこまで重要に思えないから0.05固定
        num_training_steps= total_steps
    )

    #wandb
    if args.wandb:
        import wandb
        run = wandb.init(project="sd-pfg", name=args.output,dir=os.path.join(args.output,'wandb'))
    
    #全ステップ
    global_step = 0
    
    #プログレスバー
    progress_bar = tqdm(range(total_steps), desc="Total Steps", leave=False)
    loss_ema = None #訓練ロスの指数平均
    
    #学習ループ
    for epoch in range(args.epochs):
        for batch in dataloader:
            #時間計測
            b_start = time.perf_counter()
            
            #テキスト埋め込みベクトル
            tokens = tokenizer(batch["caption"], max_length=tokenizer.model_max_length, padding=True, truncation=True, return_tensors='pt').input_ids.to(device)
            encoder_hidden_states = text_encoder(tokens, output_hidden_states=True).last_hidden_state.to(device)
            
            #潜在変数
            latents = batch['latents'].to(device) * 0.18215 #bucketを使う場合はあらかじめlatentを計算している
            
            #すぺるみす？しらん
            controlls = batch["controll"].to(device)
            network.set_input(controlls)
                
            #ノイズを生成
            noise = torch.randn_like(latents)
            bsz = latents.shape[0]
            #画像ごとにstep数を決める
            timesteps = torch.randint(step_range[0], step_range[1], (bsz,), device=latents.device)
            timesteps = timesteps.long()

            #steps数に応じてノイズを付与する
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
            
            #推定ノイズ
            with torch.autocast("cuda",enabled=args.amp):
                noise_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample
            
            if args.v_prediction:
                noise = noise_scheduler.get_velocity(latents, noise, timesteps)
                    
            loss = torch.nn.functional.mse_loss(noise_pred.float(), noise.float(), reduction="mean")
            if loss_ema is None:
                loss_ema = loss.item()
            else:
                loss_ema = loss_ema * 0.9 + loss.item() * 0.1
            
            #混合精度学習の場合アンダーフローを防ぐために自動でスケーリングしてくれるらしい。
            scaler.scale(loss).backward()
            #勾配降下
            scaler.step(optimizer)
            scaler.update()
        
            lr_scheduler.step()
            
            #勾配リセット
            optimizer.zero_grad()
            
            #ステップ更新
            global_step += 1
            
            #時間計測
            b_end = time.perf_counter()
            time_per_steps = b_end - b_start
            samples_per_time = bsz / time_per_steps
            
            #プログレスバー更新
            logs={"loss":loss_ema,"samples_per_second":samples_per_time,"lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.update(1)
            progress_bar.set_postfix(logs)
            
            #wandbのlog更新
            if args.wandb:    
                run.log(logs, step=global_step)
        
        #モデルのセーブと検証画像生成
        print(f'{epoch+1} epoch 目が終わりました。訓練lossは{loss_ema}です。')
        if (epoch + 1) % args.save_n_epochs == 0:
            print(f'チェックポイントをセーブするよ!')
            pipeline = StableDiffusionPipeline.from_pretrained(
                    args.model,
                    text_encoder=text_encoder,
                    vae=vae,
                    unet=unet,
                    tokenizer=tokenizer,
                    scheduler=DDIMScheduler.from_pretrained(args.model, subfolder="scheduler"),
                    feature_extractor = None,
                    safety_checker = None
            )
            
            #検証画像生成
            with torch.autocast('cuda', enabled=args.amp):
                num = min(bsz,NUMBER_OF_IMAGE_LOGS) #基本4枚だがバッチサイズ次第
                images = []
                for i in range(num):
                    prompt = batch["caption"][i]
                    controlls = batch["controll"][i].unsqueeze(0).to(device)
                    network.set_input(controlls)
                    image = pipeline(prompt,width=size[0],height=size[1],negative_prompt=NEGATIVE_PROMPT).images[0]
                    if args.wandb:    
                        images.append(wandb.Image(image,caption=prompt))
                    else:
                        images.append(image)
            
            if args.wandb:
                run.log({'images': images}, step=global_step)
            else:
                [image.save(os.path.join(args.image_log,f'image_log_epoch_{str(epoch).zfill(3)}_{i}.png')) for i,image in enumerate(images)]
            
            network.save_weights(f'{args.output}.pt')
            if args.train_unet:
                #output pathをつくる。
                if not os.path.exists(args.output):
                    os.makedirs(args.output)
                pipeline.save_pretrained(f'{args.output}')
                
            del pipeline
        
        
if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
