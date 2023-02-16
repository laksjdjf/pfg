#### This code is based on https://github.com/kohya-ss/sd-scripts/blob/main/networks/lora.py

import torch
import math
import os
import itertools

PREFIX_UNET = 'pfg'

    
#Network
class PFGNetwork(torch.nn.Module):
    def __init__(self, unet, input_size:int = 768, cross_attention_dim:int = 1024, num_tokens:int = 1) -> None:
        super().__init__()
        self.input_size = input_size
        self.cross_attention_dim = cross_attention_dim
        self.num_tokens = num_tokens
        
        #inputを入れる
        self.input = None
        
        #input_dim -> token_dim * num_tokens
        self.pfg_linear = torch.nn.Linear(self.input_size, cross_attention_dim * num_tokens)
        #unetのforward書き換え
        self._hook_forwards(unet)
        self.requires_grad_(True)
        
    #見づらいのでメソッドにしちゃう webuiのdaamを参考にしている。https://github.com/kousw/stable-diffusion-webui-daam
    def _hook_forwards(self, root_module: torch.nn.Module) -> list:
        count = 0
        for unet_block in itertools.chain(root_module.input_blocks, root_module.output_blocks, [root_module.middle_block]):
            # if 'CrossAttn' in unet_block.__class__.__name__:
            for module in unet_block.modules():
                if module.__class__.__name__ == "SpatialTransformer":
                    spatial_transformer = module
                    for basic_transformer_block in spatial_transformer.transformer_blocks:
                        basic_transformer_block.attn2.to_k.forward = self._hook_forward(basic_transformer_block.attn2.to_k)
                        basic_transformer_block.attn2.to_v.forward = self._hook_forward(basic_transformer_block.attn2.to_v)
                        count += 2
        print(f"create PFG for U-Net: {count} modules.")
    
    #forward書き換え
    def _hook_forward(self, org_module):
        prev_forward = org_module.forward
        def forward(x):
            #(b,1,dim) -> #(b,1,in_dim * tokens) 
            c = self.pfg_linear(self.input)
            
            #(b,1,dim) -> #(b,tokens,in_dim) 
            c = c.reshape(-1,self.num_tokens,self.cross_attention_dim)
            
            #生成時は違う場合が多い、その場合は(uncond+cond)*batchのため2分の1倍する。
            if x.shape[0] != c.shape[0]:
                c = c.repeat(x.shape[0] // 2,1,1)
                
            #concatenate (b,N,in_dim) + (b,tokens,in_dim)
            if self.input.shape[0] == x.shape[0]:
                x = torch.cat([x,c],dim = 1)
            else: #CFG対応：diffusers pipelineでは(uncond, cond)
                x_uncond, x_cond = x.chunk(2)
                x_uncond = torch.cat([x_uncond,x_uncond[:,-1:,:].repeat(1,self.num_tokens,1)],dim=1) #EOSをコピー（これでいいのか知らん）
                x_cond = torch.cat([x_cond,c],dim = 1) 
                x = torch.cat([x_uncond,x_cond])

            return prev_forward(x)
        return forward
    
    def prepare_optimizer_params(self, unet_lr):
        self.requires_grad_(True)
        param_data = {'params': self.parameters()}
        if unet_lr is not None:
            param_data['lr'] = unet_lr
        
        return [param_data]
            

    def save_weights(self, file, dtype=None):
        state_dict = self.state_dict()

        if dtype is not None:
            for key in list(state_dict.keys()):
                v = state_dict[key]
                v = v.detach().clone().to("cpu").to(dtype)
                state_dict[key] = v

        if os.path.splitext(file)[1] == '.safetensors':
            from safetensors.torch import save_file
            save_file(state_dict, file)
        else:
            torch.save(state_dict, file)
    
    def set_input(self,input_tensor:torch.Tensor) -> None:
        self.input = input_tensor
