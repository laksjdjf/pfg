# Prompt Free Generation
　ガイド画像とWD14Taggerを利用して、プロンプトなしで好きなキャラとかを学習・生成する手法です。学習と生成のコードを置いておきます。
このリポジトリは自分の訓練コードにPFG機能だけに限定して実装したものです。なんのテストもしていないですが、理論だけ貼ってはい終わりというのはあれなので・・・。60万枚くらいのデータで試そうと思っているので、うまくいったらモデルを公開します。

# Usage
[wd-v1-4-vit-tagger-v2](https://huggingface.co/SmilingWolf/wd-v1-4-vit-tagger-v2)を使いますので、ダウンロードしてください。直下に直接入れるなりシンボリックリンクしてください。

## Train
bucketing.pyでばけってぃんぐして、latent.pyでれいてんとにして、tagger_control.pyでwd14taggerの埋め込みをげっちゅします。datasetディレクトリにlatent(hoge.npy)とtaggerのemb(hoge.npz)とメタデータ(buckets.json)があればおっけー。

訓練はこんな感じです。wandbを使うにはwandbのあぴきーが必要です。
```
python3 main.py \
--model "modemode" \
--dataset "detadeta" \
--output "outout" \
--image_log "imageimage" \
--resolution "640,896" \
--batch_size 4096 \
--lr "1e-5" \
--lr_scheduler "constant" \
--epochs 156 \
--save_n_epochs 1 \
--amp \
--pfg_input_size 768 \
--pfg_cross_attention_dim 1024 \
--pfg_num_tokens 5 \
--pfg_prompt "illustration of *" \
--wandb
```

--train_unetでたぶんUNetごと学習できます。そのままだとPFGLinearの重みファイルのみが生えてきます。

## Generate
generate.pyの上のほうの5項目をうまくかえて起動してください。

# 引用りぽ
訓練コード全体：https://github.com/harubaru/waifu-diffusion

PFGNetworkの定義やPreprocessコード等：https://github.com/kohya-ss/sd-scripts

WD14Tagger周り：https://github.com/toriato/stable-diffusion-webui-wd14-tagger

compvis版のモジュール書き換え：https://github.com/kousw/stable-diffusion-webui-daam
