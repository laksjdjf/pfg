# Prompt Free Generation
　ガイド画像とWD14Taggerを利用して、プロンプトなしで好きなキャラとかを学習・生成する手法です。学習と生成のコードを置いておきます。
このリポジトリは自分の訓練コードからPFG機能だけに限定して実装したものです。ほとんどテストしていないですが、理論だけ貼ってはい終わりというのはあれなので・・・。

学習済みモデル：https://huggingface.co/furusu/PFG

https://github.com/laksjdjf/sd-trainer の方が訓練はしやすいです。

# 使い方
[wd-v1-4-vit-tagger-v2](https://huggingface.co/SmilingWolf/wd-v1-4-vit-tagger-v2)を使いますので、ダウンロードしてください。直下に直接入れるなりシンボリックリンクしてください。tensorflowのことは何にも分からないので、依存関係も分かりません。tensorflow==2.9.1では動いているようです。

## 訓練
学習用画像に対して、bucketing.pyでばけってぃんぐして、latent.pyでれいてんとにして、tagger_control.pyでwd14taggerの埋め込みをげっちゅします。datasetディレクトリにlatent(hoge.npy)とtaggerのemb(hoge.npz)とメタデータ(buckets.json)があればおっけー。キャプションデータはいらないです。Image is All You Need（？）

```
python3 preprocess/bucketing.py -d <image_directory> -o <dataset_directory> --resolutionとかなんとか 任意
python3 preprocess/latent.py -d <dataset_directory> -o <dataset_directory> -m "<diffusers_model>"
python3 preprocess/tagger_control.py -d <dataset_directory> -o <dataset_directory>
```

詳しくは``` python hoge.py -h ```で・・・あんまり詳しくないかも。

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

--pfg_cross_attention_dimはv1は768、v2は1024です。またv_predictionもでるのときは--v_predictionしてください。

学習設定等は全然調査できてません。学習率が特に分からない。

## 生成
```
python generate.py -m <diffusers_path> -p <pfg_path> -c <768 or 1024> -n <num_tokens>
```

-sでgradioのshare=Trueになります。

# 既知の問題
+ gradient_checkpointingがうまく機能していないかも。

# 引用りぽ
訓練コード全体：https://github.com/harubaru/waifu-diffusion

PFGNetworkの定義やPreprocessコード等：https://github.com/kohya-ss/sd-scripts

WD14Tagger周り：https://github.com/toriato/stable-diffusion-webui-wd14-tagger

compvis版のモジュール書き換え：https://github.com/kousw/stable-diffusion-webui-daam

# 記事
https://note.com/gcem156/n/ne334e7be9eb7

