# Prompt Free Generation
　ガイド画像とWD14Taggerを利用して、プロンプトなしで好きなキャラとかを学習・生成する手法です。学習と生成のコードを置いておきます。
このリポジトリは自分の訓練コードにPFG機能だけに限定して実装したものです。なんのテストもしていないですが、理論だけ貼ってはい終わりというのはあれなので・・・。60万枚くらいのデータで試そうと思っているので、うまくいったらモデルを公開します。

# 引用りぽ
訓練コード全体：https://github.com/harubaru/waifu-diffusion

PFGNetworkの定義やPreprocessコード等：https://github.com/kohya-ss/sd-scripts

WD14Tagger周り：https://github.com/toriato/stable-diffusion-webui-wd14-tagger

compvis版のモジュール書き換え：https://github.com/kousw/stable-diffusion-webui-daam
