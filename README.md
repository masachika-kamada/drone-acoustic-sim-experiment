# PyRoomAcoustics Trial

## Quick Start

```shell
git clone [REPOSITORY_URL]
cd pyroomacostics-trial
pip install -r requirements.txt
```

## ディレクトリ構造

```
.
├── archive
├── data
│   ├── processed
│   ├── raw
│   └── simulation
├── experiments
│   ├── 001
│   └── ...
├── lib
│   └── doa
├── notebook
├── src
└── tests
```

* `archive`: 過去のバージョンや使用しなくなったコードを保存
* `data`: データ関連のフォルダ
  * `processed`: 前処理済みのデータを保存
  * `raw`: オリジナルの生データを保存
  * `simulation`: シミュレーション結果等を保存
* `experiments`: 各実験の設定と結果を保存
* `lib`: カスタマイズした外部ライブラリを保存
  * `doa`: カスタマイズした DOA 関連のコード
* `notebook`: Jupyter notebook 等、データ解析やモデルの試行錯誤に使うノートブックを保存
* `src`: プロジェクトで使用するモジュールやパッケージを配置
* `tests`: テストコードを配置
