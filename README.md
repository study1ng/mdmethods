# mdmethods

3次元医用画像を対象とした、深層学習手法の実験用リポジトリです。PyTorch、Lightning、MONAIなどを利用し、自己教師あり学習、セグメンテーション、転移学習に関する学習・推論・前処理を扱います。

多くの処理ではNVIDIA GPUと、Gitには含まれないローカルデータセットが必要です。

## 主な構成

```text
.
├── main.py                 # コマンドラインエントリーポイント
├── experiments/            # 実験、データ処理、モデル、共通処理
│   ├── nets/               # 再利用可能なネットワーク構成要素
│   ├── utils/              # ファイル操作などの共通機能
│   └── <method>/
│       └── experiments/    # 手法ごとの名前付き実験バリアント
├── plans/                  # 実験で使用するJSON形式のプラン
├── data/                   # ローカルデータ・生成物（Git管理外）
├── .container/             # Dockerによる開発環境
├── pyproject.toml          # Pythonプロジェクトと依存関係の定義
└── uv.lock                 # uvのロックファイル
```

## 必要な環境

- NVIDIA GPU
- NVIDIA Container Toolkitを利用できるDocker環境
- Docker Compose

コンテナを使わない場合はPython 3.12以降とCUDA 12.1に対応した環境が必要です。依存関係は`pyproject.toml`と`uv.lock`で管理されています。特にPyTorchはCUDA 12.1向けパッケージに固定されています。

## 開発環境の再現

`.container/Dockerfile`と`.container/docker-compose.yml`に開発環境の定義があります。コンテナにはPython 3.12、CUDA 12.1.1、uv、およびプロジェクトの依存関係がインストールされます。`causal-conv1d`と`mamba-ssm`は`uv sync`の後にソースから追加インストールされます。

### 1. 環境変数の設定

`.container/.env`に、使用する環境に合わせた値を設定します。

```dotenv
UID=1000
GID=1000
USER_NAME=<ユーザー名>
DATA_DIR=<データディレクトリの絶対パス（先頭の / を除く）>
CONTEXT=<このリポジトリの絶対パス>
TMP=<コンテナ内で使用する一時ディレクトリ>
```

`docker-compose.yml`では`DATA_DIR`の先頭に`/`を付けてマウントするため、例えばホスト側のデータが`/mnt/data`にある場合は`DATA_DIR=mnt/data`とします。ホストの`HOME`もコンテナ内の`/homes/<ユーザー名>`へマウントされます。

`.env`には環境固有のパスが含まれるため、リポジトリへコミットしないでください。

### 2. イメージのビルド

リポジトリのルートで実行します。

```bash
docker compose --env-file .container/.env \
  -f .container/docker-compose.yml build
```

### 3. コンテナの起動

```bash
docker compose --env-file .container/.env \
  -f .container/docker-compose.yml run --rm app bash
```

コンテナ内の作業ディレクトリは`/homes/<ユーザー名>/working/mdmethods`、Python仮想環境は`/opt/venv`です。仮想環境へのパスはあらかじめ設定されています。

## コマンドラインの使い方

基本形式は次のとおりです。

```bash
uv run python main.py <lib> <method> \
  [--experiment_name <実験名>] [手法固有の引数]
```

`<lib>`には`experiments/`直下の手法名を指定します。`--experiment_name`を指定すると、`experiments/<lib>/experiments/<実験名>.py`が読み込まれます。

`<method>`に指定できる処理は次の4種類です。

| method | 内容 |
| --- | --- |
| `analyze` | 入力データを解析し、前処理やプラン作成に必要な情報を生成します |
| `prune` | 解析結果とプランに基づいて対象データを選別します |
| `train` | モデルを学習します |
| `inference` | 学習済みチェックポイントで推論します |

モジュールによっては一部の処理を実装していません。利用可能な引数は対象モジュールのヘルプで確認してください。

```bash
# main.py側の引数を表示
uv run python main.py --help-main

# 例: mimの学習で利用できる引数を表示
uv run python main.py mim train --module-help

# 例: 名前付き実験バリアントの引数を表示
uv run python main.py munet train \
  --experiment_name overlap_0 --module-help
```

### 実行例

以下はコマンド形式の例です。実行前に対象モジュールの引数、入力パス、出力先を確認してください。

```bash
# データ解析
uv run python main.py mim analyze \
  <入力データ> <解析結果の保存先>

# 学習
uv run python main.py mim train \
  <前処理済みデータ> <出力先> <プランJSON>

# 名前付き実験による学習
uv run python main.py munet train \
  --experiment_name overlap_0 \
  <前処理済みデータ> <出力先> <プランJSON> \
  --pretrained_path <チェックポイント>

# 推論
uv run python main.py pretrained_seg inference \
  <前処理済みデータ> <出力先> <チェックポイント> <プランJSON>
```

学習ではデフォルトでGPUとbf16混合精度を使用します。出力先にはチェックポイント、CSVログ、TensorBoardログなどが作成され、MLflowはSQLiteデータベースを使用します。

## データと生成物

`data/`はGit管理外です。データセット、医用画像、前処理結果、チェックポイント、推論結果、ログ、SQLiteデータベース、生成レポートなどをコミットしないでください。

学習、推論、解析、前処理、pruneは、大量の計算資源やディスク容量を使用する場合があります。特に`prune`は対象パスと保存先の間にシンボリックリンクを作成するため、実行前にパスの対応と既存ファイルの有無を確認してください。

医用画像や患者情報を、公開ログや外部サービスへ送信しないでください。

## 開発時の確認

このリポジトリには、リポジトリ全体で統一されたテスト、リンター、フォーマッターの設定はありません。Pythonファイルを変更した場合は、まず対象ファイルの構文を確認してください。

```bash
uv run python -m py_compile path/to/changed_file.py
```

モデルやテンソル処理の変更には、可能であれば実データの代わりに小さな合成テンソルを使ったCPU上の確認を追加してください。完全な学習や推論は、必要なデータ、GPU、所要時間、出力先が確認できた場合にのみ実行してください。

## 注意事項

- `pyproject.toml`と`uv.lock`を依存関係の正として扱ってください。
- `requirements.txt`はuv環境の完全な代替ではありません。
- モデル構造やstate dictのキーを変更すると、既存チェックポイントとの互換性が失われる可能性があります。
- 本リポジトリでは主に3次元テンソルを扱います。チャンネル軸と空間軸の順序を維持してください。
