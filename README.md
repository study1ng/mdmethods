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

`<method>`に指定できる処理は次の5種類です。

| method | 内容 |
| --- | --- |
| `analyze` | 入力データを解析し、前処理やプラン作成に必要な情報を生成します |
| `prune` | 解析結果とプランに基づいて対象データを選別します |
| `train` | モデルを学習します |
| `inference` | 学習済みチェックポイントで推論します |
| `custom` | 各実験独自の処理を実行します |

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


### MUNet のチェックポイントからの推論

コンテナ内のプロジェクトルートから実行します。

```bash
uv run python main.py munet inference \
  <画像ディレクトリ> <出力ルート> <学習時のプランJSON> \
  --ckpt <MUNetのチェックポイント.ckpt> --devices 0
```

入力は `.nii.gz` が直接置かれたディレクトリです。正解ラベルは不要で、
読み込み時に plan に基づく前処理を行います。チェックポイントの UNet と
MUNet ボトルネックの両方を復元し、学習時のパッチ重複率・位置エンコーディング設定で推論します。
plan のパッチサイズ・spacing・正規化統計・ネットワーク構成が学習時と異なる場合はエラーになります。

`--experiment_name overlap_0` などを追加すると、保存先に実験名が入ります。
どのバリアントでもモデル設定はチェックポイントから復元し、実験名では上書きしません。
出力は `<出力ルート>/munet/[実験名/]<日時>/` に保存され、
予測ラベルは `test_0_<画像名>_out.nii.gz`、入力画像の復元結果は
`test_0_<画像名>_image.nii.gz` になります。実行前に新規の出力先であることを確認してください。

MUNet の Lightning チェックポイント（重みとハイパーパラメータを含む）が必要です。
通常の UNet チェックポイントや重みだけのファイルは使用できません。
保存された builder が元の事前学習チェックポイントを参照している場合、
そのファイルも保存時のパスからアクセスできる必要があります。
`--pretrained_path` は訓練用で、推論では `--ckpt` を使います。

復元処理の CPU テストは、プロジェクトの依存関係が利用できるコンテナ内で手動実行します。
一時ディレクトリに合成チェックポイントを作成し、終了時に削除します。
Mamba 演算のみ CPU 用の代替モジュールに置き換えるため、実際の CUDA 演算は検証しません。

```bash
uv run python -m unittest discover -s tests -p 'test_munet_inference.py' -v
```

### MUNet の推論と定量評価

```bash
uv run python main.py munet custom val \
  <データセット> <出力ルート> <学習時のプランJSON> \
  --ckpt <MUNetのチェックポイント.ckpt> --devices 0 --dice --hd 95
```

データセット直下の `image/` と `label/` に、対応する `.nii.gz` を置きます。
既存の `filekey` と同じく、ファイル名の最初の `.` または `_` より前を症例IDとします。
各ディレクトリ内のID重複や、画像・ラベル間の症例不足はエラーになります。
`--experiment_name` と推論用の引数は `inference` と共通です。
既存の推論・復元・保存処理を利用し、出力先も
`<出力ルート>/munet/[実験名/]<日時>/` になります。
既存の同名出力ディレクトリには書き込みません。

保存した予測と正解を元の画像グリッド上で比較し、`metrics/` に次のCSVを作成します。

| ファイル | 行の単位 |
| --- | --- |
| `full.csv` | 症例ID (`case`) × 臓器ラベルID (`organ`) |
| `case.csv` | 症例ごとの臓器平均 |
| `organ.csv` | 臓器ごとの症例平均 |

背景 (0) を除いたモデルの全ラベルIDを対象とします。
`--dice` を指定した場合のみ `dice` 列、`--hd 95` を指定した場合のみ
`hd95_mm` 列を出力します。HDのパーセンテージは `0 < p <= 100` で、
`--hd 100` は最大Hausdorff距離です。両方省略すると推論を行い、CSVはID列だけになります。
DiceはMONAIの `DiceMetric`、HDはMONAIの `compute_hausdorff_distance` をCPUで使用します。
入力画像のaffineから評価用グリッドを作ります。軸方向をSVD（極分解）で最も近い直交方向へ
補正し、voxel spacing・画像中心・軸の左右反転は維持します。
元画像のvoxel境界の8隅をすべて覆うようにグリッドサイズを決めます。
すでに直交している場合は元のグリッドを使います。軸平行化による大きな回転を避けて
補間による変化を抑える方式ですが、voxel単位の変化がなくなることを保証するものではありません。
予測・正解ラベルをMONAIの `ResampleToMatch` でこの共通グリッドへ合わせてから評価します。
補間は最近傍、変換元ラベルの範囲外からサンプリングする値は背景 (0) です。
入力画像は同じグリッドへ線形補間（3Dではtrilinear）で変換します。
元の保存済み画像は変更せず、直交化した入力画像・正解・予測を `orthogonal/` に保存します。
`val` では既存の推論保存先、`metric` では指定した予測ディレクトリの直下に作ります。
ファイル名は `test_<epoch>_<症例ID>_image.nii.gz`、`test_<epoch>_<症例ID>_gt.nii.gz`、
`test_<epoch>_<症例ID>_out.nii.gz` で、epochは元の予測ファイル名から引き継ぎます。
3種類は共通のshape・affine（mm単位）を持ち、保存するラベル配列で指標を計算します。
既存の `orthogonal/` は上書きしません。途中で失敗した場合は部分的な出力が残ることがあります。
HDは両方向の距離のパーセンタイルの最大値で、評価グリッドのvoxel spacingを使ったmm単位です。
NIfTIの空間単位はmmに換算し、未指定の場合はmmとして扱います。
入力画像にshearがある場合も、直交化した評価グリッド上でHDを計算します。

空の正解に対するDiceはMONAIの既定 (`ignore_empty=True`) に従い `nan` です。
空のマスクに対するHDもMONAIの結果 (`nan` または `inf`) をそのまま記録します。
平均は `nan` を除外し、すべて `nan` の場合は `nan`、`inf` を含む場合は `inf` です。
指標が未定義の症例・臓器について有限値に置き換えることはしません。

### 保存済みのMUNet予測の定量評価

```bash
python main.py munet custom metric \
  <imageとlabelを含むデータセット> <予測ファイルが置かれたディレクトリ> \
  --dice --hd 95
```

推論を行わず、既存の予測をCPUで評価します。チェックポイント・plan・GPU指定は不要です。
予測ディレクトリには `test_<epoch>_<症例ID>_out.nii.gz` を置きます。
入力画像の復元結果 (`*_image.nii.gz`) は評価対象になりません。
予測とデータセットの症例IDは一致する必要があり、同一症例の予測が複数ある場合はエラーです。
`--experiment_name bd0` などの指定も可能ですが、指定した予測ディレクトリを直接使用します。

指標計算とCSV出力は `custom val` と共通で、予測ディレクトリ直下の `metrics/` に
`full.csv`・`case.csv`・`organ.csv` を保存し、`orthogonal/` に直交化画像を保存します。
CLIの引数構造は変更ありません。既存の `metrics/`・`orthogonal/` は上書きしません。
モデルのクラス数を参照しないため、評価対象は共通の評価グリッドへ変換した全症例の
予測・正解に現れる前景ラベルIDの和集合です。
両方に現れないラベルIDはCSVに含みません。背景 (0) は除外し、疎なラベルIDもそのまま記録します。
`custom val` は引き続きモデルの全前景クラスを対象とします。

両コマンドとも、症例ごとの読み込み・グリッド確認・指標計算でエラーになった場合は、
`input_image: {...}`・`prediction_label: {...}`・`reference_label: {...}` の形式で
パス・shape・affine・spacing・空間単位・qform/sformを `pprint` で出力します。
画像配列は表示しません。元のグリッドが異なる場合も共通の評価グリッドへリサンプリングします。

```bash
python main.py munet custom metric --module-help
```

`custom val` と `custom metric` の評価・直交化・画像保存は症例単位でCPUプロセス並列化します。
`-w` / `--workers` でプロセス数を指定でき、既定は4、1なら逐次処理です。
`val` の推論自体は従来の処理を使います。
`metric` のラベルID収集には `Collecting labels`、両コマンドの評価・保存には
`Evaluating and saving` のtqdm進捗バーを表示し、完了した症例数を更新します。
各ワーカーのPyTorch CPUスレッド数は1とし、CSVは親プロセスで症例順に保存します。
症例ごとのエラーのメタデータは親プロセスが出力します。
並列数に応じて同時に読み込む画像が増えるので、利用可能なRAMに合わせて指定してください。

```bash
python main.py munet custom metric <データセット> <予測ディレクトリ> --dice --hd 95 --workers 4
```

直交化のCPU確認はコンテナ内で手動実行します。合成グリッドで直交性・spacing・中心・
元画像の領域保持・左右反転を確認します。データセットや出力ファイルは作成しません。
この確認はMONAIによるラベルのリサンプリング自体は検証しません。

```bash
python -m py_compile experiments/munet/evaluation.py
python - <<'PY'
from itertools import product
import numpy as np
from experiments.munet.evaluation import _evaluation_grid

shape = np.array([9, 12, 15])
for shear in (0.0, 0.0003, 0.2):
    for reflection in (1, -1):
        source = np.eye(4)
        source[:3, :3] = [[2 * reflection, shear, 0], [0, 3, 0], [0, 0, 4]]
        source[:3, 3] = [10, -20, 30]
        size, target, spacing = _evaluation_grid(shape, source)
        directions = target[:3, :3] / spacing
        np.testing.assert_allclose(directions.T @ directions, np.eye(3), atol=1e-12)
        np.testing.assert_allclose(spacing, np.linalg.norm(source[:3, :3], axis=0))
        np.testing.assert_allclose(target[:3, :3] @ ((np.array(size) - 1) / 2) + target[:3, 3],
                                   source[:3, :3] @ ((shape - 1) / 2) + source[:3, 3])
        assert np.sign(np.linalg.det(target[:3, :3])) == reflection
        corners = np.array(list(product(*[(-0.5, n - 0.5) for n in shape])))
        world = corners @ source[:3, :3].T + source[:3, 3]
        indices = (world - target[:3, 3]) @ np.linalg.inv(target[:3, :3]).T
        assert (indices >= -0.5 - 1e-9).all()
        assert (indices <= np.array(size) - 0.5 + 1e-9).all()
        if shear == 0:
            assert size == tuple(shape)
            np.testing.assert_array_equal(target, source)
print('grid checks passed')
PY
```

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
