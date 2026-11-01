# inference — 学習済みモデルの推論

[READMEに戻る](../README.md)

## Encoder のパッチ特徴量出力

`mim` の `maskratio0_6_features` と `pretrained_seg` の `features` は、
元の実験の学習設定を引き継ぎ、`inference` で1症例の指定パッチから特徴量を保存します。
コンテナに `zarr>=3` が必要です。依存定義への追加・環境更新はユーザーが手動で行ってください。

```bash
uv run python main.py mim inference \
  --experiment_name maskratio0_6_features \
  <症例.nii.gz> <MIM出力先> <plan.json> --ckpt <mim.ckpt> \
  --patch-start 32 64 16 --patch-end 160 192 144 \
  --device cuda:0 --save-dtype bfloat16

uv run python main.py pretrained_seg inference \
  --experiment_name features \
  <症例.nii.gz> <seg出力先> <plan.json> --ckpt <seg.ckpt> \
  --patch-start 32 64 16 --patch-end 160 192 144 \
  --device cuda:0 --save-dtype float32
```

座標は `load_transformd` と `planned_transformd` 適用後（foreground crop、
spacing 変更、正規化、padding 後）の画像の `(H,W,D)` 順です。
開始を含み終了を含まないため、上記は `128×128×128` です。
範囲外の座標はエラーになります。ランダム変換や MIM のマスクは適用しません。
推論は float32、保存形式は `float32`（既定）、`float16`、`bfloat16` から選択できます。

出力は `<出力先>/<画像ファイル名から .nii.gz 等を除いたID>-features.zarr/` の
`patch_32_64_16__160_192_144/` 内に、`stem`, `stage_0`, … を保存します。
各配列は `(C,H,W,D)`、最終 stage は bottleneck です。
Zarr format 2 のストアに各層を順次保存し、チャンクはチャネル最大8、空間各軸最大32です。
同じ症例でも異なる座標は追記できます。同一座標は保存 dtype が違っても上書きしません。
画像・チェックポイントのパス／サイズ／更新日時、plan のハッシュが異なる場合も拒否するため、
モデルや plan の比較には別の出力先を使ってください。同時書き込みは避けてください。

パッチ属性に座標、前処理後の画像サイズと affine、保存 dtype を記録します。
`complete=true` が正常終了の目印です。途中で失敗したパッチは `complete=false` のまま残り、
再実行でも上書きしません。別の出力先を選ぶか、該当する未完了パッチを手動で削除してください。

`bfloat16` はビット列を `uint16` として保存し、配列属性に
`logical_dtype="bfloat16"` を記録します。数値として uint16 に変換したものではありません。
読み込み時は以下のように復元します。

```python
import torch
import zarr

root = zarr.open_group("<case-id>-features.zarr", mode="r")
array = root["patch_32_64_16__160_192_144/stem"]
feature = torch.from_numpy(array[:]).view(torch.bfloat16)
```

CLI の確認は、コンテナ内で手動実行してください（特徴量は書き込みません）。

```bash
uv run python main.py mim inference --experiment_name maskratio0_6_features --module-help
uv run python main.py pretrained_seg inference --experiment_name features --module-help
```

## pretrained_seg

```bash
# 推論
uv run python main.py pretrained_seg inference \
  <前処理済みデータ> <出力先> <チェックポイント> <プランJSON>
```

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

#### パッチ間情報交換と PPE の比較

`munet inference` と `munet custom val` では、既存チェックポイントに対して
`--no-mixing` と `--no-ppe` を独立に指定できます。指定しない場合は従来の処理です。

| オプション | パッチ間情報交換 | PPE に渡す開始座標 |
| --- | --- | --- |
| なし | あり | 元の座標 |
| `--no-mixing` | なし | 元の座標 |
| `--no-ppe` | あり | 全パッチで `(0, 0, 0)` |
| `--no-mixing --no-ppe` | なし | 全パッチで `(0, 0, 0)` |

`--no-mixing` は各パッチを一つずつ既存のボトルネックに通します。
パッチ内の双方向 Mamba、学習済み重み、gamma による残差加算は維持します。
`--no-ppe` は PPE ベクトルの除去やゼロ化ではなく、原点の sin/cos 符号化を使用します。
パッチ内の位置符号化（pe）と、skip 特徴の対応付け・出力結合用の元の座標は維持します。
パッチ分割、重複率、Gaussian による logits の結合も変更しません。
`--no-ppe` 単独では、Mamba が参照するパッチの並び順は残ります。

```bash
uv run python main.py munet inference \
  <画像ディレクトリ> <比較用の出力ルート> <学習時のプランJSON> \
  --ckpt <MUNetのチェックポイント.ckpt> --devices 0 --no-mixing

uv run python main.py munet custom val \
  <imageとlabelを含むデータセット> <比較用の出力ルート> <学習時のプランJSON> \
  --ckpt <MUNetのチェックポイント.ckpt> --devices 0 \
  --dice --no-mixing --no-ppe
```

同じチェックポイント・症例・推論設定で4条件を比較すると、学習済みモデルにおける
情報交換と全画像の位置情報の寄与を調べられます。これは推論時のアブレーションであり、
独立パッチ方式を再学習した場合との比較ではありません。

復元処理の CPU テストは、プロジェクトの依存関係が利用できるコンテナ内で手動実行します。
一時ディレクトリに合成チェックポイントを作成し、終了時に削除します。
Mamba 演算のみ CPU 用の代替モジュールに置き換えるため、実際の CUDA 演算は検証しません。

```bash
uv run python -m unittest discover -s tests -p 'test_munet_inference.py' -v
```

## TotalSegmentator (`ttsg`) の推論と評価

TotalSegmentatorはコンテナ内で手動導入してください。`pyproject.toml` の依存関係への
`TotalSegmentator` の追加と、ロックファイル・環境の更新はユーザーが行います。
バージョンは既存のPyTorch・nnU-Net関連依存との互換性を確認して選択してください。
初回推論時には学習済み重みをダウンロードする場合があります。

```bash
uv run python main.py ttsg inference <画像ディレクトリ> <出力ルート> \
  --devices 0 --task total --roi_subset liver spleen

uv run python main.py ttsg custom val <データセット> <出力ルート> \
  --devices 0 --task total --roi_subset liver spleen \
  --dice --hd 95 --workers 4
```

`inference` は入力ディレクトリ直下の `.nii.gz` を処理します。
`custom val` はデータセット直下の `image/` と `label/` を使います。
症例IDの取得・重複検出は既存の規約に従います。
プランJSONと `--ckpt` は不要です。`--devices` はGPU番号を1つ指定します。
CPUは `--cpu`、低解像度モデルは `--fast` で指定します。
`--task` の既定値は `total` で、利用可能なタスクは導入したTotalSegmentatorに依存します。
`--roi_subset`（別名 `--roi-subset`）は `total`・`total_mr` に対応し、省略時は全臓器を対象にします。
推論・評価とも選択した臓器が対象で、症例に存在しない臓器も評価対象に含めます。

正解IDは既定で選択タスクのTotalSegmentator IDと解釈します。
異なるID体系では `--label-map <JSONファイル>` を追加してください。
例えば `--roi_subset liver spleen` に対して `{"liver": 1, "spleen": 2}` とします。
JSONは選択した全臓器の名前と、重複しない正整数の正解IDを指定します。
選択していない臓器名は含めません。

出力先は `<出力ルート>/ttsg/<日時>/` で、既存ディレクトリには書き込みません。
予測は `test_0_<症例ID>_out.nii.gz` に保存します。
**保存IDは、タスクの元ID順に選択臓器を1から連番にしたものです。**
元のTotalSegmentator ID・正解ID・保存IDの対応は `labels.json` に記録します。
CSVの `organ` 列と `orthogonal/` のラベルも、この保存IDを使います。
選択外のラベルは背景0です。元データは変更しません。
`val` は変換した正解を `reference/` に保存し、既存の評価処理で
`metrics/` と `orthogonal/` を出力します。
指標・空ラベルの扱い・CPU並列化はMUNet評価と共通です。
途中で失敗すると部分的な出力が残る場合があります。

構文とCLIヘルプはコンテナ内で手動確認してください。以下は推論を開始しません。

```bash
uv run python -m py_compile experiments/ttsg/__init__.py
uv run python main.py ttsg inference --module-help
uv run python main.py ttsg custom val --module-help
```

定量評価の入口は [custom val](custom-val.md) を参照してください。
