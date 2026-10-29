# custom val — 推論と定量評価

[READMEに戻る](../README.md)

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

## TotalSegmentator (`ttsg`)

```bash
uv run python main.py ttsg custom val <データセット> <出力ルート> --devices 0 --task total --roi_subset liver spleen --dice --hd 95 --workers 4
```

導入条件、入力構成、臓器選択、ラベルID対応、出力仕様は [inferenceのTotalSegmentator節](inference.md#totalsegmentator-ttsg-の推論と評価) を参照してください。

CPU並列化は -w / --workers で指定でき、既定は4、1なら逐次処理です。評価グリッドの手動確認と並列処理の詳細は [custom metric](custom-metric.md) を参照してください。
