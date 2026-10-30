# pretrained_seg custom observe

[READMEに戻る](../README.md)

`munet custom observe` と同じ方式で、学習ディレクトリを継続監視し、
書き込みが完了した10 epochごとの番号付きチェックポイントを空きGPUで評価します。
`last.ckpt` は対象外です。保存ディレクトリ名の epoch は Lightning の epoch + 1 です。
既存の pretrained_seg 推論と同じ MONAI sliding-window、overlap=0.5、Gaussian 合成、
replicate padding を使います。学習・通常推論・チェックポイント形式は変更しません。

以下はユーザーがプロジェクトのコンテナ内で手動実行してください。
`dataset/image/` と `dataset/label/` に同じ症例IDの `.nii.gz` を置き、学習時の plan を指定します。
出力先は入力・学習ディレクトリを含まない専用ディレクトリにしてください。

```bash
uv run python main.py pretrained_seg custom observe dataset/ trained/ plan.json observe-output/ \
  --observe-metrics basic quality confidence -w 4 --max-pending 2

uv run python main.py pretrained_seg custom --experiment_name lora \
  observe dataset/ "trained1/,trained2/" plan.json observe-output-lora/ \
  --observe-metrics all --hd 95 --save-predictions normal --save-orthogonal normal

uv run python main.py pretrained_seg custom observe --module-help
```

`lora`、`decoderunfreeze`、`zeroshot` からも呼び出せます。
チェックポイントに保存された builder を使って重みを厳密に復元します。
builder が参照する元の pretrained checkpoint も必要です。
head のクラス数は復元済み head から取得し、全前景クラスを評価します。

| 指標グループ | 内容 |
|---|---|
| `basic`（既定） | クラス別 voxel 数・割合。normal との比較は自己比較 |
| `quality` | 正解との Dice。HD は `--hd 95` などで明示指定 |
| `confidence` | エントロピー、最大確率、上位2クラスの差、クラス別平均確率 |
| `distribution` | normal との logits・確率比較。normal のみなので自己比較 |
| `bottleneck` | 各 BiMamba ブロックの入出力のサンプル統計と同じ段階の先頭パッチとの比較 |
| `internal` | 上記に加え LayerNorm、順方向・逆方向 Mamba、出力投影、decoder logits |
| `full-stats` | 全段階のサンプル統計、全要素統計、チャネル別統計、各 BiMamba のパラメータ統計 |
| `all` | 上記すべて。HD は明示指定が必要 |

pretrained_seg には munet 固有の gamma・PPE・パッチ間混合がないため、
条件は `normal` のみです。`--no-ppe` と `--no-mixing` は提供しません。
特徴の `position` は MONAI の呼び出し順のパッチ番号 `[index]` です。
`bottleneck` / `internal` / `full-stats` は BiMamba を含むモデルで利用できます。
先頭パッチとの比較は症例内・同じブロック・同じ段階で行います。

`--dice` / `--hd`、`--feature-samples`、`--interval`、`--devices`、
`--workers`、`--loader-workers`、`--max-pending`、`--temp-dir`、
`--save-predictions`、`--save-orthogonal` は [munet observe](custom-observe.md) と同じ指定です。
保存指定の `all` は `normal` と同じ結果になります。
推論後の logits を CPU に渡し、評価と保存は件数制限付きCPUプールで行います。
GPU上の合成 logits は通常推論と同じサイズなので、その分のGPUメモリが必要です。
一時ファイルの既定出力先はカレントディレクトリの `hidden-tmp/` です。

出力は `epoch-N/observations.jsonl`、`epoch-N/normal/metrics/`（精度評価時）、
`epoch-N/complete.json` と、ルートの `epochs.csv`、`epoch_metrics.csv`、
`paired_deltas.csv`、`summary.json` です。自己比較の delta は0です。
中断した試行は `.attempts/` に保持し、完了した epoch は再実行しません。
同じ出力先での再開は同じデータ・plan・指標・保存設定を使ってください。
学習ディレクトリの追加は可能です。Ctrl+C で終了できます。

変更後の構文・ヘルプ確認もコンテナ内で手動実行してください。

```bash
uv run python -m py_compile experiments/pretrained_seg/observe.py experiments/pretrained_seg/observe_cpu.py \
  experiments/pretrained_seg/__init__.py experiments/pretrained_seg/experiments/lora.py \
  experiments/pretrained_seg/experiments/decoderunfreeze.py experiments/pretrained_seg/experiments/zeroshot.py
uv run python main.py pretrained_seg custom observe --module-help
uv run python main.py pretrained_seg custom --experiment_name lora observe --module-help
```
