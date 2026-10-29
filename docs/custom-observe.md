# custom observe — チェックポイントの継続観測

[READMEに戻る](../README.md)

## MUNetチェックポイントの継続観測 (`custom observe`)

```bash
uv run python main.py munet custom observe data/ trained/ plan.json output/ \
  --dice --hd 95 -w 4 -i 60

# 同じ学習の継続先をまとめて指定する場合（全体を引用符で囲む）
uv run python main.py munet custom observe data/ "trained1/,trained2/,trained3/" plan.json output/ \
  --dice --hd 95 -w 4 --temp-dir /tmp --loader-workers 0

# 候補GPUを限定する場合
uv run python main.py munet custom observe data/ trained/ plan.json output/ \
  --dice --hd 95 --devices 0 1 --interval 120

# 特殊なgammaの復元には、学習時と同じexperiment_nameを指定する
uv run python main.py munet custom --experiment_name bd0_gamma_tanh \
  observe data/ trained/ plan.json output/ --dice --hd 95
```

`data/image/` と `data/label/` の `.nii.gz` を既存の症例ID規則で対応付けます。
`trained/` は一つの学習の出力ディレクトリを指定してください。中断・再開で分かれた場合は
`"trained1/,trained2/,trained3/"` のようにカンマ区切りで指定できます。空要素はエラーです。
パス自体にカンマを含むディレクトリ名には対応しません。全探索先の配下を再帰検索し、
`epoch=9` 等を含む `.ckpt` / `.pth` / `.pt` のうち、10エポック刻みのファイルを
全体をまとめて昇順で処理します。標準の `ckpt_epoch=9.pth.ckpt` は `output/epoch-10/` に対応します。
内部epochは0始まり、出力epochは完了した学習エポック数です。`last.ckpt` は対象外です。
探索先の重複や親子ディレクトリの指定によって同じファイルを発見しても一度だけ扱います。
同じepochの別ファイルは、全候補の書き込みが安定してからSHA256を比較し、内容が同一なら
一度だけ処理します。内容が異なる場合は停止します。epoch番号をリセットした別学習は
混在させず、継続学習ではチェックポイントのepoch番号も引き継いでください。

`-i / --interval` は正の秒数で既定60。ファイルのサイズ・更新時刻が、この間隔以上離れた
2回の走査で変わらないことを確認してから読み込みます。全件処理後も終了せず監視し、
Ctrl+Cで終了します。実行中の子プロセスも停止し、完了したepochは残ります。
`-w / --workers` は既存valと同様、CPUでの評価処理の並列数です。
`--loader-workers` は推論DataLoader専用の並列数で既定4、0で多重処理を無効にします。
`--temp-dir` は既存の書き込み可能なローカル一時ディレクトリを指定します。
Linuxでの既定は `/tmp`、Windowsではシステムの一時ディレクトリです。子プロセスの起動前に
`TMPDIR` / `TEMP` / `TMP` をこの値に設定し、DataLoaderと評価プロセスにも引き継ぎます。
共有ストレージ上の `pymp-*` 削除時に `Device or resource busy` が出る場合は、
ローカル領域を指定してください。それでも続く場合は `--loader-workers 0 -w 1` で
両方の多重処理を無効にして切り分けられます。このエラーは一時領域の状態にも依存するため、
ローカル領域への変更だけで必ず解消するとは限りません。既存の使用中の一時領域は自動削除しません。
DiceとHausdorff距離はそれぞれ `--dice`、`--hd 95` で有効になります。指定しない指標は
計算しません。既存の `--no-ppe` / `--no-mixing` は観測の基準条件にも適用されます。
通常の研究比較では両方とも省略してください。

GPUはチェックポイント処理の直前に `nvidia-smi` で2回確認します。計算プロセスがなく、
使用率0%、使用メモリ64MiB以下のGPUから選びます。情報が取得できないGPUは選びません。
空きがなければ処理を進めず、interval秒後に再確認します。GPU UUIDで子プロセスに1台だけを
公開し、終了時にCUDAコンテキストを解放するため、待機中はGPUを保持しません。
GPUの排他的予約ではないので、確認と利用開始の間に他のプロセスが起動する競合は残ります。
MIG識別子や未対応のtelemetryは選択対象外です。
`--devices` の整数はCUDAの可視GPUの論理番号です。CUDA driverのデバイス列挙とUUID照会で
`CUDA_VISIBLE_DEVICES` / `CUDA_DEVICE_ORDER` を反映し、nvidia-smiの番号との食い違いを避けます。
監視プロセスはCUDAコンテキストを作成しません。driverの照会も失敗した場合は待機します。
選んだGPU UUIDとnvidia-smi上の番号は `gpu.json` に残ります。

出力構成は以下です。各条件の予測は既存valと同じ逆変換・保存・評価処理を使います。

```text
output/
  observe.json                   # 入力・設定・データファイルのサイズ/更新時刻
  epochs.csv                     # 全epochの症例別・臓器別の評価値
  epoch_metrics.csv              # 症例平均の分布（平均、標準偏差、中央値など）
  paired_deltas.csv              # 同じ症例/臓器のnormalとの差（条件 - normal）
  summary.json                   # gamma、予測比較、計算時間の経時データ
  epoch-10/
    complete.json                # チェックポイントSHA256、成果物一覧、完了情報
    input.json
    gpu.json
    worker.log
    observations.jsonl           # 特徴量・パッチ・予測・環境の詳細
    normal/                      # 基準条件。image/out NIfTI、metrics/、orthogonal/
    gamma_zero/                  # encoder残差だけをdecoderに渡す
    no_ppe/                      # PPE座標を0にする（局所PEは保持）
    no_mixing/                   # パッチを独立にボトルネックに通す
    no_ppe_no_mixing/
    reverse_order/               # パッチ順を逆転。保存座標は保持
  .attempts/                     # 中断/失敗した未確定結果。自動削除しない
```

全6条件の推論、予測画像保存、正解との評価が成功してからepochディレクトリを確定します。
再実行では完了情報と成果物の存在・サイズを確認し、完了済みをスキップします。
中断した処理はepoch単位で再処理します。不完全な既存epochは `.attempts/` に保存して
やり直します。チェックポイントの内容の差し替え、入力データ/plan/観測設定の変更は停止して
別の出力先を要求します。探索先は再開時に追加できます。以前の単一探索先の設定も読み込めます。
探索先の順序変更・重複指定は結果に影響せず、既存探索先の削除は許可しません。
同一内容のチェックポイントコピーを選んだ場合も完了済みepochは再処理しません。
interval、GPU候補、CPU並列数、loader-workers、temp-dirは再開時に変更できます。
同一出力先への同時起動はOSのファイルロックで防ぎます。既存valの出力を完了済みobserveに
流用することはできません。

観測内容:

- 実効gamma、parametrization前の値、学習可否、全ボトルネックパラメータの統計。
- encoder、PE、Mamba各段、残差加算後、decoderの特徴統計、非有限値・ゼロ率、
  チャネル別平均/RMS、サンプル分位点。
- 同一パッチ・同一段のnormalとの差、および各条件の先頭パッチとの差。
  MAE、RMSE、最大差、相対L2、ノルム比、コサイン類似度を保存。
  gamma適用後の更新量とencoder特徴の比も保存します。
- stitched logits/確率の差、ラベル変更率、KL/JSダイバージェンス、エントロピー、
  確信度、上位2クラスの確率差、クラス別voxel数・増減・normalとのDice。
- 条件別時間、GPUピーク使用量、パッチ座標・順序、環境・ライブラリ情報、実装ファイルのSHA256。

特徴比較はメモリを抑えるため各パッチ・各段から等間隔サンプルを保持し、既定上限4096要素です。
`--feature-samples` で変更できます。チャネル別統計は全特徴を使います。
添字は整数演算で作り、要素数がfloat32の整数精度を超える大きな特徴テンソルでも
末尾を超えないようにしています。MetaTensorからのサンプルは通常のTensorとして保持します。
生特徴や全確率volumeは保存しません。logit/確率の最終比較はサンプルではなく全voxelです。
確率・特徴比較は出力逆変換前の推論gridで行い、正解とのDice/HDは既存valの直交gridで行います。
空ラベルやNaN/Infの扱いも既存valに従い、集約では有限値数と非有限値数を分けます。
`paired_deltas.csv` の正値はDiceでは改善、HDでは悪化です。

gamma_zeroは同じ学習済みdecoderで更新枝を無効化する介入です。元の事前学習UNetとの
比較ではありません。no_mixingは同じパッチの周囲文脈への感度、reverse_orderは順序感度を
調べますが、別パッチ同士の差には入力内容の差も含まれます。計算時間には観測・CPU転送・
stitchingが含まれ、通常推論のベンチマークではありません。6条件を実行するため通常valより
時間とディスク容量を使います。失敗時はworker.logを残して次のラウンドで再試行します。

Pythonによる検証はユーザーがコンテナ内で手動実行してください（下記は推論を開始しません）。

```bash
uv run python -m py_compile experiments/munet/observe.py experiments/munet/__init__.py \
  experiments/munet/model.py experiments/munet/munet_bottleneck.py \
  experiments/munet/experiments/bd0_gamma_zero.py tests/test_munet_observe.py
uv run python main.py munet custom observe --module-help
uv run python main.py munet custom --experiment_name bd0_gamma_tanh observe --module-help
uv run python -m unittest discover -s tests -p test_munet_inference.py
uv run python -m unittest discover -s tests -p test_munet_observe.py
```

追加したobserveの回帰テストはCPUと合成ファイルのみを使います。今回の189,464,576要素の
添字境界やint64上限付近の境界を、巨大テンソルを確保せず検証します。複数探索先の昇順・重複排除、
コピーの安定待ち、競合検出、旧設定からの再開、一時領域の子プロセスへの引き渡しも対象です。
