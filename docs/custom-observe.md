# custom observe — チェックポイントの継続観測

[READMEに戻る](../README.md)

## 使用例

`data/image/` と `data/label/` の症例IDが対応するデータと、学習時の plan を指定します。
以下の推論コマンドはユーザーがコンテナ内で手動実行してください。

```bash
# 通常監視：基本指標だけ。NIfTI は保存しない
uv run python main.py munet custom observe data/ trained/ plan.json output/ \
  --observe-metrics basic -w 4 --max-pending 2

# 基本指標・bottleneck のサンプル比較・正解との Dice
uv run python main.py munet custom observe data/ trained/ plan.json output/ \
  --observe-metrics basic bottleneck --dice -w 4 --max-pending 4

# 詳細解析：全指標、HD95、全条件の予測・直交画像を保存
uv run python main.py munet custom observe data/ trained/ plan.json output/ \
  --observe-metrics all --hd 95 --save-predictions all --save-orthogonal all

# 継続学習の探索先をまとめる。gamma の特殊な復元には学習時の experiment_name を指定
uv run python main.py munet custom --experiment_name bd0_gamma_tanh \
  observe data/ "trained1/,trained2/" plan.json output/ \
  --observe-metrics basic bottleneck --dice --temp-dir /tmp --loader-workers 0
```

## 指標の選択

`--observe-metrics` は空白区切りで複数指定できます。既定は `basic` のみです。
指定したグループだけを有効にするため、基本指標も必要なら `basic` を含めてください。

| グループ | 内容 |
|---|---|
| `basic` | normal からのラベル変更率、クラス別voxel数・割合・増減、normal との予測 Dice |
| `quality` | 正解との Dice・HD。単独指定時は Dice を有効化。HD は常に `--hd` で明示 |
| `confidence` | 平均エントロピー、平均最大確率、上位2クラスの確率差、クラス別平均確率 |
| `distribution` | normal に対する全voxelの logits・確率差、完全一致・許容誤差内の割合、KL・JS |
| `bottleneck` | encoder、Mamba出力、残差加算後のサンプル統計・比較、gamma 適用後の更新量／encoder のサンプルL2比 |
| `internal` | PPE、LayerNorm、順方向・逆方向 Mamba、出力射影、bias を除く射影、decoder を含む全段階のサンプル観測 |
| `full-stats` | 全段階のサンプル観測に加え、全要素統計、チャネル平均・RMS、空間標準偏差、全特徴ノルム、bottleneck パラメータ統計 |
| `all` | 上記すべて。Dice を有効化するが、HD は明示指定が必要 |

従来どおり `--dice` / `--hd 95` を指定しても精度評価を有効にできます。
その場合は有効グループに `quality` が追加されます。`quality --hd 95` は HD のみで、
Dice も必要なら `--dice` を追加してください。

gamma、実行環境、ソースのSHA256、条件別の時間・GPUメモリはグループにかかわらず記録します。
`basic` の予測 Dice は normal の予測との一致であり、正解との Dice とは別です。

### 特徴観測

`bottleneck` / `internal` では GPU 上でサンプルを抽出し、サンプルだけを CPU に転送します。
特徴観測を選択しなければ callback を設定しません。`full-stats` だけが全特徴を CPU に転送します。
全要素統計は `feature_full_stats`、サンプル統計は `feature.sample_stats` として区別します。

サンプルは各パッチ・各段階から等間隔に最大4096要素を取り、
`--feature-samples` で上限を変更できます。添字は整数演算で作り、非連続テンソルも
全体を flatten するためのコピーを作らず抽出します。サンプルは元テンソルの変更から独立しています。

同じパッチ・同じ段階の normal、および同条件の先頭パッチに対し、
MAE・RMSE・最大差・相対L2・ノルム比・cosine を記録します。
サンプル分位点は p01 / p05 / p50 / p95 / p99 です。
全特徴の更新量比は `full-stats`、サンプルからの比は `sample_` 付きの項目です。
生特徴や全確率volumeはファイルに保存しません。

`confidence` / `distribution` が不要なら softmax と確率volumeの保持を省略します。
normal 自身に対する出力差・KL・JSは既知の一致結果を使い、差分走査を省きます。

## 保存の選択

| オプション | 値 | 既定 |
|---|---|---|
| `--save-predictions` | `none` / `normal` / `all` | `none` |
| `--save-orthogonal` | `none` / `normal` / `all` | `none` |

予測保存は通常 val と同じラベル逆変換と NIfTI writer を使います。
正解との Dice・HD は予測ファイル保存なしでも計算でき、既存評価と同じ直交グリッド、
nearest 補間、spacing、空ラベルの扱いを使います。

直交画像の保存を有効にすると、normal に共通の入力画像・正解・予測を保存し、
他条件には予測だけを保存します。共通画像・正解は normal/orthogonal/ を参照してください。
入力画像の intensity 再サンプリングと圧縮保存を条件ごとに繰り返しません。
精度評価と直交保存の両方が無効なら、評価グリッドへの再サンプリングは行いません。
推論空間の特徴・確率比較と、元画像に近い直交空間の精度評価は区別してください。

## GPU と CPU の並列処理

GPU の推論プロセスは常に1つで、6条件を順番に実行します。
encoder の共有、モデル構造、checkpoint、forward、パッチ合成の計算式は変更していません。
`reverse_order` も従来どおり、モデルへの入力順だけを変え、CPU 合成の加算順は元に戻します。

CPU 後処理は spawn 方式の ProcessPoolExecutor で実行します。

- `-w / --workers`：CPU 後処理の最大プロセス数。既定4。各 worker の PyTorch スレッド数は1。
- `--max-pending`：実行中と待機中を合計した CPU ジョブ数の上限。既定2。
- `--loader-workers`：入力 DataLoader 専用。既定4、0で読み込みの多重処理を無効化。
- `--temp-dir`：既存のローカル一時ディレクトリ。Linux は /tmp、Windows はシステム既定。

出力後処理は「1症例の6条件」を1ジョブとして、合成・比較・保存・Dice/HDまでまとめます。
その間 GPU は次の症例を推論できます。同じ症例の normal を別workerに複製する必要がなく、
症例間で並列化します。1症例だけの場合、この後処理の並列効果は限定的です。
`full-stats` の全特徴統計は別のジョブとして推論中にも処理します。別の入れ子poolは作りません。
GPU/modelはworkerへ渡さず、logitsとまとめたサンプルを共有CPUストレージで渡します。

上限に達した場合はGPU側が待機し、無制限にRAMを消費することを防ぎます。
上限はバイト数ではありません。処理中の1症例には6条件分のパッチlogitsが必要で、
投入済みジョブに加えて現在構築中の1症例・転送用一時データ・workerの合成/評価用RAMも使います。
Linuxでは共有テンソルに /dev/shm の容量も必要です。大きな症例でメモリ不足になる場合は
まず `--max-pending 1` に下げてください。4 workerを十分使う場合は、RAMを確認して
`--max-pending 4` に上げます。`-w 1` でもCPU後処理は別プロセスで動きます。

TMPDIR / TEMP / TMP はGPU子プロセス起動前に設定し、CPU worker と DataLoader に引き継ぎます。
共有ストレージの一時領域でIPCエラーが出る場合はローカルの `--temp-dir` を指定してください。
データや一時領域は自動削除しません。

## 計時と完了管理

- `runtime`：forward、観測用データ転送、callback内のキュー待機。合成・出力指標・保存は除外。
- `cpu_runtime`：条件別の合成、予測指標、逆変換と予測保存、精度評価と直交保存の時間。
- `cpu_case_runtime`：特徴サンプル解析を含めた症例単位のCPU時間。
- `feature_full_stats.cpu_seconds`：全特徴統計のCPU時間。
- `pipeline.queue_wait_seconds`：キュー制御・最終回収でfutureを待った合計時間。
- `complete.json.seconds_including_metrics`：全症例の推論とCPUジョブ・CSV出力までの経過時間。

GPUからCPUへの共有ストレージ準備など、個別時間に含まれない管理処理もあります。
並列区間は重なるため、各時間を足してもepochの経過時間には一致しません。
`runtime` は観測設定にも依存し、通常推論だけのベンチマークではありません。

全CPUジョブと成果物出力が成功した後にだけ `complete.json` を作成します。
workerの例外はGPU側へ伝え、失敗したepochを完了扱いにしません。

## 監視・再開

探索先は `"trained1/,trained2/"` のようにカンマ区切りで指定でき、空要素はエラーです。
配下の .ckpt / .pth / .pt のうちepoch番号を含む10 epoch刻みのスナップショットを昇順に処理します。
Lightningの内部epochは0始まりなので、epoch=9 は output/epoch-10/ に対応します。
last.ckpt は対象外です。同じepochの複数コピーは安定待ち後にSHA256を比較し、競合なら停止します。

`-i / --interval` は走査間隔で既定60秒です。サイズ・更新時刻がこの間隔以上離れた
2回の走査で変わらないことを確認してから処理します。処理後も監視を続け、Ctrl+Cで終了します。

GPUは直前に2回確認し、計算プロセスなし・使用率0%・使用メモリ64MiB以下から1台選びます。
`--devices 0 1` で候補のCUDA論理番号を制限できます。UUIDで子プロセスに1台だけ公開します。
GPUの排他的予約ではないため、他プロセスとの競合の可能性は残ります。
待機中の監視プロセスはCUDAコンテキストを作成せず、epochごとの子プロセス終了で解放します。

同じ出力先はOSロックで保護し、完了済みepochは成果物の存在・サイズを確認してスキップします。
失敗・中断時の出力は .attempts/ に保持し、epoch単位で再試行します。
同じ学習の探索先は追加できますが、削除・入力データ・plan・指標・保存設定の変更は別の出力先が必要です。
interval、GPU候補、CPU worker数、max-pending、loader-workers、temp-dirは再開時に変更できます。

今回の出力形式はschema 2です。schema 1の出力先は再利用せず、新しい出力先を指定してください。
既定の指標・画像保存も旧版から変わっています。通常の custom val / custom metric の既定動作は維持します。

## 出力

```text
output/
  observe.json                   # 入力・指標・保存設定・データ署名
  epochs.csv                     # 症例別・臓器別の精度評価
  epoch_metrics.csv              # 症例平均の分布
  paired_deltas.csv              # 条件 - normal の精度差
  summary.json                   # gamma・予測比較・GPU/CPU時間
  epoch-10/
    complete.json
    input.json
    gpu.json
    worker.log
    observations.jsonl
    normal/                      # 選択時だけ NIfTI / metrics / orthogonal
    gamma_zero/
    no_ppe/
    no_mixing/
    no_ppe_no_mixing/
    reverse_order/
  .attempts/
```

精度評価を無効にした場合、評価CSVは作らず、全epoch集約の評価CSVはヘッダーのみです。
gamma_zero は学習済みdecoderにencoder残差だけを渡す介入で、元の事前学習UNetとの比較ではありません。
no_ppe はPPE座標を0にしますが局所PEは残ります。
no_mixing はパッチ間の文脈、reverse_order は入力順序への感度を調べます。
パッチ同士の特徴差には入力内容の差も含まれ、因果関係を断定する指標ではありません。

## 手動検証

エージェントはPythonを実行していません。ユーザーがコンテナ内で実行してください。
構文・ヘルプ・既存のCPU回帰テストは推論ジョブを開始しませんが、プロジェクトの依存関係は必要です。

```bash
uv run python -m py_compile experiments/munet/observe.py experiments/munet/observe_cpu.py \
  experiments/munet/evaluation.py
uv run python main.py munet custom observe --module-help
uv run python main.py munet custom --experiment_name bd0_gamma_tanh observe --module-help
uv run python -m unittest discover -s tests -p test_munet_observe.py
uv run python -m unittest discover -s tests -p test_munet_inference.py
```

新しいCPU指標の小規模な確認は、以下をコンテナのシェルで実行できます。
合成テンソルだけを使い、ファイル保存やGPU推論はしません。

```bash
uv run python - <<'PY'
import math
from pathlib import Path
from unittest.mock import patch
import torch
from experiments.munet.observe_cpu import process_case

def job(groups):
    normal = torch.zeros(1, 2, 2, 2, 2)
    other = normal.clone()
    other[:, 1] = math.log(3)
    return dict(case="synthetic", groups=groups, features=[], samples=torch.empty(0),
                gamma=1., patch_count=1, output_size=(2, 2, 2), spacing=[1, 1, 1],
                outputs=[("normal", [(normal, (0, 0, 0))]),
                         ("no_ppe", [(other, (0, 0, 0))])],
                batch=None, source={}, directory=Path("."), num_classes=2,
                dice=False, hd=None, save_predictions="none", save_orthogonal="none")

with patch.object(torch.Tensor, "softmax", side_effect=AssertionError("unneeded softmax")):
    result = process_case(job(["basic"]))
pred = [r for r in result["records"] if r["event"] == "prediction"]
assert pred[0]["changed_fraction"] == 0
assert pred[1]["changed_fraction"] == 1
assert pred[1]["classes"][1]["voxels"] == 8
result = process_case(job(["basic", "confidence", "distribution"]))
pred = [r for r in result["records"] if r["event"] == "prediction"]
assert pred[0]["kl_condition_to_normal_nats"] == 0
assert abs(pred[1]["confidence_mean"] - .75) < 1e-6
assert abs(pred[1]["kl_condition_to_normal_nats"] -
           (.75 * math.log(1.5) + .25 * math.log(.5))) < 1e-6
print("CPU metric smoke checks passed")
PY
```

複数プロセスでの受け渡し、保存なしDice/HDと通常valとの一致、逆変換後のNIfTI、
worker失敗時の未完了扱い、実データでの速度・RAMは別途コンテナで確認が必要です。
実データを使う場合は少数症例の独立した入力と、未使用の出力先で確認してください。
