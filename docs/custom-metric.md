# custom metric — 保存済み予測の評価

[READMEに戻る](../README.md)

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

指標・空ラベル・評価グリッドの共通仕様は [custom val](custom-val.md) を参照してください。
