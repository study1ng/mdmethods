# train — モデルの学習

[READMEに戻る](../README.md)

実行前に対象モジュールの引数、入力パス、出力先を確認してください。

```bash
# 学習
uv run python main.py mim train \
  <前処理済みデータ> <出力先> <プランJSON>

# 名前付き実験による学習
uv run python main.py munet train \
  --experiment_name overlap_0 \
  <前処理済みデータ> <出力先> <プランJSON> \
  --pretrained_path <チェックポイント>
```

学習ではデフォルトでGPUとbf16混合精度を使用します。出力先にはチェックポイント、CSVログ、TensorBoardログなどが作成され、MLflowはSQLiteデータベースを使用します。
