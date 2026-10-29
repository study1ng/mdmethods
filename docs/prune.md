# prune — データ選別

[READMEに戻る](../README.md)

解析結果とプランに基づいてデータを選別し、保存先にシンボリックリンクを作成します。

```bash
uv run python main.py mim prune <解析結果JSON> <対象ディレクトリ> <新規保存先> --plan_path <プランJSON>
```

対象と保存先はペアで指定し、複数のペアも指定できます。実行前にパスの対応と既存ファイルの有無を確認してください。モジュール固有の引数・選別基準は uv run python main.py mim prune --module-help で確認してください。
