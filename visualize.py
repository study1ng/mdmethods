import json
from pathlib import Path
import matplotlib.pyplot as plt
from experiments import AnalyzedData

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("save_path", type=lambda p: Path(p).expanduser().resolve())
    parser.add_argument(
        "jsons", nargs="+", type=lambda p: Path(p).expanduser().resolve()
    )
    args = parser.parse_args()
    
    # JSONファイルを開いてデータを読み込む
    datas = [[AnalyzedData(i) for i in json.load(j.open())] for j in args.jsons]

    # 保存先ディレクトリが存在しない場合は作成
    args.save_path.mkdir(parents=True, exist_ok=True)

    # 比較したい指標のリスト
    metrics = [
        "min", "max", "mean", "med", "p005", "p995",
        "p050", "p950", "std", "shape_x", "shape_y", "shape_z", "spacing_x", "spacing_y", "spacing_z"
    ]

    # x軸のラベルとしてJSONファイル名（拡張子なし）を使用
    dataset_names = [j.stem for j in args.jsons]

    # 各指標ごとにboxplotを作成し保存
    for metric in metrics:
        plot_data = []
        for dataset in datas:
            # 各データセットから特定の指標の値のリストを抽出
            values = [getattr(data, metric) for data in dataset]
            plot_data.append(values)
        
        plt.figure(figsize=(10, 6))
        plt.boxplot(plot_data, tick_labels=dataset_names)
        plt.title(f"Boxplot of {metric}")
        plt.ylabel(metric)
        plt.xlabel("Dataset")
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        # 画像として保存
        save_file = args.save_path / f"{metric}_boxplot.png"
        plt.savefig(save_file)
        plt.close()