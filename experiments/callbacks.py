import lightning as L
from lightning.pytorch.callbacks import Callback
from torchvision.utils import save_image


class ImageLogger(Callback):
    def on_train_batch_end(
        self, trainer: L.Trainer, module: L.LightningModule, outputs, batch, batch_idx
    ):
        # 100バッチごとに最初の1枚を保存
        if batch_idx % 100 == 0:
            x = batch["image"]
            save_image(x[0], f"img_{trainer.global_step}.png")
