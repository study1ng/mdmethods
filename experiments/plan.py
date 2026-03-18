from experiments.utils import loaded_json
from pathlib import Path



class Plan:
    def __init__(self, plan_path: Path):
        self.plan_path = plan_path
        self.plan = loaded_json(plan_path)
        self.splan = self.plan["foreground_intensity_properties_per_channel"]["0"]
        self.jplan = self.plan["configurations"]["3d_fullres"]
        self.patch_size = self.jplan["patch_size"][::-1]
        self.batch_size = self.jplan["batch_size"]
        self.spacing = self.jplan["spacing"][::-1]
        self.max = self.splan["max"]
        self.min = self.splan["min"]
        self.std = self.splan["std"]
        self.mean = self.splan["mean"]
        self.median = self.splan["median"]
        self.percentile_00_5 = self.splan["percentile_00_5"]
        self.percentile_99_5 = self.splan["percentile_99_5"]
        self.pool_strides = self.jplan["pool_op_kernel_sizes"]
        self.conv_kernel_size = self.jplan["conv_kernel_sizes"]
        self.stem_channel = self.jplan["UNet_base_num_features"]