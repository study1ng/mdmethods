from experiments.config import filekey
class AnalyzedData:
    def __init__(self, data):
        self.min = data["min"]
        self.max = data["max"]
        self.mean = data["mean"]
        self.med = data["median"]
        self.p005 = data["p005"]
        self.p995 = data["p995"]
        self.p050 = data["p050"]
        self.p950 = data["p950"]
        self.std = data["std"]
        self.shape = data["cropped_shape"]
        self.shape_x = self.shape[1]
        self.shape_y = self.shape[2]
        self.shape_z = self.shape[3]
        self.spacing = data["spacing_original"]
        self.spacing_x = self.spacing[0]
        self.spacing_y = self.spacing[1]
        self.spacing_z = self.spacing[2]
        self.path = data["path"]
        self.filekey = filekey(self.path)
