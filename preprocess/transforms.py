import monai.transforms


def load_transformd(keys):
    return monai.transforms.Compose(
        [
            monai.transforms.LoadImaged(keys),
            monai.transforms.EnsureChannelFirstd(keys),
            monai.transforms.EnsureTyped(keys, data_type="tensor", track_meta=True),
        ]
    )


def planned_transformd(
    plan, image_key: list = ["image"], label_key: list | None = ["label"]
):
    """planを必要とする処理"""
    if label_key is None:
        all_key = image_key
    else:
        all_key = image_key + label_key
    need_label = label_key is not None
    jplan = plan["configurations"]["3d_fullres"]
    px = jplan["spacing"]
    splan = plan["foreground_intensity_properties_per_channel"]["0"]
    a_min: float = splan[
        "percentile_00_5"
    ]  # 0.5 percentile value of whole training dataset
    a_max: float = splan[
        "percentile_99_5"
    ]  # 99.5 percentile value of whole training dataset
    std: float = splan["std"]  # standard deviation of whole training dataset
    mean: float = splan["mean"]  # average of whole training dataset
    ret = [
        monai.transforms.CropForegroundd(
            all_key,
            select_fn=lambda x: x > -900,
            source_key="image",
            margin=10,
            allow_smaller=False,
        ),
        monai.transforms.Spacingd(image_key, pixdim=px, mode="bilinear"),
    ]
    if need_label:
        ret.append(monai.transforms.Spacingd(label_key, pixdim=px, mode="nearest"))
    monai.transforms.ScaleIntensityRanged(
        image_key,
        a_min=a_min,
        a_max=a_max,
        b_min=a_min,
        b_max=a_max,
        clip=True,
    )
    monai.transforms.NormalizeIntensityd(image_key, subtrahend=mean, divisor=std),
    return monai.transforms.Compose(ret)


def padded_crop_wrapper(
    keys, crop_size, transforms: list | tuple, padding_factor: float | None = None
):
    """クロップ処理を行った後に回転などが行われて0パディングされるのを防ぐためにあらかじめパディングされたクロップサイズでクロップし, 中の処理が終わってから正しいクロップサイズで切り出すためのラッパー.
    transformsにはアンパック可能なオブジェクトを入れること"""
    if padding_factor is None:
        padding_factor = 1.5
    padded_crop_size = [int(i * padding_factor) for i in crop_size]
    return monai.transforms.Compose(
        [
            monai.transforms.RandSpatialCropd(keys, padded_crop_size),
            *transforms,
            monai.transforms.CenterSpatialCropd(
                keys, crop_size
            ),  # 正しいパッチサイズを切り出す.
        ]
    )
