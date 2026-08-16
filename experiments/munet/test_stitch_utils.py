import unittest

import torch
from monai.utils import BlendMode

from experiments.munet.stitch_utils import split_to_patch, stitch_logits


class SplitToPatchTest(unittest.TestCase):
    def test_splits_image_into_patches_with_positions(self):
        image = torch.arange(4**3).reshape(1, 1, 4, 4, 4)

        patches = split_to_patch(image, patch_size=(2, 2, 2), overlap=0.0)

        expected_positions = tuple(
            (depth, height, width)
            for depth in (0, 2)
            for height in (0, 2)
            for width in (0, 2)
        )
        self.assertEqual(tuple(position for _, position in patches), expected_positions)

        for patch, position in patches:
            depth, height, width = position
            expected = image[
                ...,
                depth : depth + 2,
                height : height + 2,
                width : width + 2,
            ]
            torch.testing.assert_close(patch, expected)

    def test_patches_cover_entire_image_with_overlap(self):
        image = torch.zeros((1, 1, 4, 4, 4))

        patches = split_to_patch(image, patch_size=(3, 3, 3), overlap=0.5)

        coverage = torch.zeros(image.shape[2:], dtype=torch.int64)
        for patch, position in patches:
            slices = tuple(
                slice(start, start + length)
                for start, length in zip(position, patch.shape[2:])
            )
            coverage[slices] += 1

        self.assertGreater(
            coverage[-1, -1, -1].item(),
            0,
            "The final corner of the image is not covered by any patch.",
        )
        uncovered = (coverage == 0).nonzero(as_tuple=False).tolist()
        self.assertEqual(uncovered, [], f"Uncovered voxel positions: {uncovered}")


class StitchLogitsTest(unittest.TestCase):
    def test_constant_blending_averages_overlapping_logits(self):
        left = torch.ones((1, 1, 2, 2, 2))
        right = torch.full((1, 1, 2, 2, 2), 3.0)

        stitched = stitch_logits(
            ((left, (0, 0, 0)), (right, (1, 0, 0))),
            BlendMode.CONSTANT,
            output_size=(3, 2, 2),
        )

        expected = torch.empty((1, 1, 3, 2, 2))
        expected[:, :, 0] = 1.0
        expected[:, :, 1] = 2.0
        expected[:, :, 2] = 3.0
        torch.testing.assert_close(stitched, expected)

    def test_constant_blending_reconstructs_split_image(self):
        image = torch.arange(4**3, dtype=torch.float32).reshape(1, 1, 4, 4, 4)
        patches = split_to_patch(image, patch_size=(3, 3, 3), overlap=0.5)

        stitched = stitch_logits(
            patches,
            BlendMode.CONSTANT,
            output_size=image.shape[2:],
        )

        torch.testing.assert_close(stitched, image)

    def test_gaussian_blending_reconstructs_split_image(self):
        image = torch.arange(4**3, dtype=torch.float32).reshape(1, 1, 4, 4, 4)
        patches = split_to_patch(image, patch_size=(3, 3, 3), overlap=0.5)

        stitched = stitch_logits(
            patches,
            BlendMode.GAUSSIAN,
            output_size=image.shape[2:],
        )

        torch.testing.assert_close(stitched, image)

    def test_infers_output_size_from_patch_extents(self):
        first = torch.ones((1, 2, 2, 3, 4))
        second = torch.full((1, 2, 2, 3, 4), 2.0)

        stitched = stitch_logits(
            ((first, (0, 0, 0)), (second, (2, 1, 1))),
            BlendMode.CONSTANT,
        )

        self.assertEqual(stitched.shape, (1, 2, 4, 4, 5))


if __name__ == "__main__":
    unittest.main()
