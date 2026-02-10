import os
import tempfile
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class ISPMetadata:
    """Container for scalar metadata used by the neural ISP."""

    iso: float
    exposure_time_ms: float


class NV12ToNPUTensorConverter:
    """Convert NV12 frame + metadata into NCHW tensor for NPU input."""

    def __init__(self, noise_scale: float = 1e-4) -> None:
        self.noise_scale = float(noise_scale)

    def convert(
        self,
        image_path: str,
        width: int,
        height: int,
        iso: float,
        exposure_time: float,
    ) -> np.ndarray:
        """
        Args:
            image_path: Raw NV12 file path.
            width: Frame width.
            height: Frame height.
            iso: ISO / analog gain scalar.
            exposure_time: Exposure time in milliseconds.

        Returns:
            output tensor with shape (1, 5, H, W), dtype float32.
            channel order: [Y, U, V, Noise_Map, ISO_Map]
        """
        metadata = ISPMetadata(iso=float(iso), exposure_time_ms=float(exposure_time))

        y_plane, uv_plane = self._load_nv12_planes(image_path, width, height)
        u_full, v_full = self._deinterleave_and_upsample_uv(uv_plane, width, height)

        # Normalize all image channels to [0, 1], shape remains (H, W)
        y_norm = y_plane.astype(np.float32) / 255.0
        u_norm = u_full.astype(np.float32) / 255.0
        v_norm = v_full.astype(np.float32) / 255.0

        # Broadcast scalar metadata into feature maps with shape (H, W)
        noise_map = np.full((height, width), metadata.iso * self.noise_scale, dtype=np.float32)
        iso_map = np.full((height, width), metadata.iso, dtype=np.float32)

        # Stack channels: (H, W, 5)
        hwc = np.stack([y_norm, u_norm, v_norm, noise_map, iso_map], axis=-1)

        # Rearrange to NCHW: (1, 5, H, W)
        nchw = np.transpose(hwc, (2, 0, 1))[np.newaxis, ...].astype(np.float32)
        return nchw

    @staticmethod
    def _load_nv12_planes(image_path: str, width: int, height: int) -> tuple[np.ndarray, np.ndarray]:
        expected_size = width * height * 3 // 2
        raw = np.fromfile(image_path, dtype=np.uint8)

        if raw.size != expected_size:
            raise ValueError(
                f"Invalid NV12 size. expected={expected_size}, got={raw.size}, "
                f"path={image_path}"
            )

        y_size = width * height
        y_plane = raw[:y_size].reshape(height, width)

        # NV12 UV plane has half vertical resolution and interleaved UV pairs
        uv_plane = raw[y_size:].reshape(height // 2, width)
        return y_plane, uv_plane

    @staticmethod
    def _deinterleave_and_upsample_uv(
        uv_plane: np.ndarray, width: int, height: int
    ) -> tuple[np.ndarray, np.ndarray]:
        # De-interleave UVUV... -> U and V, each shape: (H/2, W/2)
        u_half = uv_plane[:, 0::2]
        v_half = uv_plane[:, 1::2]

        # Bilinear upsampling to match Y plane resolution: (H, W)
        u_full = cv2.resize(u_half, (width, height), interpolation=cv2.INTER_LINEAR)
        v_full = cv2.resize(v_half, (width, height), interpolation=cv2.INTER_LINEAR)
        return u_full, v_full


def _generate_dummy_nv12(path: str, width: int, height: int) -> None:
    """Generate deterministic dummy NV12 data for local testing."""
    y = np.random.randint(0, 256, size=(height, width), dtype=np.uint8)
    uv = np.random.randint(0, 256, size=(height // 2, width), dtype=np.uint8)
    nv12 = np.concatenate([y.flatten(), uv.flatten()])
    nv12.tofile(path)


if __name__ == "__main__":
    np.random.seed(42)

    w, h = 640, 480
    iso = 800.0
    exposure_ms = 10.0

    with tempfile.TemporaryDirectory() as tmp_dir:
        nv12_path = os.path.join(tmp_dir, "dummy.nv12")
        _generate_dummy_nv12(nv12_path, w, h)

        converter = NV12ToNPUTensorConverter(noise_scale=1e-4)
        tensor = converter.convert(
            image_path=nv12_path,
            width=w,
            height=h,
            iso=iso,
            exposure_time=exposure_ms,
        )

        print("Output tensor shape:", tensor.shape)
        print("Output tensor dtype:", tensor.dtype)
        print("Channel order: [Y, U, V, Noise_Map, ISO_Map]")
