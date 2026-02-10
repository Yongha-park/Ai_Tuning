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
    """Convert NV12/P010 frame + metadata into NCHW tensor for NPU input."""

    def __init__(self, noise_scale: float = 1e-4) -> None:
        self.noise_scale = float(noise_scale)

    def convert(
        self,
        image_path: str,
        width: int,
        height: int,
        iso: float,
        exposure_time: float,
        pixel_format: str = "auto",
        frame_index: int = 0,
    ) -> np.ndarray:
        """
        Args:
            image_path: Raw NV12/P010 file path.
            width: Frame width.
            height: Frame height.
            iso: ISO / analog gain scalar.
            exposure_time: Exposure time in milliseconds.
            pixel_format: "auto", "nv12", or "p010".
            frame_index: Frame index for multi-frame raw files.

        Returns:
            output tensor with shape (1, 5, H, W), dtype float32.
            channel order: [Y, U, V, Noise_Map, ISO_Map]
        """
        metadata = ISPMetadata(iso=float(iso), exposure_time_ms=float(exposure_time))

        y_norm, uv_norm = self._load_luma_chroma_planes(
            image_path=image_path,
            width=width,
            height=height,
            pixel_format=pixel_format,
            frame_index=frame_index,
        )
        u_norm, v_norm = self._deinterleave_and_upsample_uv(uv_norm, width, height)

        # Broadcast scalar metadata into feature maps with shape (H, W)
        noise_map = np.full((height, width), metadata.iso * self.noise_scale, dtype=np.float32)
        iso_map = np.full((height, width), metadata.iso, dtype=np.float32)

        # Stack channels: (H, W, 5)
        hwc = np.stack([y_norm, u_norm, v_norm, noise_map, iso_map], axis=-1)

        # Rearrange to NCHW: (1, 5, H, W)
        nchw = np.transpose(hwc, (2, 0, 1))[np.newaxis, ...].astype(np.float32)
        return nchw

    @staticmethod
    def _load_luma_chroma_planes(
        image_path: str,
        width: int,
        height: int,
        pixel_format: str,
        frame_index: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        if frame_index < 0:
            raise ValueError("frame_index must be >= 0")

        raw = np.fromfile(image_path, dtype=np.uint8)
        raw_size = raw.size

        nv12_frame_bytes = width * height * 3 // 2
        p010_frame_bytes = width * height * 3

        fmt = pixel_format.lower().strip()
        if fmt not in {"auto", "nv12", "p010"}:
            raise ValueError("pixel_format must be one of: auto, nv12, p010")

        if fmt == "auto":
            nv12_ok = raw_size % nv12_frame_bytes == 0
            p010_ok = raw_size % p010_frame_bytes == 0

            if nv12_ok and p010_ok:
                raise ValueError(
                    "Ambiguous raw size: file matches both multi-frame NV12 and P010. "
                    "Please set pixel_format explicitly to 'nv12' or 'p010'."
                )
            if nv12_ok:
                fmt = "nv12"
            elif p010_ok:
                fmt = "p010"
            else:
                raise ValueError(
                    f"Invalid raw size for both NV12/P010. size={raw_size}, "
                    f"nv12_frame={nv12_frame_bytes}, p010_frame={p010_frame_bytes}, path={image_path}"
                )

        if fmt == "nv12":
            if raw_size % nv12_frame_bytes != 0:
                raise ValueError(
                    f"Invalid NV12 size. expected multiple of {nv12_frame_bytes}, got={raw_size}, path={image_path}"
                )
            frame_count = raw_size // nv12_frame_bytes
            if frame_index >= frame_count:
                raise ValueError(f"frame_index out of range. frame_count={frame_count}, frame_index={frame_index}")

            start = frame_index * nv12_frame_bytes
            end = start + nv12_frame_bytes
            frame = raw[start:end]

            y_size = width * height
            y_plane = frame[:y_size].reshape(height, width).astype(np.float32) / 255.0
            uv_plane = frame[y_size:].reshape(height // 2, width).astype(np.float32) / 255.0
            return y_plane, uv_plane

        # p010 path (16-bit container, usually 10-bit valid)
        if raw_size % p010_frame_bytes != 0:
            raise ValueError(
                f"Invalid P010 size. expected multiple of {p010_frame_bytes}, got={raw_size}, path={image_path}"
            )
        frame_count = raw_size // p010_frame_bytes
        if frame_index >= frame_count:
            raise ValueError(f"frame_index out of range. frame_count={frame_count}, frame_index={frame_index}")

        frame_samples_u16 = width * height * 3 // 2
        all_u16 = np.fromfile(image_path, dtype=np.uint16)
        start_s = frame_index * frame_samples_u16
        end_s = start_s + frame_samples_u16
        frame_u16 = all_u16[start_s:end_s]

        y_size_s = width * height
        y_u16 = frame_u16[:y_size_s].reshape(height, width)
        uv_u16 = frame_u16[y_size_s:].reshape(height // 2, width)

        # P010 often stores 10-bit values in MSBs of 16-bit (value << 6)
        if int(y_u16.max(initial=0)) > 1023 or int(uv_u16.max(initial=0)) > 1023:
            y_10 = (y_u16 >> 6).astype(np.float32)
            uv_10 = (uv_u16 >> 6).astype(np.float32)
        else:
            y_10 = y_u16.astype(np.float32)
            uv_10 = uv_u16.astype(np.float32)

        y_norm = np.clip(y_10 / 1023.0, 0.0, 1.0)
        uv_norm = np.clip(uv_10 / 1023.0, 0.0, 1.0)
        return y_norm, uv_norm

    @staticmethod
    def _deinterleave_and_upsample_uv(
        uv_plane: np.ndarray, width: int, height: int
    ) -> tuple[np.ndarray, np.ndarray]:
        # De-interleave UVUV... -> U and V, each shape: (H/2, W/2)
        u_half = uv_plane[:, 0::2]
        v_half = uv_plane[:, 1::2]

        # Bilinear upsampling to match Y plane resolution: (H, W)
        u_full = cv2.resize(u_half, (width, height), interpolation=cv2.INTER_LINEAR).astype(np.float32)
        v_full = cv2.resize(v_half, (width, height), interpolation=cv2.INTER_LINEAR).astype(np.float32)
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
            pixel_format="nv12",
            frame_index=0,
        )

        print("Output tensor shape:", tensor.shape)
        print("Output tensor dtype:", tensor.dtype)
        print("Channel order: [Y, U, V, Noise_Map, ISO_Map]")
