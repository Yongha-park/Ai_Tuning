import importlib.util
import sys
import tempfile
from pathlib import Path


def _missing_modules() -> list[str]:
    required = ["numpy", "streamlit", "cv2"]
    return [name for name in required if importlib.util.find_spec(name) is None]


missing = _missing_modules()
if missing:
    print(
        "[오류] 필수 모듈이 설치되어 있지 않습니다:\n"
        f"  - {', '.join(missing)}\n"
        "다음 명령으로 설치 후 다시 실행하세요:\n"
        "  pip install -r requirements.txt\n"
        "또는\n"
        "  pip install numpy opencv-python streamlit\n"
        "실행 명령:\n"
        "  streamlit run streamlit_app.py"
    )
    sys.exit(1)

import numpy as np
import streamlit as st

from nv12_preprocess import NV12ToNPUTensorConverter


st.set_page_config(page_title="NV12 → NPU Tensor", layout="wide")
st.title("NV12 → NPU 입력 텐서 전처리 UI")
st.caption("HW ISP NV12 + 메타데이터(ISO/Exposure) → (1, 5, H, W) 텐서")


def validate_nv12_size(file_size: int, width: int, height: int) -> tuple[bool, int]:
    expected = width * height * 3 // 2
    return file_size == expected, expected


with st.sidebar:
    st.header("입력 파라미터")
    width = st.number_input("Width", min_value=2, value=640, step=2)
    height = st.number_input("Height", min_value=2, value=480, step=2)
    iso = st.number_input("ISO", min_value=0.0, value=800.0, step=1.0)
    exposure_time = st.number_input("Exposure Time (ms)", min_value=0.0, value=10.0, step=0.1)
    noise_scale = st.number_input("Noise Scale", min_value=0.0, value=1e-4, format="%.6f")

    st.divider()
    uploaded = st.file_uploader("NV12 파일 업로드", type=None)
    run_button = st.button("Preprocess 실행", type="primary")

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("입력 검증")
    if width % 2 != 0 or height % 2 != 0:
        st.error("NV12 형식상 width/height는 짝수여야 합니다.")

    if uploaded is not None:
        valid_size, expected_size = validate_nv12_size(uploaded.size, int(width), int(height))
        st.write(f"업로드 파일 크기: **{uploaded.size} bytes**")
        st.write(f"예상 NV12 크기: **{expected_size} bytes**")
        if valid_size:
            st.success("파일 크기 검증 통과")
        else:
            st.error("파일 크기가 NV12 예상 값과 다릅니다.")
    else:
        st.info("NV12 파일을 업로드한 뒤 실행하세요.")

with col2:
    st.subheader("출력 결과")
    if run_button:
        if uploaded is None:
            st.warning("먼저 NV12 파일을 업로드해 주세요.")
        elif width % 2 != 0 or height % 2 != 0:
            st.error("width/height를 짝수로 설정해 주세요.")
        else:
            try:
                with tempfile.TemporaryDirectory() as tmp_dir:
                    input_path = Path(tmp_dir) / "input.nv12"
                    input_path.write_bytes(uploaded.getbuffer())

                    converter = NV12ToNPUTensorConverter(noise_scale=float(noise_scale))
                    tensor = converter.convert(
                        image_path=str(input_path),
                        width=int(width),
                        height=int(height),
                        iso=float(iso),
                        exposure_time=float(exposure_time),
                    )

                st.success("전처리 완료")
                st.write(f"Shape: **{tensor.shape}**")
                st.write(f"Dtype: **{tensor.dtype}**")
                st.write("채널 순서: **[Y, U, V, Noise_Map, ISO_Map]**")

                channel_names = ["Y", "U", "V", "Noise_Map", "ISO_Map"]
                tabs = st.tabs(channel_names)
                for i, name in enumerate(channel_names):
                    ch = tensor[0, i]
                    ch_min, ch_max = float(ch.min()), float(ch.max())
                    with tabs[i]:
                        st.write(f"min={ch_min:.6f}, max={ch_max:.6f}, mean={float(ch.mean()):.6f}")
                        if ch_max > ch_min:
                            vis = (ch - ch_min) / (ch_max - ch_min)
                        else:
                            vis = np.zeros_like(ch, dtype=np.float32)
                        st.image(vis, caption=f"{name} 시각화", clamp=True)

            except Exception as exc:
                st.error(f"실행 중 오류 발생: {exc}")
