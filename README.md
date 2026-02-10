# Ai_Tuning

## PC용 데스크톱 툴 실행 방법 (권장)

### 1) 의존성 설치
```bash
pip install -r requirements.txt
```

### 2) 데스크톱 툴 실행
```bash
python desktop_tool.py
```

### 기능
- 원본 파일(NV12/P010) 선택
- Width/Height/ISO/Exposure/Noise Scale 입력
- Pixel Format(`auto`/`nv12`/`p010`) 및 `frame_index` 설정
- 전처리 실행 후 `(1, 5, H, W)` 텐서를 `npu_input_tensor.npy`로 저장
- 채널 순서: `[Y, U, V, Noise_Map, ISO_Map]`

## 문제 해결
- `Invalid NV12 size ... got=...` 에러인데 파일 크기가 맞는 것 같을 때:
  - 파일이 **P010(16-bit 컨테이너)** 이거나, **여러 프레임이 붙은 NV12**일 수 있습니다.
  - 툴에서 `Pixel Format`을 `p010`으로 바꿔보세요.
  - 또는 `Pixel Format=nv12`로 두고 `frame_index`를 0,1,...로 바꿔보세요.
  - `auto`에서 크기가 양쪽에 모두 맞는 경우(모호한 경우)는 포맷을 명시해야 합니다.

## 참고
- `streamlit_app.py`는 웹 UI 실험용 파일입니다.
