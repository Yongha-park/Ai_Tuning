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
- NV12 파일 선택
- Width/Height/ISO/Exposure/Noise Scale 입력
- 전처리 실행 후 `(1, 5, H, W)` 텐서를 `npu_input_tensor.npy`로 저장
- 채널 순서: `[Y, U, V, Noise_Map, ISO_Map]`

## 참고
- 기존 `streamlit_app.py`는 웹 UI 실험용 파일이며, 사용자 요청 기준 기본 사용 경로는 `desktop_tool.py`입니다.
