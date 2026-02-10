# Ai_Tuning

## 실행 방법

### 1) 의존성 설치
```bash
pip install -r requirements.txt
```

### 2) Streamlit UI 실행
```bash
streamlit run streamlit_app.py
```

## 문제 해결
- `ModuleNotFoundError: No module named 'streamlit'` 발생 시:
  1. `pip install streamlit` 실행
  2. 이후 `streamlit run streamlit_app.py`로 실행

- `missing ScriptRunContext` 경고가 보일 때:
  - 원인: `python streamlit_app.py`로 직접 실행한 경우
  - 해결: 아래 명령으로 실행
    ```bash
    streamlit run streamlit_app.py
    ```
