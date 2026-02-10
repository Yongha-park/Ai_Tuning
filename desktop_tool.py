import os
import tkinter as tk
from tkinter import filedialog, messagebox

import numpy as np

from nv12_preprocess import NV12ToNPUTensorConverter


class NV12DesktopTool:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("NV12/P010 → NPU 텐서 변환기 (PC 툴)")
        self.root.geometry("700x480")

        self.input_path = tk.StringVar()
        self.output_dir = tk.StringVar(value=os.getcwd())
        self.width = tk.StringVar(value="640")
        self.height = tk.StringVar(value="480")
        self.iso = tk.StringVar(value="800")
        self.exposure = tk.StringVar(value="10.0")
        self.noise_scale = tk.StringVar(value="0.0001")
        self.pixel_format = tk.StringVar(value="auto")
        self.frame_index = tk.StringVar(value="0")

        self._build_ui()

    def _build_ui(self) -> None:
        pad = {"padx": 8, "pady": 6}

        tk.Label(self.root, text="입력 파일 (NV12 / P010)").grid(row=0, column=0, sticky="w", **pad)
        tk.Entry(self.root, textvariable=self.input_path, width=55).grid(row=0, column=1, **pad)
        tk.Button(self.root, text="찾기", command=self._browse_input).grid(row=0, column=2, **pad)

        tk.Label(self.root, text="출력 폴더").grid(row=1, column=0, sticky="w", **pad)
        tk.Entry(self.root, textvariable=self.output_dir, width=55).grid(row=1, column=1, **pad)
        tk.Button(self.root, text="찾기", command=self._browse_output).grid(row=1, column=2, **pad)

        tk.Label(self.root, text="Width").grid(row=2, column=0, sticky="w", **pad)
        tk.Entry(self.root, textvariable=self.width, width=20).grid(row=2, column=1, sticky="w", **pad)

        tk.Label(self.root, text="Height").grid(row=3, column=0, sticky="w", **pad)
        tk.Entry(self.root, textvariable=self.height, width=20).grid(row=3, column=1, sticky="w", **pad)

        tk.Label(self.root, text="ISO").grid(row=4, column=0, sticky="w", **pad)
        tk.Entry(self.root, textvariable=self.iso, width=20).grid(row=4, column=1, sticky="w", **pad)

        tk.Label(self.root, text="Exposure (ms)").grid(row=5, column=0, sticky="w", **pad)
        tk.Entry(self.root, textvariable=self.exposure, width=20).grid(row=5, column=1, sticky="w", **pad)

        tk.Label(self.root, text="Noise Scale").grid(row=6, column=0, sticky="w", **pad)
        tk.Entry(self.root, textvariable=self.noise_scale, width=20).grid(row=6, column=1, sticky="w", **pad)

        tk.Label(self.root, text="Pixel Format").grid(row=7, column=0, sticky="w", **pad)
        tk.OptionMenu(self.root, self.pixel_format, "auto", "nv12", "p010").grid(row=7, column=1, sticky="w", **pad)

        tk.Label(self.root, text="Frame Index").grid(row=8, column=0, sticky="w", **pad)
        tk.Entry(self.root, textvariable=self.frame_index, width=20).grid(row=8, column=1, sticky="w", **pad)

        tk.Button(self.root, text="전처리 실행", command=self._run, bg="#2e7d32", fg="white").grid(
            row=9, column=1, sticky="w", **pad
        )

        self.result_text = tk.Text(self.root, height=11, width=84)
        self.result_text.grid(row=10, column=0, columnspan=3, padx=8, pady=10)
        self.result_text.insert(tk.END, "준비 완료.\n")

    def _browse_input(self) -> None:
        path = filedialog.askopenfilename(title="원본 파일 선택", filetypes=[("Raw", "*.*")])
        if path:
            self.input_path.set(path)

    def _browse_output(self) -> None:
        path = filedialog.askdirectory(title="출력 폴더 선택")
        if path:
            self.output_dir.set(path)

    def _append(self, text: str) -> None:
        self.result_text.insert(tk.END, text + "\n")
        self.result_text.see(tk.END)

    def _run(self) -> None:
        try:
            input_path = self.input_path.get().strip()
            output_dir = self.output_dir.get().strip()
            width = int(self.width.get())
            height = int(self.height.get())
            iso = float(self.iso.get())
            exposure = float(self.exposure.get())
            noise_scale = float(self.noise_scale.get())
            pixel_format = self.pixel_format.get().strip().lower()
            frame_index = int(self.frame_index.get())

            if not input_path or not os.path.isfile(input_path):
                raise ValueError("유효한 입력 파일을 선택해 주세요.")
            if not os.path.isdir(output_dir):
                raise ValueError("유효한 출력 폴더를 선택해 주세요.")
            if width % 2 != 0 or height % 2 != 0:
                raise ValueError("NV12/P010 특성상 width/height는 짝수여야 합니다.")

            converter = NV12ToNPUTensorConverter(noise_scale=noise_scale)
            tensor = converter.convert(
                image_path=input_path,
                width=width,
                height=height,
                iso=iso,
                exposure_time=exposure,
                pixel_format=pixel_format,
                frame_index=frame_index,
            )

            output_npy = os.path.join(output_dir, "npu_input_tensor.npy")
            np.save(output_npy, tensor)

            self._append("전처리 완료")
            self._append(f"저장 경로: {output_npy}")
            self._append(f"Tensor shape: {tensor.shape}")
            self._append(f"Tensor dtype: {tensor.dtype}")
            self._append("채널 순서: [Y, U, V, Noise_Map, ISO_Map]")
            self._append(f"사용 포맷: {pixel_format}, frame_index={frame_index}")

            messagebox.showinfo("성공", "전처리가 완료되었습니다.")

        except Exception as exc:
            self._append(f"오류: {exc}")
            self._append("힌트: 파일 크기가 2배면 Pixel Format을 p010으로 바꾸거나 frame_index를 확인하세요.")
            messagebox.showerror("실패", str(exc))


def main() -> None:
    root = tk.Tk()
    NV12DesktopTool(root)
    root.mainloop()


if __name__ == "__main__":
    main()
