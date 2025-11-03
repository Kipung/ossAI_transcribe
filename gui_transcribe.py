#!/usr/bin/env python3
import sys, os, traceback
from datetime import timedelta
from pathlib import Path

from PySide6.QtCore import QThread, Signal
from PySide6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QPushButton, QFileDialog,
    QLabel, QTextEdit, QHBoxLayout, QComboBox, QCheckBox, QLineEdit
)

from faster_whisper import WhisperModel

DEFAULT_OUT = Path.home() / "WhisperLite"

def app_root() -> Path:
    # Works for dev and PyInstaller onefile/onedir
    return Path(getattr(sys, "_MEIPASS", Path(__file__).parent))

# Expected bundled path: models/whisper-tiny/* (CTranslate2 files)
LOCAL_MODEL_DIR = app_root() / "models" / "whisper-tiny"

def srt_ts(s):
    td = timedelta(seconds=s)
    return str(td)[:-3].replace('.', ',')

def write_txt(path, segments):
    with open(path, "w", encoding="utf-8") as f:
        for seg in segments:
            f.write(seg.text.strip() + "\n")

def write_srt(path, segments):
    with open(path, "w", encoding="utf-8") as f:
        for i, seg in enumerate(segments, 1):
            f.write(f"{i}\n{srt_ts(seg.start)} --> {srt_ts(seg.end)}\n{seg.text.strip()}\n\n")

def write_vtt(path, segments):
    def vtt_ts(s):
        td = timedelta(seconds=s)
        return str(td)[:-3]  # 00:00:00.000
    with open(path, "w", encoding="utf-8") as f:
        f.write("WEBVTT\n\n")
        for seg in segments:
            f.write(f"{vtt_ts(seg.start)} --> {vtt_ts(seg.end)}\n{seg.text.strip()}\n\n")

class TranscribeWorker(QThread):
    progress = Signal(str)
    done = Signal(str)
    failed = Signal(str)

    def __init__(self, audio_path, out_dir, model_size, language,
                 write_srt_flag, write_vtt_flag, vad_flag, device="auto"):
        super().__init__()
        self.audio_path = audio_path
        self.out_dir = Path(out_dir)
        self.model_size = model_size      # e.g., "tiny"
        self.language = language or None
        self.write_srt_flag = write_srt_flag
        self.write_vtt_flag = write_vtt_flag
        self.vad_flag = vad_flag
        self.device = device if device in ("cpu", "cuda", "auto") else "auto"

    def _load_model(self):
        chosen_device = self.device or "auto"
        compute = "float16" if chosen_device == "cuda" else "int8"

        # Prefer bundled local model if available
        if LOCAL_MODEL_DIR.exists() and any(LOCAL_MODEL_DIR.iterdir()):
            self.progress.emit(f"Loading local model: {LOCAL_MODEL_DIR}")
            # Force offline to avoid any accidental network download
            os.environ["HF_HUB_OFFLINE"] = "1"
            return WhisperModel(
                str(LOCAL_MODEL_DIR),
                device=chosen_device,
                compute_type=compute
            )

        # Fallback to model name; may download on first run if cache missing
        self.progress.emit(f"No local model found. Loading by name: {self.model_size}")
        # Do NOT set HF_HUB_OFFLINE here; allow download if online
        return WhisperModel(
            self.model_size,
            device=chosen_device,
            compute_type=compute
        )

    def run(self):
        try:
            # resolve a writable output dir
            out = self.out_dir
            if str(out) in ("/", ""):
                out = DEFAULT_OUT
            out.mkdir(parents=True, exist_ok=True)

            base = Path(self.audio_path).stem
            txt_path = out / f"{base}.txt"
            srt_path = out / f"{base}.srt"
            vtt_path = out / f"{base}.vtt"

            self.progress.emit(f"Loading model: {self.model_size}")
            model = self._load_model()

            self.progress.emit(f"Transcribing: {self.audio_path}")
            segments, info = model.transcribe(
                self.audio_path,
                language=self.language,
                beam_size=5,
                vad_filter=self.vad_flag,   # toggle VAD
                word_timestamps=False,
            )

            segs = list(segments)
            self.progress.emit(f"Detected language: {info.language} (p={info.language_probability:.3f})")
            self.progress.emit(f"Writing outputs to: {out}")

            # write; if OSError (e.g., read-only), fall back to DEFAULT_OUT
            try:
                write_txt(txt_path, segs)
                if self.write_srt_flag:
                    write_srt(srt_path, segs)
                if self.write_vtt_flag:
                    write_vtt(vtt_path, segs)
                wrote = [str(txt_path)]
                if self.write_srt_flag: wrote.append(str(srt_path))
                if self.write_vtt_flag: wrote.append(str(vtt_path))
                self.done.emit("Wrote: " + ", ".join(wrote))
            except OSError:
                fb = DEFAULT_OUT
                fb.mkdir(parents=True, exist_ok=True)
                txt_fb = fb / f"{base}.txt"
                write_txt(txt_fb, segs)
                msg = f"Wrote (fallback): {txt_fb}"
                if self.write_srt_flag:
                    srt_fb = fb / f"{base}.srt"; write_srt(srt_fb, segs); msg += f", {srt_fb}"
                if self.write_vtt_flag:
                    vtt_fb = fb / f"{base}.vtt"; write_vtt(vtt_fb, segs); msg += f", {vtt_fb}"
                self.done.emit(msg)

        except Exception as e:
            tb = traceback.format_exc()
            self.failed.emit(f"{e}\n{tb}")

class App(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("WhisperLite")
        lay = QVBoxLayout(self)

        # Row: file choose
        r1 = QHBoxLayout()
        self.file_label = QLabel("No file selected")
        btn_choose = QPushButton("Select audio/video")
        btn_choose.clicked.connect(self.choose_file)
        r1.addWidget(self.file_label); r1.addWidget(btn_choose)
        lay.addLayout(r1)

        # Row: output dir
        r2 = QHBoxLayout()
        self.out_label = QLabel(f"Output: {DEFAULT_OUT}")
        btn_out = QPushButton("Choose output folder")
        btn_out.clicked.connect(self.choose_out)
        r2.addWidget(self.out_label); r2.addWidget(btn_out)
        lay.addLayout(r2)

        # Row: model + language
        r3 = QHBoxLayout()
        r3.addWidget(QLabel("Model"))
        self.model_box = QComboBox()
        self.model_box.addItems([
            "tiny", "base", "small", "medium", "large-v3",
            "distil-small.en", "distil-medium.en"
        ])
        r3.addWidget(self.model_box)
        r3.addWidget(QLabel("Lang (optional)"))
        self.lang_edit = QLineEdit()
        self.lang_edit.setPlaceholderText("e.g., en, ko, es")
        r3.addWidget(self.lang_edit)
        lay.addLayout(r3)

        # Row: options
        r4 = QHBoxLayout()
        self.chk_srt = QCheckBox("Write SRT"); self.chk_srt.setChecked(True)
        self.chk_vtt = QCheckBox("Write VTT")
        self.chk_vad = QCheckBox("VAD filter"); self.chk_vad.setChecked(True)
        r4.addWidget(self.chk_srt); r4.addWidget(self.chk_vtt); r4.addWidget(self.chk_vad)
        lay.addLayout(r4)

        # Go
        self.btn_run = QPushButton("Transcribe")
        self.btn_run.clicked.connect(self.start_run)
        lay.addWidget(self.btn_run)

        # Log
        self.log = QTextEdit(); self.log.setReadOnly(True)
        lay.addWidget(self.log)

        # State
        self.audio_path = None
        self.user_chose_out_dir = False
        self.out_dir = str(DEFAULT_OUT)
        DEFAULT_OUT.mkdir(parents=True, exist_ok=True)

    def choose_file(self):
        fn, _ = QFileDialog.getOpenFileName(
            self, "Select audio/video", "",
            "Media files (*.wav *.mp3 *.m4a *.flac *.mp4 *.mov);;All files (*)"
        )
        if fn:
            self.audio_path = fn
            self.file_label.setText(Path(fn).name)
            if not self.user_chose_out_dir:
                self.out_dir = str(Path(fn).parent)
                self.out_label.setText(f"Output: {self.out_dir}")

    def choose_out(self):
        d = QFileDialog.getExistingDirectory(self, "Select output folder", self.out_dir)
        if d:
            self.out_dir = d
            self.user_chose_out_dir = True
            self.out_label.setText(f"Output: {d}")

    def start_run(self):
        if not self.audio_path:
            self.log.append("Select a file first.")
            return
        self.btn_run.setEnabled(False)
        self.log.append("Starting...")
        self.worker = TranscribeWorker(
            audio_path=self.audio_path,
            out_dir=self.out_dir,
            model_size=self.model_box.currentText(),
            language=self.lang_edit.text().strip(),
            write_srt_flag=self.chk_srt.isChecked(),
            write_vtt_flag=self.chk_vtt.isChecked(),
            vad_flag=self.chk_vad.isChecked(),
            device="auto",
        )
        self.worker.progress.connect(self.log.append)
        self.worker.done.connect(self.finish_ok)
        self.worker.failed.connect(self.finish_fail)
        self.worker.start()

    def finish_ok(self, msg):
        self.log.append(msg)
        self.btn_run.setEnabled(True)

    def finish_fail(self, msg):
        self.log.append("ERROR:\n" + msg)
        self.btn_run.setEnabled(True)

def main():
    app = QApplication(sys.argv)
    w = App()
    w.resize(700, 500)
    w.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()