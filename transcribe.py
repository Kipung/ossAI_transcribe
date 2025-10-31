#!/usr/bin/env python3
import argparse, os, sys, platform
from datetime import timedelta
from faster_whisper import WhisperModel

# ---------- Time formatting ----------
def _format_hhmmss_ms(seconds: float, sep: str) -> str:
    # Always 00:00:00{sep}000
    total_ms = int(round(seconds * 1000))
    if total_ms < 0:
        total_ms = 0
    hh, rem = divmod(total_ms, 3600_000)
    mm, rem = divmod(rem, 60_000)
    ss, ms = divmod(rem, 1000)
    return f"{hh:02d}:{mm:02d}:{ss:02d}{sep}{ms:03d}"

def srt_ts(s: float) -> str:
    return _format_hhmmss_ms(s, sep=",")

def vtt_ts(s: float) -> str:
    return _format_hhmmss_ms(s, sep=".")

# ---------- Writers ----------
def write_txt(path, segments):
    with open(path, "w", encoding="utf-8") as f:
        for seg in segments:
            f.write(seg.text.strip() + "\n")

def write_srt(path, segments):
    with open(path, "w", encoding="utf-8") as f:
        for i, seg in enumerate(segments, 1):
            f.write(f"{i}\n{srt_ts(seg.start)} --> {srt_ts(seg.end)}\n{seg.text.strip()}\n\n")

def write_vtt(path, segments):
    with open(path, "w", encoding="utf-8") as f:
        f.write("WEBVTT\n\n")
        for seg in segments:
            f.write(f"{vtt_ts(seg.start)} --> {vtt_ts(seg.end)}\n{seg.text.strip()}\n\n")

# ---------- Main ----------
def main():
    p = argparse.ArgumentParser(description="Transcribe audio with faster-whisper", conflict_handler="resolve")
    p.add_argument("audio", help="Path to audio/video file")
    p.add_argument("--model", default="small", help="tiny, base, small, medium, large-v3, distil-*. Default: small")
    p.add_argument("--language", default=None, help="Force language code, e.g., en, ko, es")
    p.add_argument("--beam-size", type=int, default=5)
    p.add_argument("--vad-filter", action="store_true", help="Enable basic VAD filtering")
    p.add_argument("--output", default=None, help="Output basename without extension")
    p.add_argument("--srt", action="store_true", help="Write SRT subtitles")
    p.add_argument("--vtt", action="store_true", help="Write WebVTT subtitles")
    p.add_argument("--timestamps", action="store_true", help="Print per-segment timestamps")

    # Single definitions only
    p.add_argument("--device", choices=["cpu", "metal", "cuda", "auto"], default="metal",
                   help="metal for Apple Silicon, cuda for NVIDIA, cpu otherwise")
    p.add_argument("--compute-type", default="float16",
                   help="Precision: default|float16|int8|int8_float16|int16")
    p.add_argument("--num-workers", type=int, default=1, help="Batch concurrency (maps to inter_threads)")
    p.add_argument("--cpu-threads", type=int, default=0, help="Intra-op threads (maps to intra_threads)")

    args = p.parse_args()

    # Resolve "auto" device
    if args.device == "auto":
        is_arm_mac = (sys.platform == "darwin" and platform.machine().lower() in {"arm64", "aarch64"})
        args.device = "metal" if is_arm_mac else "cpu"

    # Guard invalid CPU + float16 combo
    if args.device == "cpu" and args.compute_type.lower().startswith("float16"):
        args.compute_type = "int8"

    base = args.output or os.path.splitext(os.path.basename(args.audio))[0]
    txt_path, srt_path, vtt_path = base + ".txt", base + ".srt", base + ".vtt"

    # Load model (use high-level threading knobs only)
    model = WhisperModel(
        args.model,
        device=args.device,
        compute_type=args.compute_type,
        device_index=0,
        num_workers=args.num_workers,
        cpu_threads=args.cpu_threads,
    )

    segments_iter, info = model.transcribe(
        args.audio,
        language=args.language,
        beam_size=args.beam_size,
        vad_filter=args.vad_filter,
        word_timestamps=False,
    )

    # Materialize segments
    segments = list(segments_iter)

    if args.timestamps:
        for seg in segments:
            print(f"[{srt_ts(seg.start)} -> {srt_ts(seg.end)}] {seg.text.strip()}")

    write_txt(txt_path, segments)
    if args.srt:
        write_srt(srt_path, segments)
    if args.vtt:
        write_vtt(vtt_path, segments)

    # Summary
    lang = getattr(info, "language", None)
    lang_prob = getattr(info, "language_probability", None)
    duration = getattr(info, "duration", None)
    if lang is not None and lang_prob is not None:
        print(f"Language: {lang} | Prob: {lang_prob:.3f}")
    if duration is not None:
        print(f"Audio duration: {duration/60.0:.1f} min")
    wrote = [txt_path]
    if args.srt: wrote.append(srt_path)
    if args.vtt: wrote.append(vtt_path)
    print("Wrote: " + ", ".join(wrote))

if __name__ == "__main__":
    main()