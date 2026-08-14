#!/usr/bin/env python3
"""Live Aria Gen 2 microphone -> MimicMetrics audio model -> Rerun."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch
from torch import nn


EMOTIONS = (
    "Admiration",
    "Amusement",
    "Determination",
    "Empathic Pain",
    "Excitement",
    "Joy",
)
COLORS = (
    (127, 180, 255),
    (255, 196, 87),
    (174, 112, 255),
    (255, 113, 128),
    (184, 255, 44),
    (255, 139, 61),
)


def attention_pool(x: torch.Tensor, mask: torch.Tensor, attn: nn.Linear) -> torch.Tensor:
    scores = attn(x).squeeze(-1).masked_fill(~mask, -1e9)
    return torch.bmm(torch.softmax(scores, dim=1).unsqueeze(1), x).squeeze(1)


class AudioAttentionEncoder(nn.Module):
    """Audio-only MimicMetrics prediction head used during unimodal training."""

    def __init__(self, input_dim=1024, hidden_dim=384, num_labels=6, dropout=0.45):
        super().__init__()
        self.attn = nn.Linear(input_dim, 1)
        self.proj = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_dim),
        )
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden_dim, num_labels))

    def forward(self, audio: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return self.head(self.proj(attention_pool(audio, mask, self.attn)))


class LiveAudioModel:
    def __init__(self, model_dir: Path, device: torch.device):
        checkpoint_path = model_dir / "audio_encoder_best.pt"
        extractor_path = model_dir / "audio_feature_extractor.pt"
        missing = [str(path) for path in (checkpoint_path, extractor_path) if not path.exists()]
        if missing:
            raise FileNotFoundError("Missing model file(s): " + ", ".join(missing))

        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        state = checkpoint.get("model_state_dict", checkpoint)
        input_dim = int(checkpoint.get("audio_input_dim", 1024))
        hidden_dim = int(checkpoint.get("audio_hidden", 384))
        dropout = float(checkpoint.get("dropout", 0.45))

        self.regressor = AudioAttentionEncoder(input_dim, hidden_dim, 6, dropout).to(device)
        self.regressor.load_state_dict(state)
        self.regressor.eval()
        self.extractor = torch.jit.load(str(extractor_path), map_location=device).eval()
        self.device = device

    @torch.inference_mode()
    def predict(self, waveform: np.ndarray) -> np.ndarray:
        audio = torch.from_numpy(waveform).float().to(self.device).unsqueeze(0)
        features = self.extractor(audio)
        if isinstance(features, (tuple, list)):
            features = features[0]
        if isinstance(features, dict):
            features = features.get("last_hidden_state", next(iter(features.values())))
        if features.ndim == 2:
            features = features.unsqueeze(0)
        if features.ndim != 3:
            raise ValueError(f"Feature extractor must return [B,T,D] or [T,D], got {tuple(features.shape)}")
        mask = torch.ones(features.shape[:2], dtype=torch.bool, device=self.device)
        values = self.regressor(features.float(), mask).squeeze(0)
        return values.clamp(0.0, 1.0).cpu().numpy()


class AriaMicrophone:
    def __init__(self, seconds: float, sample_rate: int, channel: int):
        self.sample_rate = sample_rate
        self.channel = channel
        self.samples = deque(maxlen=int(seconds * sample_rate * 2))
        self.last_audio_at = 0.0
        self.receiver = None

    def _sample_rate(self, record) -> int:
        timestamps = getattr(record, "capture_timestamps_ns", None)
        if timestamps is None or len(timestamps) < 2:
            return self.sample_rate
        diffs = np.diff(np.asarray(timestamps, dtype=np.float64))
        diffs = diffs[diffs > 0]
        if len(diffs):
            estimated = int(round(1e9 / float(np.median(diffs))))
            if 4_000 <= estimated <= 192_000:
                return estimated
        return self.sample_rate

    def _callback(self, audio_data, audio_record, num_channels: int, device_id=None):
        raw = np.asarray(audio_data.data)
        channels = max(int(num_channels or 1), 1)
        usable = raw[: raw.size - (raw.size % channels)] if raw.size % channels else raw
        if usable.size == 0:
            return
        samples = np.stack(np.array_split(usable.astype(np.float32), channels), axis=1)
        scale = float(getattr(audio_data, "max_amplitude", 0.0) or np.finfo(np.float32).max)
        samples /= scale
        rms = np.sqrt(np.mean(np.square(samples), axis=0) + 1e-12)
        channel = self.channel if 0 <= self.channel < channels else int(np.argmax(rms))
        mono = samples[:, channel]
        source_rate = self._sample_rate(audio_record)
        if source_rate != self.sample_rate and len(mono) > 1:
            length = max(1, round(len(mono) * self.sample_rate / source_rate))
            mono = np.interp(
                np.linspace(0, 1, length, endpoint=False),
                np.linspace(0, 1, len(mono), endpoint=False),
                mono,
            ).astype(np.float32)
        self.samples.extend(mono.tolist())
        self.last_audio_at = time.time()

    def start(self, address: str, port: int):
        import aria.sdk_gen2 as sdk_gen2
        import aria.stream_receiver as stream_receiver

        config = sdk_gen2.HttpServerConfig()
        config.address = address
        config.port = port
        self.receiver = stream_receiver.StreamReceiver(enable_image_decoding=False, enable_raw_stream=False)
        self.receiver.set_server_config(config)
        self.receiver.register_audio_callback(self._callback)
        self.receiver.start_server()

    def window(self, seconds: float) -> np.ndarray | None:
        count = int(seconds * self.sample_rate)
        if len(self.samples) < count:
            return None
        return np.asarray(list(self.samples)[-count:], dtype=np.float32)

    def stop(self):
        if self.receiver is not None:
            self.receiver.stop_server()


def aria_cli() -> str:
    found = shutil.which("aria_gen2")
    if not found:
        raise FileNotFoundError("aria_gen2 was not found. Run with the Aria Gen 2 environment.")
    return found


def run_cli(*args: str, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run([aria_cli(), *args], text=True, check=check)


def choose_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def log_to_rerun(values: np.ndarray, step: int):
    import rerun as rr

    rr.set_time("inference_step", sequence=step)
    rr.log("emotion_intensity/current", rr.BarChart(values.tolist()))
    for emotion, value, color in zip(EMOTIONS, values, COLORS):
        path = f"emotion_intensity/history/{emotion.lower().replace(' ', '_')}"
        rr.log(path, rr.Scalars(float(value)))
        if step == 0:
            rr.log(path, rr.SeriesLines(names=emotion, colors=[color]), static=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=Path("models"))
    parser.add_argument("--window-seconds", type=float, default=4.0)
    parser.add_argument("--interval", type=float, default=0.5)
    parser.add_argument("--sample-rate", type=int, default=16_000)
    parser.add_argument("--audio-channel", type=int, default=-1, help="-1 selects the loudest channel")
    parser.add_argument("--device", default="auto", help="auto, cpu, or mps")
    parser.add_argument("--receiver-address", default="0.0.0.0")
    parser.add_argument("--receiver-port", type=int, default=6768)
    args = parser.parse_args()

    device = choose_device(args.device)
    print(f"Loading audio-only MimicMetrics model on {device}…")
    model = LiveAudioModel(args.model_dir, device)
    microphone = AriaMicrophone(args.window_seconds, args.sample_rate, args.audio_channel)

    try:
        print("Starting Aria audio receiver…")
        microphone.start(args.receiver_address, args.receiver_port)
        run_cli("device", "list")
        run_cli("streaming", "start")

        import rerun as rr
        rr.init("MimicMetrics · Aria audio emotion intensity", spawn=True)
        rr.log("status", rr.TextLog("Aria microphone connected. Waiting for the first audio window."))
        print("Rerun opened. Speak naturally; press Ctrl+C to stop.")

        step = 0
        while True:
            waveform = microphone.window(args.window_seconds)
            if waveform is not None:
                values = model.predict(waveform)
                log_to_rerun(values, step)
                strongest = int(np.argmax(values))
                print(f"{EMOTIONS[strongest]} {values[strongest]:.2f}")
                step += 1
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("Stopping…")
    finally:
        run_cli("streaming", "stop", check=False)
        microphone.stop()
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise
