"""Central configuration: every magic number lives here so experiments are reproducible."""
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# RAVDESS filename code (3rd field) -> emotion label
EMOTIONS = {
    "01": "neutral", "02": "calm", "03": "happy", "04": "sad",
    "05": "angry", "06": "fearful", "07": "disgust", "08": "surprised",
}
LABELS = list(EMOTIONS.values())          # fixed class order used everywhere
LABEL2ID = {l: i for i, l in enumerate(LABELS)}

RAVDESS_URL = "https://zenodo.org/records/1188976/files/Audio_Speech_Actors_01-24.zip"


@dataclass
class AudioCfg:
    sr: int = 22_050            # target sampling rate (RAVDESS is 48 kHz -> resampled)
    duration: float = 3.0       # seconds kept per clip (trim silence, then centre crop / zero-pad)
    top_db: int = 30            # silence trimming threshold
    n_mfcc: int = 40
    n_mels: int = 128
    n_fft: int = 2048
    hop_length: int = 512

    @property
    def n_samples(self) -> int:
        return int(self.sr * self.duration)

    @property
    def n_frames(self) -> int:
        return 1 + self.n_samples // self.hop_length

    @property
    def n_features(self) -> int:  # MFCC + delta + delta-delta
        return self.n_mfcc * 3


@dataclass
class SplitCfg:
    """Speaker-independent split: no actor appears in more than one partition."""
    train_actors: tuple = tuple(range(1, 17))    # 16 actors
    val_actors: tuple = tuple(range(17, 21))     # 4 actors
    test_actors: tuple = tuple(range(21, 25))    # 4 actors


@dataclass
class TrainCfg:
    epochs: int = 60
    batch_size: int = 32
    lr: float = 1e-3
    patience: int = 12
    aug_copies: int = 2         # augmented copies per training clip
    seed: int = 42


@dataclass
class Cfg:
    audio: AudioCfg = field(default_factory=AudioCfg)
    split: SplitCfg = field(default_factory=SplitCfg)
    train: TrainCfg = field(default_factory=TrainCfg)
