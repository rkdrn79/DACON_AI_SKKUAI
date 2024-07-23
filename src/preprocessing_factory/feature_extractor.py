import os
import numpy as np
import torchaudio
import torchvision.transforms
from PIL import Image
import torch

class FeatureExtractor:
    def __init__(self, feature_type, args):
        self.feature_type = feature_type
        self.args = args

        if self.feature_type == 'spectrogram':
            self.extractor = self._get_extract_spectrogram
        else:
            raise ValueError(f"Unsupported feature type: {self.feature_type}")


    def _get_extract_spectrogram(self, y):
        sr = self.args.SR
        num_channels = 3
        window_sizes = [25, 50, 100]
        hop_sizes = [10, 25, 50]

        specs = []
        for i in range(num_channels):
            window_length = int(round(window_sizes[i] * sr / 1000))
            hop_length = int(round(hop_sizes[i] * sr / 1000))
            
            y = torch.Tensor(y)
            spec = torchaudio.transforms.MelSpectrogram(
                sample_rate=sr, n_fft=3200, win_length=window_length, hop_length=hop_length, n_mels=128
            )(y)
            eps = 1e-6
            spec = spec.numpy()
            spec = np.log(spec + eps)
            spec = np.asarray(torchvision.transforms.Resize((128, 250))(Image.fromarray(spec)))

            specs.append(spec)

        return np.array(specs)

    def extract_features(self, y):
        if self.feature_type == 'spectrogram':
            return self.extractor(y)
        else:
            raise ValueError(f"Unsupported feature type: {self.feature_type}")


