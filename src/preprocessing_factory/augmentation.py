import os
import librosa
import os
import numpy as np


class AudioAugmentation:
    def __init__(self, args, train = True):
        self.args = args
        self.train = train

        if train: #train
            directorys = ["./data/train_noise"]
            self.noise_directorys = self._get_all_noise_files(directorys)
            print("use train")
        if train == False: #Validation
            directorys = ["./data/valid_noise"]
            print("use validation")
            self.noise_directorys = self._get_all_noise_files(directorys)

    def augmentation(self, audio1, audio2, label1, label2):
        audio1 = self._fix_length_to_5s(audio1, self.args.SR)
        audio2 = self._fix_length_to_5s(audio2, self.args.SR)

        audio1 = self._change_volume(audio1)
        audio2 = self._change_volume(audio2)

        if self.train:
            augmented_audio, label, people_cnt = self._combine_audios(audio1, audio2, label1, label2)
            augmented_audio, is_noise = self._add_diffusion_noise(augmented_audio)
        else:
            augmented_audio, label, people_cnt = self._combine_audios(audio1, audio2, label1, label2)
            augmented_audio, is_noise = self._add_diffusion_noise(augmented_audio)
            
        return augmented_audio, label, people_cnt, is_noise
    
    def _fix_length_to_5s(self, audio, sr, target_length = 5):
        target_length_samples = int(target_length * sr)
        current_length = len(audio)
        
        if current_length > target_length_samples:
            # Randomly crop to 5 seconds
            start_idx = np.random.randint(0, current_length - target_length_samples)
            return audio[start_idx:start_idx + target_length_samples]
        else:
            # Random zero padding to make it 5 seconds
            pad_length = target_length_samples - current_length
            pad_before = np.random.randint(0, pad_length+1)
            pad_after = pad_length - pad_before
            return np.pad(audio, (pad_before, pad_after), 'constant')
        
    def _combine_audios(self, audio1, audio2, label1, label2):
        prob = np.random.rand()
        if prob < self.args.two_people_prob:
            data = audio1 + audio2
            label_vector = label1 + label2
            label_vector[label_vector>1] = 1
            return data, label_vector, np.array([2], dtype=np.float32)
        
        elif prob < self.args.two_people_prob + self.args.zero_people_prob:
            return np.zeros_like(audio1), np.zeros(2, dtype=np.float32), np.zeros(1, dtype=np.float32)
        
        else:
            return audio1, label1, np.ones(1, dtype=np.float32)

    def _change_volume(self, audio, volume_range=(0.5, 1.2)):
        volume_factor = np.random.uniform(volume_range[0], volume_range[1])
        return audio * volume_factor

    def _get_all_noise_files(self, directories):
        noise_files = []
        for directory in directories:
            for root, _, files in os.walk(directory):
                for file in files:
                    if file.endswith('.wav') or file.endswith('ogg'):
                        noise_files.append(os.path.join(root, file))
        return noise_files
    
    def _add_diffusion_noise(self, audio):
        if np.random.rand() < self.args.noise_prob:
            noise_file = np.random.choice(self.noise_directorys)
            noise, sr = librosa.load(noise_file, sr=self.args.SR)
            
            return audio + self._change_volume(noise), np.ones(1, dtype=np.float32)
        else:
            return audio, np.zeros(1, dtype=np.float32)