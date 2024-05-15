import random
import torch
import torch.nn.functional as F
import torch.utils.data
import torchaudio
from transformers import AutoTokenizer

from utils.audio import load_audio


def load_libritts_r(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        filepaths_and_text = []
        bad_lines = 0
        for line in f:
            components = line.strip().split('|')
            assert len(components) == 3, f'{filename} contains bad entries: {line}'
            filepaths_and_text.append([components[0], components[1], components[2]])
    return filepaths_and_text


class TextWavLoader(torch.utils.data.Dataset):
    def __init__(self, hparams):
        # metadata
        self.path = hparams.metadata_path

        # desired speech sample rate
        self.sample_rate = hparams.sample_rate

        # length to pad clips to in seconds
        self.speech_dur = hparams.speech_dur
        self.speech_dur_samples = int(self.speech_dur * self.sample_rate)
        self.should_skip_too_long_wav = hparams.should_skip_too_long_wav

        # text tokenizer
        base_model = hparams.base_model
        self.tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=True)

        # what length to pad/truncate all text tokens to
        self.should_skip_too_long_text = hparams.should_skip_too_long_text
        self.text_max_length = hparams.text_max_length

        # load metafile
        self.audiopaths_and_meta = load_libritts_r(self.path)
        self.seed = hparams.seed
        random.seed(self.seed)
        random.shuffle(self.audiopaths_and_meta)

        # number of speech samples
        self.total_clips = len(self.audiopaths_and_meta)


    def get_audio(self, audiofile) -> torch.Tensor:
        ext = audiofile.split('.')[-1]
        if ext in ['flac', 'opus', 'wav']:
            s, sr = load_audio(audiofile)
        else:
            raise ValueError(f'Unknown audio format: {ext}\n')

        # Workaround for some data being saved at a different sample rate, we'll just resample it for now until we can fix it
        if sr != self.sample_rate:
            s = torchaudio.functional.resample(s[None], sr, self.sample_rate)[0]
            sr = self.sample_rate
            assert sr == self.sample_rate, "sample rate mismatch"
        return s

    def get_wav_text_pair(self, audiopath_and_meta):
        # separate filename and text
        audiopath, text, _ = audiopath_and_meta[0], audiopath_and_meta[1], audiopath_and_meta[2]
        wav = self.get_audio(audiopath)
        return (wav, text, audiopath)

    def tokenize_text(self, text):
        tokens = torch.tensor(self.tokenizer.encode(text), dtype=torch.int)
        return tokens

    def __getitem__(self, index):

        # get wav and meta pairs
        wav, text, audiofile = self.get_wav_text_pair(self.audiopaths_and_meta[index])

        # speech: check for duration and pad
        orig_wav_len = wav.shape[-1]

        if self.should_skip_too_long_wav and orig_wav_len > self.speech_dur_samples:
            print(f'Skipping {audiofile} because it is too long.')
            return self[(index+1) % len(self)]

        # pad speech
        if orig_wav_len != self.clip_length_samples:
            wav = F.pad(wav, (0, self.clip_length_samples - orig_wav_len))

        # text processing
        text_tokens = self.tokenize_text(text)
        text_tokens_len = text_tokens.shape[0]
        if self.should_skip_too_long_text and text_tokens_len > self.text_max_length:
            print(f'Skipping {text} because it is too long.')
            return self[(index+1) % len(self)]

        if text_tokens_len < self.text_max_length:
            text_tokens = F.pad(
                text_tokens,
                (0, self.text_max_length - text_tokens.shape[0]),
            )
        elif text_tokens_len > self.text_max_length:
            text_tokens = text_tokens[: self.text_max_length]
            text_tokens_len = self.text_max_length

        rv = {
            'text': text,
            'text_tokens': text_tokens,
            'text_lengths': torch.tensor(text_tokens_len, dtype=torch.long),
            'speech': wav,
            'speech_lengths': torch.tensor(orig_wav_len, dtype=torch.long),
            'audiofile': audiofile
        }
        return rv

    def __len__(self):
        return len(self.audiopaths_and_meta)


if __name__ == '__main__':
    from data import create_dataset, create_dataloader
    cfg_file = OmegaConf("configs/llama_tts_audio_trial01.yml")
    cfg.datasets.train.batch_size. = 8
    dataset = create_dataset(cfg.datasets.train)
    dataloader = create_dataloader(dataset, cfg.datasets.train)
    for i, batch in enumerate(dataloader):
        print(f'* batch {i}, real text: {batch["text"]}')
        print(f'\ttext tokens: {batch["text_tokens"]}\n')
        print(f'\tshape of text tokens: {batch["text_lengths"].shape}\n')
        print(f'\ttext lengths: {batch["text_lengths"]}\n')
        print(f'\tshape of wavs: {batch["wav"].shape}\n')
        if i > 5:
            break

