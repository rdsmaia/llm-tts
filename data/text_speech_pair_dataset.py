import random
import torch
import torchaudio
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from utils.audio import load_audio
from omegaconf import OmegaConf


def load_libritts_r(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        metadata = []
        for line in f:
            components = line.strip().split('|')
            assert len(components) == 3, f'{filename} contains bad entries: {line}'
            metadata.append([components[0], components[1], components[2]])
    return metadata


def create_dataloader(hparams,
                      sampler=None,
                      collate_fn=None,
                      shuffle=True):
    dataset = TextSpeechDataset(hparams)
    if hparams.train.dist:
        world_size = torch.distributed.get_world_size()
        num_workers = hparams.train.n_workers
        assert hparams.train.batch_size % world_size == 0
        batch_size = hparams.train.batch_size // world_size
    else:
        num_workers = hparams.train.n_workers
        batch_size = hparams.train.batch_size
    return DataLoader(dataset,
                      batch_size=batch_size,
                      shuffle=shuffle,
                      num_workers=num_workers,
                      sampler=sampler,
                      drop_last=True,
                      pin_memory=hparams.train.pin_memory,
                      collate_fn=collate_fn)


class TextSpeechDataset(Dataset):
    def __init__(self, hparams):
        # metadata
        self.path = hparams.data.train_meta

        # model speech sample rate
        self.sample_rate = hparams.audio.sample_rate

        # length to pad speech
        self.max_speech_dur = hparams.audio.max_speech_dur
        self.max_speech_dur_samples = int(self.max_speech_dur * self.sample_rate)
        self.should_skip_very_long_speech = hparams.model.should_skip_very_long_speech

        # text tokenizer
        tokenizer_base_model = hparams.text.tokenizer_base_model
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_base_model, use_fast=True)

        # length to pad text tokens
        self.should_skip_very_long_text = hparams.model.should_skip_very_long_text
        self.max_text_tokens = hparams.model.max_text_tokens

        # load metafile
        self.metadata = load_libritts_r(self.path)
        self.seed = hparams.train.seed
        random.seed(self.seed)
        random.shuffle(self.metadata)

        # number of speech-text pairs
        self.total_samples = len(self.metadata)

    def get_audio(self, audiofile):
        ext = audiofile.split('.')[-1]
        if ext in ['flac', 'opus', 'wav']:
            s, sr = load_audio(audiofile)
        else:
            raise ValueError(f'Unknown audio format: {ext}\n')
        if sr != self.sample_rate:
            s = torchaudio.functional.resample(s, sr, self.sample_rate)
            sr = self.sample_rate
        assert sr == self.sample_rate, "sample rate mismatch: sr={sr} and sample_rate={self.sample_rate}"
        return s

    def get_audio_text_pair(self, meta_info):
        audiopath, text, _ = meta_info
        wav = self.get_audio(audiopath)
        return (wav, text, audiopath)

    def tokenize_text(self, text):
        tokens = self.tokenizer(text, return_tensors="pt", add_special_tokens=False).input_ids
        return tokens

    def __getitem__(self, index):

        # get speech
        wav, text, audiofile = self.get_audio_text_pair(self.metadata[index])

        # speech: check for duration
        wav_len = wav.shape[-1]
        if self.should_skip_very_long_speech and wav_len > self.max_speech_dur_samples:
            print(f'Skipping {audiofile} because it is too long.')
            return self[(index+1) % len(self)]

        # pad speech
        if wav_len < self.max_speech_dur_samples:
            wav = F.pad(
                wav,
                (0, self.max_speech_dur_samples - wav_len))
        elif wav_len > self.max_speech_dur_samples:
            wav = wav[: self.max_speech_dur]
            wav_len = self.max_speech_dur

        # tokenize text
        text_tokens = self.tokenize_text(text).squeeze(0)

        # text tokens: check for length
        text_tokens_len = text_tokens.shape[-1]
        if self.should_skip_very_long_text and text_tokens_len > self.max_text_tokens:
            print(f'Skipping {text} because it is too long.')
            return self[(index+1) % len(self)]

        # pad text tokens
        if text_tokens_len < self.max_text_tokens:
            text_tokens = F.pad(
                text_tokens,
                (0, self.max_text_tokens - text_tokens_len),
            )
        elif text_tokens_len > self.max_text_tokens:
            text_tokens = text_tokens[: self.max_text_tokens]
            text_tokens_len = self.max_text_tokens

        rv = {
            'text': text,
            'text_tokens': text_tokens,
            'text_lengths': torch.tensor(text_tokens_len, dtype=torch.long),
            'speech': wav,
            'speech_lengths': torch.tensor(wav_len, dtype=torch.long),
            'audiofile': audiofile
        }
        return rv

    def __len__(self):
        return len(self.metadata)


if __name__ == '__main__':
    cfg = OmegaConf.load("config/llama-tts_trial01.yml")
    cfg.train.batch_size = 4
    dataloader = create_dataloader(cfg)
    for i, batch in enumerate(dataloader):
        print(f'* batch {i}, real text: {batch["text"]}')
        print(f'\ttext tokens: {batch["text_tokens"]}\n')
        print(f'\tshape of text tokens: {batch["text_tokens"].shape}\n')
        print(f'\ttext lengths: {batch["text_lengths"]}\n')
        print(f'\tshape of speech: {batch["speech"].shape}\n')
        print(f'\tspeech lengths: {batch["speech_lengths"]}\n')
        print(f'\t audiofiles: {batch["audiofile"]}\n')
        if i > 5:
            break

