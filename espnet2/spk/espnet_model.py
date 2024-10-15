# Copyright 2023 Jee-weon Jung
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import os
from typing import Any, Callable, Dict, Optional, Tuple, Union

import numpy as np
import parselmouth
import torchaudio
from espnet2.asr.encoder.abs_encoder import AbsEncoder
from espnet2.asr.frontend.abs_frontend import AbsFrontend
from espnet2.asr.specaug.abs_specaug import AbsSpecAug
from espnet2.layers.abs_normalize import AbsNormalize
from espnet2.spk.loss.abs_loss import AbsLoss
from espnet2.spk.pooling.abs_pooling import AbsPooling
from espnet2.spk.projector.abs_projector import AbsProjector
from espnet2.torch_utils.device_funcs import force_gatherable
from espnet2.train.abs_espnet_model import AbsESPnetModel
from espnet2.train.preprocessor import SpkPreprocessor
from numpy.typing import NDArray
from typeguard import typechecked

import torch


class ESPnetSpeakerModel(AbsESPnetModel):
    """Speaker embedding extraction model.

    Core model for diverse speaker-related tasks (e.g., verification, open-set
    identification, diarization)

    The model architecture comprises mainly 'encoder', 'pooling', and
    'projector'.
    In common speaker recognition field, the combination of three would be
    usually named as 'speaker_encoder' (or speaker embedding extractor).
    We splitted it into three for flexibility in future extensions:
      - 'encoder'   : extract frame-level speaker embeddings.
      - 'pooling'   : aggregate into single utterance-level embedding.
      - 'projector' : (optional) additional processing (e.g., one fully-
                      connected layer) to derive speaker embedding.

    Possibly, in the future, 'pooling' and/or 'projector' can be integrated as
    a 'decoder', depending on the extension for joint usage of different tasks
    (e.g., ASR, SE, target speaker extraction).
    """

    @typechecked
    def __init__(
        self,
        frontend: Optional[AbsFrontend],
        specaug: Optional[AbsSpecAug],
        # rirmusan: Optional[Callable[[str, Dict[str, np.array]], Dict[str, np.ndarray]]],
        rirmusan: Optional[SpkPreprocessor],
        normalize: Optional[AbsNormalize],
        encoder: Optional[AbsEncoder],
        pooling: Optional[AbsPooling],
        projector: Optional[AbsProjector],
        loss: Optional[AbsLoss],
    ):

        super().__init__()

        self.frontend = frontend
        self.specaug = specaug
        self.rirmusan = rirmusan
        self.normalize = normalize
        self.encoder = encoder
        self.pooling = pooling
        self.projector = projector
        self.loss = loss

    @typechecked
    def forward(
        self,
        speech: torch.Tensor,
        spk_labels: Optional[torch.Tensor] = None,
        task_tokens: Optional[torch.Tensor] = None,
        extract_embd: bool = False,
        **kwargs,
    ) -> Union[Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor], torch.Tensor]:
        """Feed-forward through encoder layers and aggregate into utterance-level

        feature.

        Args:
            speech: (Batch, samples)
            speech_lengths: (Batch,)
            extract_embd: a flag which doesn't go through the classification
                head when set True
            spk_labels: (Batch, )
            one-hot speaker labels used in the train phase
            task_tokens: (Batch, )
            task tokens used in case of token-based trainings
        """
        if spk_labels is not None:
            assert speech.shape[0] == spk_labels.shape[0], (
                speech.shape,
                spk_labels.shape,
            )
        if task_tokens is not None:
            assert speech.shape[0] == task_tokens.shape[0], (
                speech.shape,
                task_tokens.shape,
            )
        batch_size = speech.shape[0]

        # 1. split audio into two parts
        # half_length = speech.shape[1] // 2
        # speech_1 = speech[:, :half_length]  # (N, n_mel)
        # speech_2 = speech[:, half_length:]  # (N, n_mel)
        speech_1, speech_2 = self.random_crop(wav_tensor=speech)  # (256, 8000) each

        # 2. pitch_shift audio
        speech_1_pitch, shift_intensities1 = self.random_pitch_shift(speech_1)
        speech_2_pitch, shift_intensities2 = self.random_pitch_shift(speech_2)

        # 3. extract low-level feats (e.g., mel-spectrogram or MFCC)
        # Will do nothing for raw waveform-based models (e.g., RawNets)
        feats, _ = self.extract_feats(
            speech_1, None, False
        )  # we only use first 0.5 segment at inference # extract features + normalize
        feats_clean_1, _ = self.extract_feats(
            speech_1, None, True
        )  # rir/musan aug + extract feats + specaug + normalize
        feats_clean_2, _ = self.extract_feats(
            speech_2, None, True
        )  # rir/musan aug + extract feats + specaug + normalize
        feats_1_pitch, _ = self.extract_feats(speech_1_pitch, None, False)  # (N, n_mel)
        feats_2_pitch, _ = self.extract_feats(speech_2_pitch, None, False)  # (N, n_mel)

        # 4. extract frame-level feats
        frame_level_feats = self.encode_frame(feats)
        frame_level_feats_clean_1 = self.encode_frame(feats_clean_1)
        frame_level_feats_clean_2 = self.encode_frame(feats_clean_2)
        frame_level_feats_pitch1 = self.encode_frame(feats_1_pitch)
        frame_level_feats_pitch2 = self.encode_frame(feats_2_pitch)

        # 5. aggregation into utterance-level
        utt_level_feat = self.pooling(frame_level_feats, task_tokens)
        utt_level_feat_clean_1 = self.pooling(frame_level_feats_clean_1, task_tokens)
        utt_level_feat_clean_2 = self.pooling(frame_level_feats_clean_2, task_tokens)
        utt_level_feat_pitch1 = self.pooling(frame_level_feats_pitch1, task_tokens)
        utt_level_feat_pitch2 = self.pooling(frame_level_feats_pitch2, task_tokens)

        # 6. (optionally) go through further projection(s)
        spk_embd = self.project_spk_embd(utt_level_feat)
        spk_embd_clean_1 = self.project_spk_embd(utt_level_feat_clean_1)  # (N, nout)
        spk_embd_clean_2 = self.project_spk_embd(utt_level_feat_clean_2)  # (N, nout)
        spk_embd_pitch1 = self.project_spk_embd(utt_level_feat_pitch1)  # (N, nout)
        spk_embd_pitch2 = self.project_spk_embd(utt_level_feat_pitch2)  # (N, nout)

        # 7. concat to make (2N, n_mel)
        embd_clean = torch.cat([spk_embd_clean_1, spk_embd_clean_2], dim=0)
        embd_pitch1 = torch.cat(
            [spk_embd_clean_1, spk_embd_pitch1], dim=0
        )  # pair with same context
        embd_pitch2 = torch.cat(
            [spk_embd_clean_2, spk_embd_pitch2], dim=0
        )  # pair with same context
        # speech_1_pitch = torch.cat([speech_2, speech_1_pitch], dim=0)  # pair with different context
        # speech_2_pitch = torch.cat([speech_1, speech_2_pitch], dim=0)  # pair with different context

        # # 0-2. concat to make (2N, n_mel)
        # feats_clean = torch.cat([feats_clean_1, feats_clean_2], dim=0)
        # feats_1_pitch = torch.cat([feats_clean_1, feats_1_pitch], dim=0)  # pair with same context
        # feats_2_pitch = torch.cat([feats_clean_2, feats_2_pitch], dim=0)  # pair with same context
        # # speech_1_pitch = torch.cat([speech_2, speech_1_pitch], dim=0)  # pair with different context
        # # speech_2_pitch = torch.cat([speech_1, speech_2_pitch], dim=0)  # pair with different context

        # # 1. extract low-level feats (e.g., mel-spectrogram or MFCC)
        # # Will do nothing for raw waveform-based models (e.g., RawNets)
        # feats, _ = self.extract_feats(
        #     speech_1, None, False
        # )  # we only use first 0.5 segment at inference # extract features + specaugment + normalize
        # feats_clean, _ = self.extract_feats(speech_clean, None, True) # includes rir/musan augment
        # feats_1_pitch, _ = self.extract_feats(speech_1_pitch, None, False)  # (2N, n_mel)
        # feats_2_pitch, _ = self.extract_feats(speech_2_pitch, None, False)  # (2N, n_mel)

        # frame_level_feats = self.encode_frame(feats)
        # frame_level_feats_clean = self.encode_frame(feats_clean)
        # frame_level_feats_pitch1 = self.encode_frame(feats_1_pitch)
        # frame_level_feats_pitch2 = self.encode_frame(feats_2_pitch)

        # # 2. aggregation into utterance-level
        # utt_level_feat = self.pooling(frame_level_feats, task_tokens)
        # utt_level_feat_clean = self.pooling(frame_level_feats_clean, task_tokens)
        # utt_level_feat_pitch1 = self.pooling(frame_level_feats_pitch1, task_tokens)
        # utt_level_feat_pitch2 = self.pooling(frame_level_feats_pitch2, task_tokens)

        # # 3. (optionally) go through further projection(s)
        # spk_embd = self.project_spk_embd(utt_level_feat)
        # spk_embd_clean = self.project_spk_embd(utt_level_feat_clean)  # (2N, nout)
        # spk_embd_pitch1 = self.project_spk_embd(utt_level_feat_pitch1)  # (2N, nout)
        # spk_embd_pitch2 = self.project_spk_embd(utt_level_feat_pitch2)  # (2N, nout)

        if extract_embd:
            return spk_embd

        # 4. calculate loss
        # assert spk_labels is not None, "spk_labels is None, cannot compute loss" # removed for contrastive loss

        loss_spk = self.loss(embd_clean, None)

        loss_pitch1 = self.loss(embd_pitch1, shift_intensities1)
        loss_pitch2 = self.loss(embd_pitch2, shift_intensities2)
        loss_pitch = self.loss.weight2 * loss_pitch1 + self.loss.weight3 * loss_pitch2

        loss = self.loss.weight1 * loss_spk + loss_pitch

        stats = dict(loss=loss.detach())

        loss, stats, weight = force_gatherable((loss, stats, batch_size), loss.device)
        return loss, stats, weight

    def extract_feats(
        self, speech: torch.Tensor, speech_lengths: torch.Tensor, noiseaug: bool
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # speech: (batch_size, num_samples)
        batch_size = speech.shape[0]
        length = speech.shape[1]
        speech_lengths = (
            speech_lengths
            if speech_lengths is not None
            else torch.ones(batch_size).int() * speech.shape[1]
        )

        # Apply musan+RIR augmentation
        if self.rirmusan is not None and self.training and noiseaug:
            device = speech.device
            speech_aug = np.zeros((batch_size, length))
            if self.rirmusan.noise_apply_prob > 0 or self.rirmusan.rir_apply_prob > 0:
                for i, wav in enumerate(speech):
                    wav_numpy = wav.detach().cpu().numpy()
                    aug_wav_numpy = self.rirmusan._apply_data_augmentation(wav_numpy)
                    # speech_aug[i] = torch.Tensor(aug_wav_numpy).to(device)
                    speech_aug[i] = aug_wav_numpy

                # speech = torch.Tensor(speech_aug).to(device)
                speech = torch.tensor(speech_aug, dtype=speech.dtype, device=device)

        # 1. extract feats
        if self.frontend is not None:
            feats, feat_lengths = self.frontend(speech, speech_lengths)
        else:
            feats = speech
            feat_lengths = None

        if self.specaug is not None and self.training:
            feats, _ = self.specaug(feats, feat_lengths)
            # print("^^7: specaug applied!")  # added by jaehwan

        # 3. normalize
        if self.normalize is not None:
            feats, _ = self.normalize(feats, feat_lengths)

        return feats, feat_lengths

    def encode_frame(self, feats: torch.Tensor) -> torch.Tensor:
        frame_level_feats = self.encoder(feats)

        return frame_level_feats

    def aggregate(self, frame_level_feats: torch.Tensor) -> torch.Tensor:
        utt_level_feat = self.aggregator(frame_level_feats)

        return utt_level_feat

    def project_spk_embd(self, utt_level_feat: torch.Tensor) -> torch.Tensor:
        if self.projector is not None:
            spk_embd = self.projector(utt_level_feat)
        else:
            spk_embd = utt_level_feat

        return spk_embd

    def collect_feats(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        spk_labels: torch.Tensor = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        feats, feats_lengths = self.extract_feats(speech, speech_lengths)
        return {"feats": feats}

    # ================================added by jaehwan================================
    def random_crop(self, wav_tensor, sample_rate=16000, crop_length=0.5):
        """
        Randomly crop two non-overlapping 0.5-second sequences from each 3-second WAV tensor in a batch.

        Args:
        wav_tensor (torch.Tensor): Input WAV tensor of shape (batch, num_samples)
        sample_rate (int): Sample rate of the audio (default: 16000)
        crop_length (float): Length of each crop in seconds (default: 0.5)

        Returns:
        tuple: Two torch tensors, each containing 0.5-second crops for the entire batch
        """
        assert len(wav_tensor.shape) == 2, "Input tensor should have shape (batch, num_samples)"

        batch_size, total_samples = wav_tensor.shape
        crop_samples = int(crop_length * sample_rate)

        # Ensure there is enough room for two non-overlapping segments
        assert (
            total_samples >= 2 * crop_samples
        ), "Not enough samples for two non-overlapping segments."

        # for i in range(batch_size):
        #     # Randomly choose the start position for the first segment
        #     start1 = torch.randint(0, total_samples - 2 * crop_samples, (1,)).item()
        #     end1 = start1 + crop_samples

        #     # Randomly choose the start position for the second segment after the first segment ends
        #     start2 = torch.randint(end1, total_samples - crop_samples, (1,)).item()
        #     end2 = start2 + crop_samples

        #     # Extract the two non-overlapping segments
        #     segment1 = wav_tensor[i, start1:end1]
        #     segment2 = wav_tensor[i, start2:end2]

        possible_starts = total_samples - 2 * crop_samples + 1
        start1 = torch.randint(0, possible_starts, (batch_size,))
        start2 = torch.randint(crop_samples, total_samples - crop_samples + 1, (batch_size,))

        # Ensure start2 is always after start1 + crop_samples
        start2 = torch.where(
            start2 <= start1 + crop_samples,
            start1
            + crop_samples
            + torch.randint(0, total_samples - 3 * crop_samples, (batch_size,)),
            start2,
        )
        start2 = torch.where(
            start2 > (total_samples - crop_samples), total_samples - crop_samples, start2
        )

        batch_indices = torch.arange(batch_size).unsqueeze(1)
        crop_indices = torch.arange(crop_samples).unsqueeze(0)

        crop1 = wav_tensor[batch_indices, start1.unsqueeze(1) + crop_indices]
        crop2 = wav_tensor[batch_indices, start2.unsqueeze(1) + crop_indices]

        return crop1, crop2

    # ================================added by jaehwan================================
    def random_pitch_shift(
        self,
        audio: torch.Tensor,
        # ) -> Tuple[torch.Tensor, np.ndarray[Any, np.dtype[np.int32]]]:
    ) -> Tuple[torch.Tensor, np.ndarray]:
        """
        Shift the pitch of the input audio by randomly sampled n_step.

        Args:
            audio: (Batch, time) - input audio.

        Returns:
            pitch_shifted_audio: (Batch, time) - pitch-shifted audio.
            shift_intensity: The intensities of the pitch shift applied.
        """
        batch_size, time = audio.shape
        device = audio.device
        pitch_shifted_audio = []

        # n_steps = np.random.normal(
        #     size=(batch_size,)
        # ).astype(np.int32)  # Number of steps to shift the pitch. Positive values increase pitch, negative values decrease pitch.
        n_steps_float = np.random.uniform(size=(batch_size,), low=-1.0, high=1.0)
        n_steps_int = (n_steps_float * 100).astype(int)

        for i in range(batch_size):
            wav = audio[i]

            # jio's implement
            wav_shifted = self.change_pitch(
                wav.detach().cpu().numpy(),
                sr=16000,
                fs_ratio=1.0,
                pr_factor=1.0,
                ms_factor=n_steps_int[i],
            )
            wav_shifted = torch.Tensor(wav_shifted).to(device)

            pitch_shifted_audio.append(wav_shifted)

        pitch_shifted_audio = torch.stack(pitch_shifted_audio)

        return pitch_shifted_audio, np.abs(n_steps_float)

    # ================================added by jaehwan================================
    def change_pitch(
        self, wav: NDArray[np.float32], sr: int, fs_ratio: float, pr_factor: float, ms_factor: float
    ) -> NDArray[np.float32]:

        F0_MIN = 50.0
        F0_MAX = 850.0
        F0_MEDIAN_SAFEAREA = 15.0

        snd = parselmouth.Sound(wav, sampling_frequency=sr)

        try:
            snd_f0 = snd.to_pitch_ac(
                pitch_floor=F0_MIN, pitch_ceiling=F0_MAX, time_step=0.8 / F0_MIN
            ).selected_array["frequency"]
        except:  # <- no voice here
            return wav

        snd_f0_exists = snd_f0[snd_f0 != 0]

        if len(snd_f0_exists) == 0:
            return wav  # <- no voice here

        snd_f0_median = np.median(snd_f0_exists)

        if np.isnan(snd_f0_median):  # <- no voice here
            return wav

        snd_f0_shifted = float(
            np.clip(
                snd_f0_median + ms_factor, F0_MIN + F0_MEDIAN_SAFEAREA, F0_MAX - F0_MEDIAN_SAFEAREA
            )
        )

        out: NDArray[np.float64] = (
            parselmouth.praat.call(
                snd, "Change gender", F0_MIN, F0_MAX, fs_ratio, snd_f0_shifted, pr_factor, 1.0
            )
            .as_array()
            .flatten()
        )

        assert len(wav) == len(out)

        return out.astype(np.float32)
