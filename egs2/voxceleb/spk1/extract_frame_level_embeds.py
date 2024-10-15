from typing import List, Tuple, Union
import numpy as np
import soundfile
import torch
from espnet2.tasks.spk import SpeakerTask
from espnet2.torch_utils.device_funcs import to_device


def pad_or_slice_audio(audio_list, target_length):
    padded_list = []
    for audio in audio_list:
        if audio.shape[0] < target_length:
            # Pad
            padded_audio = np.pad(
                audio, (0, target_length - audio.shape[0]), mode="constant"
            )
        else:
            # Slice
            padded_audio = audio[:target_length]
        padded_list.append(padded_audio)
    return np.array(padded_list)


def soundfile_read(
    wavs: Union[str, List[str]],
    dtype=None,
    always_2d: bool = False,
    concat_axis: int = 1,
    start: int = 0,
    end: int = None,
    return_subtype: bool = False,
) -> Tuple[np.array, int]:
    if isinstance(wavs, str):
        wavs = [wavs]

    arrays = []
    subtypes = []
    prev_rate = None
    prev_wav = None
    for wav in wavs:
        with soundfile.SoundFile(wav) as f:
            f.seek(start)
            if end is not None:
                frames = end - start
            else:
                frames = -1
            if dtype == "float16":
                array = f.read(
                    frames,
                    dtype="float32",
                    always_2d=always_2d,
                ).astype(dtype)
            else:
                array = f.read(frames, dtype=dtype, always_2d=always_2d)
            rate = f.samplerate
            subtype = f.subtype
            subtypes.append(subtype)

        if len(wavs) > 1 and array.ndim == 1 and concat_axis == 1:
            # array: (Time, Channel)
            array = array[:, None]

        if prev_wav is not None:
            if prev_rate != rate:
                raise RuntimeError(
                    f"'{prev_wav}' and '{wav}' have mismatched sampling rate: "
                    f"{prev_rate} != {rate}"
                )

            dim1 = arrays[0].shape[1 - concat_axis]
            dim2 = array.shape[1 - concat_axis]
            if dim1 != dim2:
                raise RuntimeError(
                    "Shapes must match with "
                    f"{1 - concat_axis} axis, but got {dim1} and {dim2}"
                )

        prev_rate = rate
        prev_wav = wav
        arrays.append(array)

    if len(arrays) == 1:
        array = arrays[0]
    else:
        array = np.concatenate(arrays, axis=concat_axis)

    if return_subtype:
        return array, rate, subtypes
    else:
        return array, rate


spk_train_config = "/home/jhwan98/seah/espnet/egs2/voxceleb/spk1/exp/spk_train_ECAPA_mel_jhwan_raw/config.yaml"
spk_model_file = "/home/jhwan98/seah/espnet/egs2/voxceleb/spk1/exp/spk_train_ECAPA_mel_jhwan_raw/valid.eer.best.pth"
device = "cuda"
output_dir = "/home/jhwan98/seah/espnet/egs2/voxceleb/spk1/exp/spk_train_ECAPA_mel_jhwan_raw/inference_frame_level"

spk_model, spk_train_args = SpeakerTask.build_model_from_file(
    spk_train_config, spk_model_file, device
)
spk_model.eval()
print("build model complete")

spk_embd_dic = {}

wavs = [
    "./vocalset_straight_sample.wav",
    "./spk0_0_cut.wav",
]

audio_list = []
for wav in wavs:
    audio, rate = soundfile_read(wav, dtype="float32")
    audio_list.append(audio)

max_length = max(audio.shape[0] for audio in audio_list)
padded_audio = pad_or_slice_audio(audio_list, max_length)

# speech = soundfile_read(
#     wavs=wavs,
#     dtype="float32",
# )[0]  # (speech, sample_rate)
# print("^^7 type: ", type(speech))
# print("^^7 speech: ", speech)
# print("^^7 shape: ", speech.shape)

speech = torch.from_numpy(padded_audio)
# speech = torch.from_numpy(speech).unsqueeze(0)
speech = to_device(speech, "cuda")

print("Audio shape after padding/slicing:", speech.shape)

# print("\nafter conversion")
# print("^^7 type: ", type(speech))
# print("^^7 speech: ", speech)
# print("^^7 shape: ", speech.shape)

spk_embd = spk_model(
    speech=speech,
    spk_labels=None,
    extract_embd=True,
    task_tokens=None,
)
print("^^7 type: ", type(spk_embd))
# print("^^7 speech: ", spk_embd)
print("^^7 shape: ", spk_embd.shape)

# spk_embd = spk_embd.squeeze(0)
for i, embd in enumerate(spk_embd):
    spk_embd_dic[str(i)] = embd.detach().cpu().numpy()
# spk_embd_dic["00"] = spk_embd.squeeze(0).detach().cpu().numpy()

np.savez(output_dir + "/embeddings_frame_level3", **spk_embd_dic)
print(f"saved to {output_dir + '/embeddings_frame_level3'}")
