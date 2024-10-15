import librosa
import soundfile as sf


def convert_mp3_to_wav(input_file, output_file, target_sr=16000):
    # Read the audio file (MP3 or WAV)
    y, sr = librosa.load(input_file, sr=None)

    # Resample if the current sample rate is not the target sample rate
    if sr != target_sr:
        y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)

    # Save the resampled audio as WAV
    sf.write(output_file, y, target_sr, subtype="PCM_16")


# Example usage
input_file = "spk0_0_pitch_shift_timevarying4.wav"
output_file = "spk0_0_pitch_shift_timevarying4_16Hz.wav"
convert_mp3_to_wav(input_file, output_file)

print(f"Conversion complete. Resampled WAV file saved as {output_file}")
