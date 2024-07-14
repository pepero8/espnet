import argparse
import os
import sys

from pydub import AudioSegment


def main(args):
    src = args.src
    dst = args.dst

    spk2utt = {}
    utt2spk = []
    wav_list = []

    for r, ds, fs in os.walk(src):
        for f in fs:
            file_extension = os.path.splitext(f)[1]
            # if os.path.splitext(f)[1] != ".wav":
            if file_extension != ".wav":
                if file_extension == ".m4a":
                    # Convert m4a to wav
                    m4a_path = os.path.join(r, f)
                    wav_path = os.path.join(r, os.path.splitext(f)[0] + ".wav")
                    audio = AudioSegment.from_file(m4a_path, format="m4a")
                    audio.export(wav_path, format="wav")
                    utt_dir = wav_path
                else:
                    continue
            else:
                utt_dir = os.path.join(r, f)

            spk, vid, utt = utt_dir.split("/")[-3:]  # speaker, video, utterance
            utt_id = "/".join([spk, vid, utt.split(".")[0]])
            if spk not in spk2utt:
                spk2utt[spk] = []
            spk2utt[spk].append(utt_id)
            utt2spk.append([utt_id, spk])
            wav_list.append([utt_id, utt_dir])

    with open(os.path.join(dst, "spk2utt"), "w") as f_spk2utt, \
        open(os.path.join(dst, "utt2spk"), "w") as f_utt2spk, \
        open(os.path.join(dst, "wav.scp"), "w") as f_wav:
        for spk in spk2utt:
            f_spk2utt.write(f"{spk}")
            for utt in spk2utt[spk]:
                f_spk2utt.write(f" {utt}")
            f_spk2utt.write("\n")

        for utt in utt2spk:
            f_utt2spk.write(f"{utt[0]} {utt[1]}\n")

        for utt in wav_list:
            f_wav.write(f"{utt[0]} {utt[1]}\n")

    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VoxCeleb 1&2 downloader")
    parser.add_argument(
        "--src",
        type=str,
        required=True,
        help="source directory of voxcelebs",
    )
    parser.add_argument(
        "--dst",
        type=str,
        required=True,
        help="destination directory of voxcelebs",
    )
    args = parser.parse_args()

    sys.exit(main(args))
