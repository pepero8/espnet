espnet2 spk 논문: [ESPnet-SPK: full pipeline speaker embedding toolkit with reproducible recipes, self-supervised front-ends, and off-the-shelf models](https://arxiv.org/abs/2401.17230v2)

위 논문에서 구현한 speaker verification 레시피를 수정하여 훈련코드를 작성했습니다.

[[공식 사이트]](https://espnet.github.io/espnet/installation.html)

### Installation
conda environment: [environment.yaml](./environment.yaml)
<br>
위의 conda 환경 파일로 espnet 환경 생성
```bash
$ conda env create -n espnet -f environment.yaml
```
만약 오류가 뜰 경우 위의 공식 사이트의 Installation에 Step 2) - 3. Setup Python environment를 참고하여 환경을 만든 후, 4. Install ESPnet 을 진행




### how to train
Vox1, Vox2 데이터셋은 다운이 되어있다고 가정한다.

0. Vox1, Vox2 데이터셋 경로는 각각 voxceleb1/, voxceleb2/ 하위에 위치해 있어야 한다.

    예시)
    <br>
    /shared/oil/voice/voxceleb1/test/wav
    <br>
    /shared/oil/voice/voxceleb1/dev/wav
    <br>
    /shared/oil/voice/voxceleb2/dev/wav
    <br>
    ...

1. egs2/voxceleb/spk1/ 으로 이동
2. local/data.sh에 데이터셋 경로를 수정한다. 예) data_dir_prefix=/shared/oil/voice/

    db.sh에도 VOXCELEB path를 수정한다. 예) VOXCELEB=/shared/oil/voice/

    이후 다음을 실행(데이터셋 준비).
    ```bash
    $ ./local/data.sh --stage 4
    ```

3. 다음을 실행(모델 훈련). 이후 exp/spk_train_ECAPA_mel_jhwan_raw 폴더가 생기고 여기에 train log랑 checkpoint, 훈련 결과 등이 저장된다.
    ```bash
    $ ./run.sh --spk_config conf/tuning/train_ECAPA_mel_jhwan.yaml --stage 3 --stop-stage 5 --speed_perturb_factors "" --ngpu 1
    ```

4. 이후 훈련을 중단했다가 다시 이어서 진행하고자 할 경우 stage 5만 실행하면 된다.
    ```bash
    $ ./run.sh --spk_config conf/tuning/train_ECAPA_mel_jhwan.yaml --stage 5 --stop-stage 5 --speed_perturb_factors "" --ngpu 1
    ```

### 주요 파일:

- ```egs2/voxceleb/spk1/db.sh```: 데이터셋 경로 파일

- ```egs2/voxceleb/spk1/local/data.sh```: voxceleb1&2 데이터셋 다운 및 train,val,test 데이터셋 파일 만드는 스크립트.

- ```egs2/voxceleb/spk1/conf/tuning/train_ECAPA_mel_jhwan.yaml```: 훈련 하이퍼파라미터 설정 파일.

- ```egs2/voxceleb/spk1/spk.sh```: 전반적인 SV task과정을 실행하는 스크립트. 위 논문에서 언급한 내용처럼 데이터셋 준비, 훈련, inference, 등이 step별로 나뉘어져 있음.

- ```espnet2/spk/espnet_model.py```: 훈련 모델. raw wave로 구성된 하나의 배치를 입력으로 받는 forward 함수가 구현되어 있음.

- ```espnet2/spk/loss/contrastive_loss.py```: 임베딩 배치가 입력으로 주어졌을 때 SimCLR loss값을 계산해주는 forward 함수가 구현되어 있는 loss 클래스.

- ```espnet2/train/trainer.py```: train_one_epoch()가 있는 모델 훈련 클래스. train_one_epoch()에서 훈련 모델의 forward를 호출한다.

- ```espnet2/train/spk_trainer.py```: validate_one_epoch()가 있는 SV 전용 훈련 클래스(위의 Trainer를 상속받음).

- ```espnet2/tasks/spk.py```: SV task 관련 객체들(ex. preprocessor, loss, model)을 생성해주는 build 함수들이 포함된 클래스.
