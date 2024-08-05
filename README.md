espnet2 spk 논문: [ESPnet-SPK: full pipeline speaker embedding toolkit with reproducible recipes, self-supervised front-ends, and off-the-shelf models](https://arxiv.org/abs/2401.17230v2)

위 논문에서 구현한 speaker verification 레시피를 수정하여 훈련코드를 작성했습니다.

[[공식 사이트]](https://espnet.github.io/espnet/installation.html)

주요 파일:

egs2/voxceleb/spk1/db.sh: 데이터셋 경로 파일

egs2/voxceleb/spk1/local/data.sh: voxceleb1&2 데이터셋 다운 및 train,val,test 데이터셋 파일 만드는 스크립트.

egs2/voxceleb/spk1/conf/tuning/train_ECAPA_mel_jhwan.yaml: 훈련 하이퍼파라미터 설정 파일.

egs2/voxceleb/spk1/spk.sh: 전반적인 SV task과정을 실행하는 스크립트. 위 논문에서 언급한 내용처럼 데이터셋 준비, 훈련, inference, 등이 step별로 나뉘어져 있음.

espnet2/spk/espnet_model.py: 훈련 모델. raw wave로 구성된 하나의 배치를 입력으로 받는 forward 함수가 구현되어 있음.

espnet2/spk/loss/contrastive_loss.py: 임베딩 배치가 입력으로 주어졌을 때 SimCLR loss값을 계산해주는 forward 함수가 구현되어 있는 loss 모델.

espnet2/train/trainer.py: 모델의 forward를 호출하는 train_one_epoch()가 포함된 모델 훈련 클래스.

espnet2/train/spk_trainer.py: validate_one_epoch()를 오버라이딩한 SV 전용 훈련 클래스(위의 Trainer를 상속받음).

espnet2/tasks/spk.py: SV task 관련 객체들(ex. preprocessor, loss, model)을 생성해주는 build 함수들이 포함된 클래스.
