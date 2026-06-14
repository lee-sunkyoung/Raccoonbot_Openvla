# PhysicalAI Project

라쿤봇 OpenVLA에 새로운 오브젝트 타입(box), push task를 추가하고, 언어 지시문을 확장하였습니다. 
또한 추론 시 4dof로 모션 속도를 빠르게 움직이도록 수정하고 로그를 추가하였습니다.

## demo_videos 폴더에서 영상 및 시각자료 확인이 가능합니다. 

## 레포지토리 구성

`/client` 폴더 : local 환경 내부에서 실행되는 코드
`/demo_videos` 폴더: make_video.py`로 에피소드를 저장한 시뮬레이션, 실제 영상 등 시각자료 폴더
`/Project Report.md` 
`/기타 폴더들은 fork 이후 변경점들을 추가하거나 변경하였습니다.`

## 과제1 : 데이터 증강
  
  주요 변경 폴더
  `Raccoonbot_Openvla1/Mujoco/Raccoon_colored_cylinder.xml`
  `Raccoonbot_Openvla1/Mujoco/raccoon_grasp_multicolor_scene_dataset.py`

  ### Add New Objects : cube 
  - 기존 실린더 4색 + 큐브 1개 추가

  ### Add New Task : Push
  - 큐브를 집은 후 해당 위치부터 높이를 유지한 채 월드좌표계 기준으로 이동

  ### Diverse Language Instructions
  - sentence templates / prefix / verb / noun 을 랜덤으로 생성
  -templates =  `["{prefix} {verb} the {noun}", "{prefix} {verb} the {noun} up", "{prefix} {verb} the {noun} for me","{prefix} {verb} the {color} {noun},"{prefix} {verb} the {color} {noun} up", "{prefix} {verb} the {color} {noun} for me"]` 형태로 다양화
  - prefixes =  `["please", "can you", "carefully", "gently", ""] `
  - verbs = ` ["grasp", "pick up", "grab", "catch", "take","push", "slide", "move"] `
  - nouns = ` ["cylinder", "box", "cube", "block", "object", "target box"] `

  - 예시 :  `please slide the box for me ` /  `carefully grasp the red cylinder `




## 과제2 : 코드 개선 및 실제 로봇 동작

  주요 변경 폴더
  `openvla_multicolor_client_real_robot.py`
  `finetune.py / finetune2.py`
  `/client/batch_rollout.sh`

  ### 7D -> 4DOF
  - 이미 raccoon_env에서 실제 사용시 4dof 만 사용하도록 되어있음 (roll/pitch/yaw는 무시)

  ### 로봇 속도 개선
  - 실물 로봇 동작 속도 - 이미지 포맷 및 settle 타임 변경 (동일seed 수행시간 비교결과 : 기존 5min46s->변경2min5s)
  - 명령어 및 예시 결과
```
  python openvla_multicolor_client_real_robot.py   --server_url http://127.0.0.1:8001   --target_color green   --seed 42   --image_quality 30   --no_save_frames   --use_real_robot   --use_viewer
[CLEANUP] removed 0 existing image files from rollout_outputs/episode_000001
Raccoon[0] Connected: /dev/ttyACM0 D5:BF:DA:6B:6F:D6
[REAL_ROBOT] 하드웨어 연결 성공
[DEBUG] current ee from get_ee_pose = (0.00, 14.74, 9.44) cm
[DEBUG] IK(current ee) = [-0.005280205391976754, -10.001870609331936, -140.00075796191655, 60.00262857124849]
[SCENE] instruction='grasp the green cylinder' | target_color='green' | target_xy=(0.057, 0.172) | objects={'red': {'body_name': 'target_object', 'xy': [-0.054552255643044625, 0.20991263083142514], 'yaw': -0.6851542519228805}, 'blue': {'body_name': 'target_object_blue', 'xy': [-0.025840395153483756, 0.24340884899637416], 'yaw': 0.2259828021766146}, 'green': {'body_name': 'target_object_green', 'xy': [0.05721286105539078, 0.17153022694079914], 'yaw': -0.077933586511017}, 'yellow': {'body_name': 'target_object_yellow', 'xy': [0.07171958398227649, 0.22276312261534276], 'yaw': -0.6374647312682433}}
[INFER] server request time: 0.411s | image_quality=30
[REAL_ROBOT] target_cm=[0.5, 15.01, 9.36] | joint_deg=[-1.91, -11.81, -138.38, 60.19] | gripper=open
[000] OK | final_delta=[0.005, 0.0028, -0.0006] | move=[0.005, 0.0028, -0.0006] | target=[0.005, 0.1501, 0.0936] | gripper=0.0 | retries=0
[INFER] server request time: 0.328s | image_quality=30
```

  ### 로그 / 시각화

  - client 내부에서 추론 결과 확인시 한번에 여러번 테스트 가능한 쉘 코드 추가
  - 기타 여러 폴더 내 디버깅용 로그 추가
  - 예시 결과

```
  ==========================================
Batch Rollout & Advanced Condition Logging
Target Episodes: 1
Analysis Report: ./경로.csv
==========================================

[12:03:11] Episode 000001/1 시작...
[✓ LIFT_SUCCESS] Color: red | Verb: grasp | Noun: cylinder | Goal: N/A
          Result: 성공 | 누적성공률: 100% (1/1)

==========================================
        배치 롤아웃 조건별 최종 통계
==========================================
총 에피소드 수 : 1회
총 성공 횟수   : 1회
최종 전체 성공률: 100%

TARGET COLOR 통계 리포트
조건 이름             | 총 호출수     | 성공률
------------------------------------------
red                | 1          | 100%

COMMAND VERB 통계 리포트
조건 이름             | 총 호출수     | 성공률
------------------------------------------
grasp              | 1          | 100%

OBJECT NOUN 통계 리포트
조건 이름             | 총 호출수     | 성공률
------------------------------------------
cylinder           | 1          | 100%

PUSH GOAL 통계 리포트
조건 이름             | 총 호출수     | 성공률
------------------------------------------
==========================================
결과 CSV 저장 완료: ./경로.csv
==========================================
```






---

# Raccoonbot_Openvla

⭐ 1~3번은 직접 finetuning을 진행하는 내용이니 체크포인트를 불러와서 사용하는 경우 0번과 4번만 진행<br>

0~3번 server에서 실행, 4번 local-server 실행<br>


## 0. Dependencies
```
git clone https://github.com/KWU-FAIR-LAB/Raccoonbot_Openvla.git
```

필요한 패키지 설치
```
apt update
apt install -y \
  libegl1 \
  libgl1 \
  libglvnd0 \
  libglx0 \
  libopengl0 \
  libgles2 \
  libegl1-mesa \
  libegl1-mesa-dev \
  mesa-utils

cd Raccoonbot_Openvla/openvla
pip install .
```

## 1. Dataset 생성
MuJoCo 가상환경에서 finetuning을 위한 데이터를 수집 <br>
(main 함수 `num_episodes`으로 dataset sample 수 변경 가능)
```
cd /data/Raccoonbot_Openvla/Mujoco
python raccoon_grasp_multicolor_scene_dataset.py
```
실행하면 /data/Raccoonbot_Openvla/Mujoco/raccoon_grasp_colored_cylinder 하위에 episode별로 dataset png 확인 가능

## 2. rlds 파일 변환
raw data를 rlds builder에 맞게 변경
아래 명령문 그대로 실행
```
cd /data/Raccoonbot_Openvla/Mujoco/raccoon_dataset
python convert_raw_to_openvla_rlds_intermediate.py \
--raw_root /data/Raccoonbot_Openvla/Mujoco/raccoon_grasp_colored_cylinder \
--out_root /data/Raccoonbot_Openvla/Mujoco/raccoon_dataset/openvla_rlds_intermediate \
--val_ratio 0.1
```

## 2-1. rlds builder
rlds builder 실행
아래 명령문 그대로 실행
```
cd /data/Raccoonbot_Openvla/Mujoco/rlds_dataset_builder/raccoon_pick_place
tfds build --overwrite
```
실행하면 root 하위에 tensorflow_datasets 폴더 생성됨
```
mv /root/tensorflow_datasets /data/Raccoonbot_Openvla/
```

## 3. Raccoonbot 기반 OpenVLA finetuning
아래 명령어 그대로 실행 <br>
(`max_steps`, `save_steps` 변경 가능)
```
cd /data/Raccoonbot_Openvla/openvla
export PYTHONPATH=/data/Raccoonbot_Openvla/openvla:$PYTHONPATH

WANDB_MODE=disabled CUDA_VISIBLE_DEVICES=0 \
torchrun --standalone --nnodes 1 --nproc-per-node 1 vla-scripts/finetune.py \
  --vla_path openvla/openvla-7b \
  --data_root_dir /data/Raccoonbot_Openvla/tensorflow_datasets \
  --dataset_name raccoon_pick_place \
  --run_root_dir /data/Raccoonbot_Openvla/openvla/openvla-runs \
  --adapter_tmp_dir /data/Raccoonbot_Openvla/openvla/openvla-adapter-tmp \
  --lora_rank 32 \
  --batch_size 8 \
  --grad_accumulation_steps 2 \
  --learning_rate 5e-4 \
  --max_steps 30000 \
  --save_steps 30000 \
  --run_id_note raccoon-eef-v100
```

## 4. Mujoco 환경 Inference (local-server)
1~3번을 진행했다면 4-1은 건너뛰고 이후 명령어에서 본인이 finetuning한 모델 경로로 modelpath를 변경하여 진행

## 4-1. Hugging Face에서 RaccoonBot finetuned OpenVLA 모델 다운로드
서버에서 terminal에 아래 명령어를 입력하여 모델 다운로드
```
pip install -U huggingface_hub

hf download fair-lab/openvla-7b-finetuned-raccoonbot --local-dir /data/Raccoonbot_Openvla/openvla/openvla-runs/openvla-7b-finetuned-raccoonbot
``` 

## 4-2. 서버측 코드 실행
server 실행 명령문<br>
만약 1~3번을 진행하여 직접 finetuning했다면 model path를 openvla-runs/ 아래에 있는 모델 디렉토리로 변경하고 진행<br>
```
cd /data/Raccoonbot_Openvla/openvla
CUDA_VISIBLE_DEVICES=0 python openvla_server.py \
  --model_path /data/Raccoonbot_Openvla/openvla/openvla-runs/openvla-7b-finetuned-raccoonbot \
  --default-unnorm-key raccoon_pick_place \
  --host 0.0.0.0 \
  --port 8000 \
  --device cuda
```

## 4-3. 클라이언트측에서 실행할 환경 설정
클라이언트측 코드와 MuJoCo xml 파일 [다운로드](https://drive.google.com/drive/folders/1xrH3FoTfKC9CiUE-kDRorxTKMMq0O7Px?usp=sharing) 후 압축 풀기 <br>
파일: openvla_multicolor_client.py, openvla_multicolor_client_real_robot.py, raccoon_env.py, Raccoon_colored_cylinder.xml, RaccoonBot_S.xml, requirements.txt

VSCode로 압축 풀은 상위 폴더를 열고 terminal에서 환경설정
```
pip install -r requirments.txt
```

## 4-4. 클라이언트측 코드 실행
target_color를 **[red, blue, green, yellow]** 로 수정하면 그에 맞게 prompt가 변경됨

⭐ local 실행 명령문
```
python openvla_multicolor_client.py --server_url http://127.0.0.1:8000 --xml_path Raccoon_colored_cylinder.xml --target_color red --use_viewer
```

## 4-5. 실제 라쿤봇을 연결하여 실행
openvla_multicolor_client_real_robot.py를 실행하면 MuJoCo 환경에서 동작하는 Action을 로봇이 동일하게 수행

⭐ local 실행 명령문
```
python openvla_multicolor_client_real_robot.py --server_url http://127.0.0.1:8000 --target_color red --use_real_robot --use_viewer
```
