# AIBA Backend

> FastAPI-based AIBA Backend

## Disclaimer

이 백엔드 Instance는 YOLO 및 Monocular Depth Estimation 비디오 스트림과 장치 등록 등을 위한 중간 단계용 백엔드입니다.

오로지 테스트 목적을 위한 것이기에, 아무런 User Authentication (JWT Token, Bearer Auth 등) 이 없습니다.

## Usage

실행 전, YOLO-slowfast 환경이 셋팅되어 있어야 함

### Initial Run

```bash
conda activate {env_name}

pip install -r requirements.txt
```

### Run

```bash
poetry run uvicorn --host=0.0.0 app:app
```
