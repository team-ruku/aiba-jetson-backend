# AIBA Backend for Standalone-Hardware

> FastAPI-based AIBA Backend for Standalone-Hardware

## Disclaimer

이 백엔드 Instance는 YOLO 및 Monocular Depth Estimation + TDoA 비디오 스트림을 위한 하드웨어용 백엔드입니다.

CUDA (엔비디아) 외 Apple Silicon (MPS) 또한 지원하나, Apple Silicon 환경에서의 YOLO-slowfast 구동은 CPU를 이용하니 ([#1](../../issues/1)), **CUDA 구동을 권장합니다.**

## Usage

실행 전, YOLO-slowfast 모델을 `yolo/deep_sort/deep_sort/deep/checkpoint/ckpt.t7` 에, Mono Depth Estimation 모델을 `vision/weights/dpt_swin2_tiny_256.pt` 에 넣어주세요.

### Initial Run

```bash
conda create -n {your_env_name} python=3.9

pip install -r requirements.txt
```

### Run

```bash
uvicorn --host=0.0.0.0 app:app
```

## Troubleshooting

If CUDA or Pytorch doesn't work, try this

```bash
pip3 install —pre torch torchvision torchaudio —index-url https://download.pytorch.org/whl/nightly/cu121
```

When PyTorch Apple Silicon backend (MPS) doesn't work, add this variable

```bash
conda env config vars set PYTORCH_ENABLE_MPS_FALLBACK=1
```
