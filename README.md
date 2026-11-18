# GAMST Coin Spin Halo

**비싼 VLM이 모든 프레임을 보는 대신, 저렴한 OpenCV 신호가 회전을 세고 GAMST가 두 번만 모델을 부릅니다.**

이 저장소는 GAMST의 halo demo입니다. 동전처럼 보이는 물체를 프레임마다 추적하고, 타원의 단축/장축 비율이 주기적으로 줄어드는 것을 이용해 회전 수와 RPM을 계산합니다. 스핀 시작과 종료에만 VLM 확인을 요청하며, VLM 설정이 없으면 네트워크 호출 없이 전체 분석이 완료됩니다.

## 왜 GAMST인가?

회전 계산 자체를 직접 구현할 수는 있습니다. GAMST가 제공하는 것은 다음 공통 계층입니다.

- 모든 프레임의 측정값에 정렬된 캡처 시각, 센서 epoch, evidence ID를 부여
- 등록된 `CapabilitySpec`으로 측정 스키마와 provenance 검증
- 시간이 정렬된 bounded temporal memory와 packet checksum 제공
- `CallPolicy`로 `small_vlm` escalation을 결정하고 실제 호출 수와 분리된 정책 근거 기록

VLM은 회전 수를 계산하지 않습니다. 시작 프레임에서 장면을 확인하고, 종료 프레임에서 최종 면(heads/tails/uncertain)과 숫자의 타당성을 확인하는 소비자입니다.

## 설치

Python 3.10+가 필요합니다. 이 데모는 로컬 GAMST checkout을 사용하며 해당 저장소를 수정하지 않습니다.

```bash
cd /Users/jinminseong/Desktop/GamstCapture
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[test]'

# 로컬 GAMST SDK (dependency-free core)
export PYTHONPATH=/Users/jinminseong/Desktop/catch-your-mind${PYTHONPATH:+:$PYTHONPATH}
```

다른 위치의 GAMST checkout을 사용한다면 해당 경로로 바꾸거나, 그 checkout에서 `python -m pip install -e .`를 실행하면 됩니다.

## 바로 실행

### 1. 결정론적 합성 영상 생성 및 분석

```bash
python -m gamst_capture generate-synthetic --output outputs/synthetic_coin.mp4
python -m gamst_capture run outputs/synthetic_coin.mp4 \
  --report outputs/report.json \
  --annotated-video outputs/annotated.mp4 \
  --no-vlm
python -m pytest
```

생성기는 고정 seed, 가우시안 blur, 센서 noise, 변하는 각속도를 사용하며 정확히 8회전을 렌더링합니다. `.truth.json` sidecar에 정답을 기록하지만 perception 코드는 이를 읽지 않습니다. CLI는 sidecar가 있을 때만 정답과 비교합니다.

### 2. 일반 영상 또는 웹캠

```bash
python -m gamst_capture run my_coin.mp4 --report outputs/my-report.json \
  --annotated-video outputs/my-annotated.mp4

# 표시 창에서는 q, 터미널에서는 Ctrl-C로 종료
python -m gamst_capture run --webcam 0 --display \
  --report outputs/webcam-report.json --annotated-video outputs/webcam.mp4
```

웹캠 장치가 없으면 OpenCV가 source를 열 수 없다는 오류를 반환합니다. `--max-frames N`으로 처리 길이를 제한할 수 있습니다.

### 3. 선택적 OpenAI-compatible VLM

```bash
export GAMSTCAPTURE_VLM_BASE_URL=http://127.0.0.1:8080/v1
export GAMSTCAPTURE_VLM_MODEL=my-vision-model
export GAMSTCAPTURE_VLM_API_KEY=...  # 로컬 endpoint가 요구하지 않으면 생략
python -m gamst_capture run outputs/synthetic_coin.mp4 \
  --report outputs/vlm-report.json --annotated-video outputs/vlm-annotated.mp4
```

base URL에 `/chat/completions`가 없으면 자동으로 붙입니다. `OPENAI_BASE_URL`, `OPENAI_MODEL`, `OPENAI_API_KEY`도 fallback으로 지원합니다. 자격 증명은 코드나 report에 저장하지 않습니다. endpoint 오류는 VLM call의 `status: error`로 기록하고 저렴한 회전 분석 결과는 유지합니다. 설정이 없으면 화면과 report에 **VLM disabled**, 실제 호출 **0**이 표시됩니다.

## 측정된 합성 실행

2026-11-18, 이 저장소의 기본 fixture(320×320, 30 FPS)에 대해 위 CLI를 직접 실행한 결과:

| 항목 | 측정값 |
| --- | ---: |
| 처리 프레임 | 240 |
| 정답 / 검출 회전 수 | 8 / **8** |
| 정답 / 검출 스핀 시작 | 1.000 s / **1.033 s** |
| 정답 / 검출 스핀 종료 | 6.000 s / **6.033 s** |
| 평균 RPM | **96.00** |
| GAMST signals | 242 (프레임 240 + start/stop events 2) |
| 실제 VLM calls, VLM disabled | **0** |
| 정책상 선택된 VLM 시점 | 2 (spin start/stop) |
| 테스트 VLM을 주입한 integration test | 2 calls / spin |

즉 기본 spin 하나는 240프레임 전부를 GAMST에 기록하지만 VLM을 켰을 때의 escalation 상한은 2회입니다. 이 숫자는 token 비용 절감 벤치마크가 아니라, synthetic 실행에서 관측한 호출 횟수입니다. 실제 token 수는 endpoint가 `usage`를 반환할 때만 집계합니다.

출력:

- `outputs/annotated.mp4`: 추적 contour, axis ratio, 상태, 회전 수, RPM, frames/signals/VLM counters overlay
- `outputs/report.json`: 모든 프레임 측정, RPM timeline, spin start/stop evidence 시각, GAMST 정책 결정, temporal context checksum, VLM 응답/토큰, 정답 비교

## 알고리즘

1. HSV에서 채도가 높은 금/동색 connected component를 찾고 가장 큰 contour를 추적합니다.
2. `fitEllipse`의 단축/장축 비율을 median filter로 평활합니다.
3. 서로 다른 진입/탈출 임계값(hysteresis)으로 edge-on 통과를 검출하고 최소 반주기로 중복을 거부합니다. edge-on 두 번이 한 바퀴입니다.
4. 연속 움직임으로 spin start를, 지속된 정지로 stop을 검출합니다. 인접 edge-on 간격으로 순간 RPM을 계산합니다.
5. 매 프레임 `coin.rotation`, 경계에 `coin.spin_event`, 성공한 VLM 응답에 `coin.vlm_result`를 GAMST에 넣습니다. replay timestamp는 wall clock보다 6시간 과거에 고정해 GAMST의 future timestamp 제한과 retention을 지킵니다.

## 한계

- **실제 촬영한 동전 영상은 검증하지 않았습니다.** 정확한 8회전 주장은 제공된 렌더링 fixture에만 적용됩니다.
- 현재 segmentation은 어둡고 채도가 낮은 배경의 금/동색 동전을 전제로 합니다. 반사, 손가락 가림, 복잡한 배경, severe camera motion, 다른 노란 물체는 오탐을 만들 수 있습니다.
- 프레임레이트가 edge-on 통과를 충분히 샘플링해야 합니다. 매우 빠른 회전, 심한 motion blur 또는 0.08 s보다 짧은 반주기는 undercount될 수 있습니다.
- 두 edge-on 사건을 한 회전으로 환산하므로 시작/종료가 반바퀴에 걸치면 완전한 회전만 세고 나머지는 버립니다.
- VLM의 heads/tails 판독은 보조 응답이며 정확도를 주장하지 않습니다. 모델 응답이 없거나 실패한 경우에도 OpenCV count는 그대로입니다.

## English summary

A deterministic OpenCV/numpy coin tracker counts full rotations and RPM on every frame. Real GAMST capabilities retain timestamped measurements and temporal evidence; `CallPolicy` selects only spin-start confirmation and spin-stop final-face/count-sanity VLM calls. The default synthetic run measured **8/8 rotations over 240 frames**, 242 GAMST signals, and **0 VLM calls with VLM disabled**; the injected-VLM test makes exactly **2 calls**. Run `python -m pytest` with `PYTHONPATH` pointing to the local GAMST checkout. Real-world coin footage remains untested.
