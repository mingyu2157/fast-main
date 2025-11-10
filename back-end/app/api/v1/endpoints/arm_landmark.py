# app/api/v1/endpoints/arm_landmark.py
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, conlist
from typing import List, Literal, Optional, Dict
import numpy as np
import math

from app.services.inference.arm_xgb_runner import predict_proba_and_label
from app.services.features.arm_features import (
    FEATURE_COLS,
    ordered_feature_items,  # 옵션: 필요 없으면 import 제거
    features_vector,
    format_features,
)

router = APIRouter(tags=["arm_landmark"])

# ==== 스키마 ====
class LMPoint(BaseModel):
    # 정규화 좌표(0~1) 기대. 실제 범위 체크/클램프는 서버에서 처리
    x: float
    y: float
    # 선택: MediaPipe 랜드마크 인덱스(0~20). 있으면 정확 슬롯에 매핑
    i: Optional[int] = None

class HandPack(BaseModel):
    label: Literal["Left", "Right", "Unknown"]
    # TIP 5개(4,8,12,16,20)만 보내도 되고, 전체 21개를 보내도 됨
    points: conlist(LMPoint, min_items=5, max_items=21)

class Payload(BaseModel):
    frame_width: int = Field(..., gt=0)
    frame_height: int = Field(..., gt=0)
    start: List[HandPack]
    end: List[HandPack]
    # 프론트 프리뷰가 selfieMode(좌우 반전)이면 True
    wasMirrored: Optional[bool] = False

# ==== 유틸 ====
TIP_IDX = [4, 8, 12, 16, 20]

def _to_xy(hands: List[HandPack], w: int, h: int, unmirror: bool) -> Dict[str, Optional[np.ndarray]]:
    """
    반환: {"Left": (21,2) ndarray or None, "Right": (21,2) ndarray or None}
    - 들어온 포인트가 21개가 아니어도 21 슬롯(0~20)을 항상 생성
    - p.i가 있으면 그 인덱스에, 없고 5개면 TIP_IDX(4,8,12,16,20)에 순서대로 매핑
    - 그 외는 0..20 순서대로 채움
    - 좌표 0~1로 클램프 후 픽셀 스케일
    - unmirror=True면 x만 원복
    """
    out = {"Left": None, "Right": None}
    for hp in hands:
        if hp.label not in ("Left", "Right"):
            continue

        slots = [[0.0, 0.0] for _ in range(21)]
        pts_in = hp.points

        if all(p.i is not None for p in pts_in):
            pairs = [(int(p.i), p) for p in pts_in if 0 <= int(p.i) <= 20]
        elif len(pts_in) == 5:
            pairs = list(zip(TIP_IDX, pts_in))
        else:
            pairs = list(zip(range(min(21, len(pts_in))), pts_in))

        for idx, p in pairs:
            px = max(0.0, min(1.0, float(p.x)))
            py = max(0.0, min(1.0, float(p.y)))
            x = px * w
            y = py * h
            if unmirror:
                x = w - x
            slots[idx] = [x, y]

        out[hp.label] = np.array(slots, dtype=np.float32)

    return out

def _angle_deg(p1, p2) -> float:
    dx = float(p2[0] - p1[0])
    dy = float(p2[1] - p1[1])
    return abs(dy / dx)  # [-180, 180]
    

def _features_from_xy(s_xy, e_xy, H: int) -> Dict[str, float]:
    """
    피처:
      - 엄지(4) ↔ 새끼(20) 선의 각도(시작/끝/차이)
      - TIP 5개(4,8,12,16,20)의 y변화량(프레임 높이로 정규화)
    """
    def hand_feats(hand: str):
        s_hand, e_hand = s_xy.get(hand), e_xy.get(hand)
        

        def ang(hxy):
            if hxy is None:
                return 0.0
            return _angle_deg(hxy[4], hxy[20])

        s = ang(s_hand)
        e = ang(e_hand)
        diff = abs(e - s)

        youts = []
        for idx in TIP_IDX:
            if s_hand is None or e_hand is None:
                youts.append(0.0)
            else:
                youts.append(float((e_hand[idx][1] - s_hand[idx][1]) / max(H, 1)))

        return s, e, diff, youts

    ls, le, ld, ly = hand_feats("Left")
    rs, re, rd, ry = hand_feats("Right")

    feats = {
        "left_start_slope": ls, "left_end_slope": le, "left_slope_diff": ld,
        "right_start_slope": rs, "right_end_slope": re, "right_slope_diff": rd,
        "left_y0": ly[0], "left_y1": ly[1], "left_y2": ly[2], "left_y3": ly[3], "left_y4": ly[4],
        "right_y0": ry[0], "right_y1": ry[1], "right_y2": ry[2], "right_y3": ry[3], "right_y4": ry[4],
    }
    for k in FEATURE_COLS:
        feats.setdefault(k, 0.0)
    return feats

# ==== 엔드포인트 ====
@router.post("/predict_landmarks")
def predict_from_landmarks(body: Payload):
    try:
        w = int(body.frame_width)
        h = int(body.frame_height)
        if not (w > 0 and h > 0):
            raise ValueError("invalid frame size")

        s_xy = _to_xy(body.start, w, h, unmirror=bool(body.wasMirrored))
        e_xy = _to_xy(body.end,   w, h, unmirror=bool(body.wasMirrored))

        feats = _features_from_xy(s_xy, e_xy, H=h)
        # 디버그 로그(서버 콘솔)
        print("[ARM] features:", format_features(feats))

        proba, label = predict_proba_and_label(feats)
        return {
            "result": label,
            "proba": proba,
            "features": feats,                         # dict (키-값)
            "features_order": FEATURE_COLS,            # 컬럼 순서
            "features_vector": features_vector(feats), # 모델 입력 벡터
            "features_pretty": format_features(feats), # 사람이 보기 좋은 문자열
        }
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"invalid landmarks: {e}")
