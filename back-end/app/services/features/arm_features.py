# back-end/app/services/features/arm_features.py
from __future__ import annotations
import io, math
import cv2, numpy as np
import mediapipe as mp
from typing import Dict, Tuple, Optional

FEATURE_COLS = [
    "left_start_slope","left_end_slope","left_slope_diff",
    "right_start_slope","right_end_slope","right_slope_diff",
    "left_y0","left_y1","left_y2","left_y3","left_y4",
    "right_y0","right_y1","right_y2","right_y3","right_y4",
]

mp_hands = mp.solutions.hands

def _decode_bgr(image_bytes: bytes) -> np.ndarray:
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("이미지 디코딩 실패")
    return img

def _extract_xy21(img_bgr: np.ndarray) -> Dict[str, Optional[np.ndarray]]:
    h, w = img_bgr.shape[:2]
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    with mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.5) as hands:
        res = hands.process(rgb)

    out: Dict[str, Optional[np.ndarray]] = {"Left": None, "Right": None}
    if not res.multi_hand_landmarks or not res.multi_handedness:
        return out

    for lm, handed in zip(res.multi_hand_landmarks, res.multi_handedness):
        label = handed.classification[0].label  # "Left" / "Right"
        pts = np.array([[int(p.x * w), int(p.y * h)] for p in lm.landmark], dtype=np.int32)  # (21,2)
        out[label] = pts
    return out

def _slope_xy(p1: np.ndarray, p2: np.ndarray) -> float:
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    if dx == 0:
        return float("inf") if dy >= 0 else -float("inf")
    return dy / dx

def extract_features_from_two_images(start_bytes: bytes, end_bytes: bytes) -> Dict[str, float]:
    start = _decode_bgr(start_bytes)
    end = _decode_bgr(end_bytes)

    s_xy = _extract_xy21(start)
    e_xy = _extract_xy21(end)

    def hand_feats(hand: str):
        # landmark 4(엄지끝) ↔ 20(새끼끝)
        def slope(hxy):
            if hxy is None: return 0.0
            return float(_slope_xy(hxy[4], hxy[20]))
        s = slope(s_xy.get(hand))
        e = slope(e_xy.get(hand))
        diff = abs(e - s)
        # 첫 5개 랜드마크 y 변화량
        def y_deltas():
            TIP_IDX = [4, 8, 12, 16, 20]  # thumb, index, middle, ring, pinky (tip)
            out = []
            s_hand = s_xy.get(hand)
            e_hand = e_xy.get(hand)
            for idx in TIP_IDX:
                if s_hand is None or e_hand is None:
                    out.append(0.0)
                else:
                    # 이미지 좌표계에서 y는 아래로 증가 → 아래로 내려가면 +값
                    out.append(float(e_hand[idx][1] - s_hand[idx][1]))
            return out
        return s, e, diff, y_deltas()

    ls, le, ld, ly = hand_feats("Left")
    rs, re, rd, ry = hand_feats("Right")

    feats = {
        "left_start_slope": ls, "left_end_slope": le, "left_slope_diff": ld,
        "right_start_slope": rs, "right_end_slope": re, "right_slope_diff": rd,
        "left_y0": ly[0], "left_y1": ly[1], "left_y2": ly[2], "left_y3": ly[3], "left_y4": ly[4],
        "right_y0": ry[0], "right_y1": ry[1], "right_y2": ry[2], "right_y3": ry[3], "right_y4": ry[4],
    }
    # 누락 키 없이 보장
    for k in FEATURE_COLS:
        feats.setdefault(k, 0.0)
    return feats
