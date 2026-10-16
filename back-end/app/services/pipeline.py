from typing import Dict, Tuple
from app.core.config import Modality
from app.services.features.face_features import extract_features_from_image_bytes as face_from_image
from app.services.inference.tabnet_runner import predict_proba_and_label
# NOTE: arm_landmarks 경로의 의존성은 지연 import 한다.
#   arm_features.features_from_landmarks 가 아직 구현되지 않아 모듈 상단에서
#   import 하면 ImportError 로 앱 전체가 기동 불가가 된다(measure.py → pipeline.py).
#   깨진 모듈이 다른 기능까지 막지 않도록 호출 시점으로 미룬다.

def run_pipeline(modality: Modality, payload: bytes) -> Tuple[Dict[str, float], float, int]:
    if modality == "face":  
        feats, _ = face_from_image(payload)
        if not feats: raise ValueError("얼굴 인식 실패")
    elif modality == "arm":
        # 레거시(이미지 업로드) 경로는 더 이상 사용 안 함.
        # 필요하면 /api/v1/arm/predict_landmarks 로 전환하세요.
        raise RuntimeError("Use 'arm_landmarks' modality (A-path) instead of legacy image-based 'arm'.")
    elif modality == "arm_landmarks":
        import numpy as np   # 상단에 없어서 NameError 가 나던 것도 함께 보완
        from app.services.features.arm_features import features_from_landmarks
        from app.services.inference.arm_xgb_runner import predict_proba_and_label as arm_predict
        from app.services.arm_result import compose_arm_result

        # payload 예시:
        # {"frame_width":1280,"frame_height":720,"wasMirrored":true,
        #  "start":[{"label":"Left","points":[{x,y}*21]},...],
        #  "end":[...]}
        w = int(payload["frame_width"]);  h = int(payload["frame_height"])
        start = payload["start"];         end = payload["end"]
        was_mir = bool(payload.get("wasMirrored", False))

        def to_xy(hands):
            out = {"Left": None, "Right": None}
            for hp in hands:
                label = hp.get("label")
                if label not in ("Left", "Right"):
                    continue
                pts = []
                for p in hp.get("points", []):
                    x = float(p["x"]) * w
                    y = float(p["y"]) * h
                    if was_mir:
                        x = w - x   # x만 원복
                    pts.append([x, y])
                if pts:
                    out[label] = np.array(pts, dtype=np.float32)
            return out

        s_xy = to_xy(start)
        e_xy = to_xy(end)
        feats = features_from_landmarks(s_xy, e_xy, H=h)
        proba, label = arm_predict(feats)
        text = compose_arm_result(proba, label, feats)
        return {"proba": proba, "label": label, "text": text, "features": feats}
    elif modality == "speech":
        raise NotImplementedError("speech pipeline 준비 중")
    else:
        raise ValueError("Unknown modality")
    proba, label = predict_proba_and_label(modality, feats)
    return feats, proba, label
