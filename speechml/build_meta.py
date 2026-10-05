"""TORGO 메타데이터 테이블 생성 — 2,000개 wav를 한 번 훑어 요약표를 만든다."""
from pathlib import Path
import pandas as pd
import numpy as np
import librosa

DATA_CSV = Path('torgo_data/data.csv')
OUT_CSV = Path('speechml/outputs/meta.csv')

def parse_filename(rel_path: str) -> tuple[str, str] :
    """ID와 세션을 추출
    'torgo_data/dysarthria_male/M01_Session1_0001.wav' -> ('M01', 'Session1')"""
    parts = Path(rel_path).parts
    speaker_id = parts[-1].split('_')[0] # 'M01'
    session = parts[-1].split('_')[1] # 'Session1'
    return speaker_id, session

def probe_audio(path: Path) -> dict:
    """wav 파일을 읽어 샘플링레이트(sr) / duration_sec / n_samples / rms_mean / peak 를 담은 dict를 반환"""
    y, sr = librosa.load(path, sr=None)
    n_samples = len(y)
    duration_sec = n_samples / sr
    peak = np.abs(y).max()
    rms_mean = librosa.feature.rms(y=y).mean()

    return {
        'sr': sr,
        'duration_sec': duration_sec,
        'n_samples': n_samples,
        'rms_mean': rms_mean,
        'peak': peak
    }

def build_table(labels: pd.DataFrame) -> tuple[pd.DataFrame, list]:
    """labels를 순회하며 행을 쌓음, (완성된 테이블, 실패목록) 반환"""
    rows = []
    failed = []
    for i, row in labels.iterrows():
        rel_path = row['filename']
        if i % 100 == 0: # 100개마다 한 줄씩 찍기
            print(f"Processing row {i}")
        try:
            speaker_id, session = parse_filename(rel_path)
            audio_info = probe_audio(Path(rel_path))
            rows.append({
                'rel_path' : rel_path,
                'speaker_id' : speaker_id,
                'is_dysarthria' : row['is_dysarthria'],
                'gender' : row['gender'],
                'session' : session,
                **audio_info
            })
        except Exception as e:
            print(f"Failed to process {i}: {e}")
            failed.append((rel_path, e))
    return pd.DataFrame(rows), failed

def main():
    # 1. data.csv 읽기 2. build_table 호출 3. 실패 보고 4. meta.csv 저장
    labels = pd.read_csv(DATA_CSV)
    meta_df, failed = build_table(labels)
    if failed:
        print(f'{len(failed)} failed. 처음 10건:')
        for path, err in failed[:10]:
            print(f'- {path}: {err}')


    OUT_CSV.parent.mkdir(parents = True, exist_ok = True)
    meta_df.to_csv(OUT_CSV, index=False)
    print(f'{len(meta_df)} rows -> {OUT_CSV}')

if __name__ == '__main__':
    main()
