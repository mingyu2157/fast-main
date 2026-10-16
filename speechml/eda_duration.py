"""TORGO 발화 길이 EDA — meta.csv를 읽어 자가 검증과 길이 분포를 출력한다."""
import pandas as pd

META = 'speechml/outputs/meta.csv'


def check_basics(d: pd.DataFrame) -> None:
    """데이터 무결성 자가 검증."""
    print('=== 자가 검증 ===')
    print(f"행 수: {len(d)}")
    print(f"speaker_id 개수: {d['speaker_id'].nunique()}")
    print(f"speaker_id 종류: {sorted(d['speaker_id'].unique())}")
    print(f"is_dysarthria: {d['is_dysarthria'].value_counts().to_dict()}")
    print(f"gender: {d['gender'].value_counts().to_dict()}")
    print(f"sr 종류: {d['sr'].unique().tolist()}")
    print(f"duration_sec NaN: {d['duration_sec'].isna().sum()} / 0초: {(d['duration_sec'] == 0).sum()}")
    print(f"duration_sec 통계량:\n{d['duration_sec'].describe()}")
    print(f"session별 개수:\n{d['session'].value_counts()}")
    print()


def duration_by_utterance(d: pd.DataFrame) -> None:
    """방법 1 — 발화 단위로 클래스 비교. 화자별 발화 수 편차에 취약하다."""
    print('=== 방법 1 — 발화 단위 (n=1999) ===')
    print(d.groupby('is_dysarthria')['duration_sec'].describe().to_string())
    print()


def duration_by_speaker(d: pd.DataFrame) -> None:
    """방법 2 — 화자별 중앙값을 먼저 구해 한 사람당 한 표로 비교."""
    print('=== 방법 2 — 화자 단위 (n=15) ===')
    per_speaker = (
        d.groupby(['speaker_id', 'is_dysarthria'])
         .agg(median_sec=('duration_sec', 'median'),
              n_utt=('duration_sec', 'size'))
         .reset_index()
    )

    print('[화자별 중앙값 — 짧은 순]')
    print(per_speaker.sort_values('median_sec').to_string(index=False))
    print()

    print('[화자 15명을 클래스별로 비교]')
    print(per_speaker.groupby('is_dysarthria')['median_sec'].describe().to_string())
    print()


def short_file_breakdown(d: pd.DataFrame) -> None:
    """짧은 파일이 어느 클래스에 쏠려 있는지 — 필터 기준 판단용."""
    print('=== 짧은 파일 분포 ===')
    for t in (0.1, 0.3, 0.5, 1.0, 1.5):
        sub = d[d['duration_sec'] < t]
        print(f'  < {t:>4}s : {len(sub):4d}개  {sub["is_dysarthria"].value_counts().to_dict()}')
    print()


def main() -> None:
    d = pd.read_csv(META)
    check_basics(d)
    duration_by_utterance(d)
    duration_by_speaker(d)
    short_file_breakdown(d)


if __name__ == '__main__':
    main()
