import pandas as pd

def load_building_info(path='data/building_info.csv'):
    building = pd.read_csv(path, encoding='utf-8-sig')
    building['냉방률'] = building['냉방면적(m2)'] / building['연면적(m2)']
    # 문자형 숫자 처리 및 결측치 0 대체
    for col in ['태양광용량(kW)', 'ESS저장용량(kWh)', 'PCS용량(kW)']:
        building[col] = building[col].replace('-', 0)
        building[col] = pd.to_numeric(building[col], errors='coerce')

    return building

def split_building_types(
    building: pd.DataFrame,
    building_types: list[str] | None = None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    선택된 건물 유형과 그 외를 분리하여 반환한다.
    building_types가 빈 리스트 또는 None이면 selected는 빈 DF, rest는 전체.
    """
    if building_types:
        mask = building['건물유형'].isin(building_types)
    else:
        mask = pd.Series(False, index=building.index)

    selected = building[mask]
    rest = building[~mask]
    return selected, rest

def encode_building(building: pd.DataFrame) -> pd.DataFrame:
    """
    building DataFrame에 대해 one-hot encoding 수행
    """
    building_encoded = pd.get_dummies(building)
    return building_encoded

def load_and_merge(path='data/train.csv', building_info=None):
    dataset = pd.read_csv(path, encoding='utf-8-sig')
    return pd.merge(dataset, building_info, on='건물번호', how='right')