import os
import platform
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------
# 1. 페이지 설정 및 한글 폰트 환경 강제 등록
# ---------------------------------------------------------
st.set_page_config(
    page_title="서울시 심야 교통사고 요인 분석 (20920·20922)",
    page_icon="🚦",
    layout="wide",
)

# Seaborn 스타일을 먼저 적용한 뒤 한글 폰트를 덮어써야 폰트가 초기화되지 않습니다.
sns.set_style("whitegrid")


def setup_korean_font():
    # 1순위: koreanize_matplotlib 내장 TTF 폰트 직접 등록 (리눅스·클라우드 환경 대응)
    try:
        import koreanize_matplotlib  # noqa: F401
        font_dir = os.path.join(os.path.dirname(koreanize_matplotlib.__file__), "fonts")
        for fname in os.listdir(font_dir):
            if fname.endswith(".ttf"):
                fpath = os.path.join(font_dir, fname)
                fm.fontManager.addfont(fpath)
                plt.rcParams["font.family"] = fm.FontProperties(fname=fpath).get_name()
                plt.rcParams["axes.unicode_minus"] = False
                return
    except Exception:
        pass

    # 2순위: 운영체제별 기본 한글 폰트 탐색
    system_name = platform.system()
    if system_name == "Windows":
        plt.rcParams["font.family"] = "Malgun Gothic"
    elif system_name == "Darwin":
        plt.rcParams["font.family"] = "AppleGothic"
    else:
        for path in [
            "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
            "/usr/share/fonts/truetype/nanum/NanumBarunGothic.ttf",
        ]:
            if os.path.exists(path):
                fm.fontManager.addfont(path)
                plt.rcParams["font.family"] = fm.FontProperties(fname=path).get_name()
                break
        else:
            plt.rcParams["font.family"] = "NanumGothic"

    plt.rcParams["axes.unicode_minus"] = False


setup_korean_font()

# 서울시 25개 자치구 개방자치단체코드 매핑 (OA-23056 원본 컬럼 대응)
SEOUL_GU_CODE_MAP = {
    "3000000": "종로구", "3010000": "중구", "3020000": "용산구",
    "3030000": "성동구", "3040000": "광진구", "3050000": "동대문구",
    "3060000": "중랑구", "3070000": "성북구", "3080000": "강북구",
    "3090000": "도봉구", "3100000": "노원구", "3110000": "은평구",
    "3120000": "서대문구", "3130000": "마포구", "3140000": "양천구",
    "3150000": "강서구", "3160000": "구로구", "3170000": "금천구",
    "3180000": "영등포구", "3190000": "동작구", "3200000": "관악구",
    "3210000": "서초구", "3220000": "강남구", "3230000": "송파구",
    "3240000": "강동구",
}


# ---------------------------------------------------------
# 2. 데이터 로드 및 OA-23056(일반음식점 인허가) 원본 CSV 처리 함수
# ---------------------------------------------------------
def read_csv_auto_encoding(file_or_path) -> pd.DataFrame:
    """서울열린데이터광장 CSV(CP949 / UTF-8-SIG / UTF-8) 인코딩 자동 판별"""
    for enc in ("cp949", "utf-8-sig", "utf-8"):
        try:
            if hasattr(file_or_path, "seek"):
                file_or_path.seek(0)
            return pd.read_csv(file_or_path, encoding=enc, low_memory=False)
        except UnicodeDecodeError:
            continue
    raise ValueError("CSV 파일 인코딩을 읽을 수 없습니다.")


def parse_oa23056_restaurant_csv(raw_df: pd.DataFrame) -> pd.Series:
    """
    서울열린데이터광장 OA-23056 (서울시 일반음식점 인허가 정보) 원본 CSV에서
    '영업/정상' 상태인 업소만 필터링하여 25개 자치구별 개수를 집계
    """
    df = raw_df.copy()

    # 1) 영업 중인 업소만 필터링 (폐업 제외)
    if "영업상태명" in df.columns:
        df = df[df["영업상태명"].astype(str).str.contains("영업", na=False)]
    elif "영업상태구분코드" in df.columns:
        df = df[df["영업상태구분코드"].astype(str).str.strip().isin(["1", "01"])]

    # 2) 자치구명 추출 (개방자치단체코드 우선, 없으면 주소 문자열에서 추출)
    if "개방자치단체코드" in df.columns:
        codes = df["개방자치단체코드"].astype(str).str.replace(".0", "", regex=False).str.strip()
        df["자치구명"] = codes.map(SEOUL_GU_CODE_MAP)
    else:
        df["자치구명"] = np.nan

    addr_col = next(
        (c for c in ["소재지전체주소", "도로명전체주소", "소재지도로명주소"] if c in df.columns),
        None,
    )
    if addr_col is not None:
        extracted = (
            df[addr_col]
            .astype(str)
            .str.extract(r"서울특별시\s+([가-힣]+구)")[0]
        )
        df["자치구명"] = df["자치구명"].fillna(extracted)

    return df.dropna(subset=["자치구명"]).groupby("자치구명").size()


@st.cache_data
def load_base_table() -> pd.DataFrame:
    """서울시 25개 자치구 기본 기준 테이블 (생활인구·거주인구·면적·심야교통사고)"""
    data = [
        ("강남구", 39.50, 545000, 685000, 12450, 1890),
        ("강동구", 24.59, 463000, 412000, 4620, 640),
        ("강북구", 23.60, 292000, 268000, 3410, 495),
        ("강서구", 41.45, 568000, 515000, 6380, 890),
        ("관악구", 29.57, 498000, 472000, 5290, 780),
        ("광진구", 17.06, 348000, 365000, 4850, 690),
        ("구로구", 20.12, 415000, 388000, 4410, 670),
        ("금천구", 13.02, 241000, 229000, 3180, 480),
        ("노원구", 35.44, 502000, 435000, 3920, 560),
        ("도봉구", 20.65, 309000, 264000, 2540, 375),
        ("동대문구", 14.22, 354000, 349000, 4590, 685),
        ("동작구", 16.36, 389000, 362000, 3680, 540),
        ("마포구", 23.85, 375000, 468000, 8420, 1210),
        ("서대문구", 17.63, 318000, 325000, 4120, 580),
        ("서초구", 46.98, 412000, 495000, 7650, 1340),
        ("성동구", 16.86, 284000, 305000, 4290, 620),
        ("성북구", 24.58, 438000, 398000, 4150, 595),
        ("송파구", 33.87, 658000, 624000, 7890, 1180),
        ("양천구", 17.41, 439000, 378000, 3490, 510),
        ("영등포구", 24.55, 398000, 452000, 7310, 1150),
        ("용산구", 21.87, 227000, 298000, 5340, 790),
        ("은평구", 29.71, 468000, 395000, 3980, 550),
        ("종로구", 23.91, 150000, 265000, 7120, 860),
        ("중구", 9.96, 131000, 248000, 6480, 820),
        ("중랑구", 18.50, 387000, 342000, 3890, 575),
    ]
    columns = [
        "자치구명",
        "행정구역_면적",
        "단순_거주인구",
        "심야_유동인구",
        "일반음식점_수",
        "심야_교통사고_건수",
    ]
    return pd.DataFrame(data, columns=columns)


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """자치구 면적 및 인구 규모 편향을 보정하기 위한 파생변수(밀도·비율) 생성"""
    df = df.copy()
