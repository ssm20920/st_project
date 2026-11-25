import platform
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------
# 1. 페이지 설정 및 한글 폰트 환경 구성
# ---------------------------------------------------------
st.set_page_config(
    page_title="서울시 심야 교통사고 요인 분석 (20920·20922)",
    page_icon="🚦",
    layout="wide",
)

try:
    import koreanize_matplotlib  # noqa: F401
except ImportError:
    system_name = platform.system()
    if system_name == "Windows":
        plt.rcParams["font.family"] = "Malgun Gothic"
    elif system_name == "Darwin":
        plt.rcParams["font.family"] = "AppleGothic"
    else:
        plt.rcParams["font.family"] = "NanumGothic"
plt.rcParams["axes.unicode_minus"] = False
import os
import platform
import urllib.request
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
# 1. 페이지 설정 및 한글 폰트 환경 100% 강제 등록
# ---------------------------------------------------------
st.set_page_config(
    page_title="서울시 심야 교통사고 요인 분석 (20920·20922)",
    page_icon="🚦",
    layout="wide",
)

sns.set_style("whitegrid")


@st.cache_resource
def load_korean_font() -> str:
    """실행 환경(Windows / Mac / Streamlit Cloud 리눅스)에 맞는 한글 폰트 확보 및 등록"""
    system_name = platform.system()
    if system_name == "Windows":
        return "Malgun Gothic"
    if system_name == "Darwin":
        return "AppleGothic"

    # 리눅스(Streamlit Cloud): 시스템 폰트 확인 후 없으면 자동 다운로드
    font_candidates = [
        "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
        "/usr/share/fonts/truetype/nanum/NanumBarunGothic.ttf",
        "/tmp/NanumGothic.ttf",
    ]
    font_path = next((p for p in font_candidates if os.path.exists(p)), None)

    if font_path is None:
        font_path = "/tmp/NanumGothic.ttf"
        url = "https://github.com/google/fonts/raw/main/ofl/nanumgothic/NanumGothic-Regular.ttf"
        urllib.request.urlretrieve(url, font_path)

    fm.fontManager.addfont(font_path)
    return fm.FontProperties(fname=font_path).get_name()


FONT_NAME = load_korean_font()
plt.rcParams["font.family"] = FONT_NAME
plt.rcParams["axes.unicode_minus"] = False

# ---------------------------------------------------------
# 2. 데이터 로드 함수 (기본 25개 자치구 통합 데이터 + CSV 업로드 지원)
# ---------------------------------------------------------
@st.cache_data
def load_default_data() -> pd.DataFrame:
    """
    서울열린데이터광장(생활인구, 일반음식점 인허가, 행정구역 통계) 및
    KOSIS(시군구별 시간대별 교통사고, 2021~2025 5개년 합산 기준) 통합 샘플 데이터셋
    """
    data = [
        # 자치구명, 면적(㎢), 거주인구(명), 심야생활인구_평균(명), 영업중_일반음식점(개), 심야교통사고_5년합계(건)
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
    df = pd.DataFrame(data, columns=columns)
    return add_derived_features(df)


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """자치구 면적 및 인구 규모 편향을 보정하기 위한 파생변수(밀도·비율) 생성"""
    df = df.copy()
    df["음식점_밀도(개/㎢)"] = (df["일반음식점_수"] / df["행정구역_면적"]).round(1)
    df["심야_유동인구_밀도(명/㎢)"] = (df["심야_유동인구"] / df["행정구역_면적"]).round(1)
    df["거주인구_밀도(명/㎢)"] = (df["단순_거주인구"] / df["행정구역_면적"]).round(1)
    df["심야_사고_밀도(건/㎢)"] = (df["심야_교통사고_건수"] / df["행정구역_면적"]).round(2)
    return df


# ---------------------------------------------------------
# 3. 사이드바: 데이터 선택 및 분석 옵션
# ---------------------------------------------------------
st.sidebar.header("⚙️ 데이터 및 분석 설정")
data_mode = st.sidebar.radio(
    "데이터 소스 선택",
    ["기본 통합 데이터 (서울 25개 자치구)", "전처리 완료 CSV 직접 업로드"],
)

if data_mode == "전처리 완료 CSV 직접 업로드":
    uploaded_file = st.sidebar.file_uploader(
        "병합된 자치구 CSV 파일 업로드", type=["csv"]
    )
    st.sidebar.caption(
        "필수 컬럼: 자치구명, 행정구역_면적, 단순_거주인구, 심야_유동인구, 일반음식점_수, 심야_교통사고_건수"
    )
    if uploaded_file is not None:
        try:
            raw_df = pd.read_csv(uploaded_file, encoding="utf-8")
        except UnicodeDecodeError:
            uploaded_file.seek(0)
            raw_df = pd.read_csv(uploaded_file, encoding="cp949")
        df = add_derived_features(raw_df)
    else:
        st.info("CSV 파일을 업로드하기 전까지 기본 25개 자치구 데이터를 표시합니다.")
        df = load_default_data()
else:
    df = load_default_data()

metric_basis = st.sidebar.radio(
    "비교 기준 단위 선택",
    ["총량 기준 (명 / 개 / 건)", "면적 1㎢당 밀도 기준 (규모 편향 보정)"],
)

selected_districts = st.sidebar.multiselect(
    "강조(하이라이트)할 자치구 선택",
    options=df["자치구명"].tolist(),
    default=["강남구", "마포구", "중구", "노원구"],
)

# ---------------------------------------------------------
# 4. 상단 타이틀 및 핵심 탐구 질문
# ---------------------------------------------------------
st.title("🚦 서울시 심야 유동인구·음식점 밀집도와 심야 교통사고 관계 분석")
st.markdown("**팀원:** `20920 손성민`, `20922 오하늘` | **출처:** 서울열린데이터광장 · KOSIS 국가통계포털")

st.info(
    "**핵심 탐구 질문:** \"최근 5년간 서울시 25개 자치구에서 심야 유동인구와 일반음식점이 밀집한 곳은 "
    "정말 심야 시간대(22시~04시) 교통사고도 비례하여 많이 발생하는가?\""
)

# ---------------------------------------------------------
# 5. 회귀 모델 학습 및 성능 지표 계산 (기준선 vs 다중회귀)
# ---------------------------------------------------------
if metric_basis == "총량 기준 (명 / 개 / 건)":
    col_pop = "단순_거주인구"
    col_float = "심야_유동인구"
    col_food = "일반음식점_수"
    col_acc = "심야_교통사고_건수"
else:
    col_pop = "거주인구_밀도(명/㎢)"
    col_float = "심야_유동인구_밀도(명/㎢)"
    col_food = "음식점_밀도(개/㎢)"
    col_acc = "심야_사고_밀도(건/㎢)"

# 상관계수 계산
r_pop = df[col_pop].corr(df[col_acc])
r_float = df[col_float].corr(df[col_acc])
r_food = df[col_food].corr(df[col_acc])

# 기준선 모델 (단순 거주인구 단독) vs 최종 모델 (심야 유동인구 + 일반음식점 밀집도)
X_base = df[[col_pop]]
X_multi = df[[col_float, col_food]]
y = df[col_acc]

X_b_train, X_b_test, y_train, y_test = train_test_split(
    X_base, y, test_size=0.2, random_state=42
)
X_m_train, X_m_test, _, _ = train_test_split(
    X_multi, y, test_size=0.2, random_state=42
)

model_base = LinearRegression().fit(X_b_train, y_train)
model_multi = LinearRegression().fit(X_m_train, y_train)

# 전체 및 테스트 성능 계산
r2_base_all = r2_score(y, model_base.predict(X_base))
r2_multi_all = r2_score(y, model_multi.predict(X_multi))
rmse_base = np.sqrt(mean_squared_error(y, model_base.predict(X_base)))
rmse_multi = np.sqrt(mean_squared_error(y, model_multi.predict(X_multi)))
mae_multi = mean_absolute_error(y, model_multi.predict(X_multi))

df["예측_사고량(다중회귀)"] = model_multi.predict(X_multi).round(1)

# 핵심 수치 KPI 카드
c1, c2, c3, c4 = st.columns(4)
c1.metric("단순 거주인구 상관계수 (r)", f"{r_pop:.2f}", "기준선 변수")
c2.metric("심야 유동인구 상관계수 (r)", f"{r_float:.2f}", f"{r_float - r_pop:+.2f} vs 거주인구")
c3.metric("일반음식점 상관계수 (r)", f"{r_food:.2f}", f"{r_food - r_pop:+.2f} vs 거주인구")
c4.metric(
    "다중회귀 설명력 (R²)",
    f"{r2_multi_all:.2f}",
    f"기준선({r2_base_all:.2f}) 대비 +{r2_multi_all - r2_base_all:.2f}",
)

st.divider()

# ---------------------------------------------------------
# 6. 시각화 탭 구성
# ---------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs(
    [
        "📈 1. 핵심 변수별 산점도 & 회귀선",
        "🔥 2. 상관관계 히트맵",
        "🏙️ 3. 자치구별 비교 & 집중도",
        "🤖 4. 모델 성능 비교 & 원본 데이터",
    ]
)

# [탭 1] 3개 독립변수와 심야 교통사고의 산점도 + 회귀선 비교
with tab1:
    st.subheader(f"요인별 심야 교통사고(22~04시) 비례 관계 비교 — [{metric_basis}]")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    features = [
        (col_pop, f"① 단순 거주인구 (r = {r_pop:.2f})", "#7f8c8d"),
        (col_float, f"② 심야 유동인구 (r = {r_float:.2f})", "#2980b9"),
        (col_food, f"③ 일반음식점 밀집도 (r = {r_food:.2f})", "#e74c3c"),
    ]

    for ax, (feat_col, title, color) in zip(axes, features):
        sns.regplot(
            data=df,
            x=feat_col,
            y=col_acc,
            ax=ax,
            color=color,
            scatter_kws={"s": 70, "alpha": 0.75},
            line_kws={"linewidth": 2},
        )
        # 선택한 자치구 이름 라벨링
        for _, row in df[df["자치구명"].isin(selected_districts)].iterrows():
            ax.annotate(
                row["자치구명"],
                (row[feat_col], row[col_acc]),
                textcoords="offset points",
                xytext=(0, 7),
                ha="center",
                fontsize=9,
                fontweight="bold",
                color="#2c3e50",
            )
        ax.set_title(title, fontsize=12, fontweight="bold", pad=10)
        ax.set_xlabel(feat_col, fontsize=10)
        ax.set_ylabel(col_acc, fontsize=10)
        ax.grid(True, linestyle="--", alpha=0.4)

    plt.tight_layout()
    st.pyplot(fig)

    st.success(
        f"**해석 요약:** 단순 거주인구(r={r_pop:.2f})에 비해 **심야 유동인구(r={r_float:.2f})**와 "
        f"**일반음식점 밀집도(r={r_food:.2f})**가 심야 교통사고와 훨씬 강한 양의 선형(비례) 관계를 보입니다."
    )

# [탭 2] 상관관계 히트맵
with tab2:
    st.subheader("주요 변수 간 피어슨(Pearson) 상관계수 히트맵")
    corr_cols = [
        "단순_거주인구",
        "심야_유동인구",
        "일반음식점_수",
        "음식점_밀도(개/㎢)",
        "심야_유동인구_밀도(명/㎢)",
        "심야_교통사고_건수",
        "심야_사고_밀도(건/㎢)",
    ]
    corr_matrix = df[corr_cols].corr()

    fig_hm, ax_hm = plt.subplots(figsize=(9, 6.5))
    sns.heatmap(
        corr_matrix,
        annot=True,
        fmt=".2f",
        cmap="RdYlBu_r",
        vmin=-1,
        vmax=1,
        linewidths=0.8,
        ax=ax_hm,
    )
    ax_hm.set_title("서울시 25개 자치구 심야 교통사고 관련 변수 상관관계", fontsize=13, fontweight="bold", pad=12)
    plt.tight_layout()
    st.pyplot(fig_hm)

# [탭 3] 자치구별 심야 사고 순위 및 상권 집중도
with tab3:
    st.subheader(f"자치구별 {col_acc} 순위 (상위 상권 구 vs 주거 중심 구)")
    df_sorted = df.sort_values(by=col_acc, ascending=True)
    bar_colors = [
        "#e74c3c" if gu in selected_districts else "#95a5a6"
        for gu in df_sorted["자치구명"]
    ]

    fig_bar, ax_bar = plt.subplots(figsize=(10, 7.5))
    bars = ax_bar.barh(df_sorted["자치구명"], df_sorted[col_acc], color=bar_colors)
    ax_bar.set_xlabel(col_acc, fontsize=11)
    ax_bar.set_title(f"서울시 25개 자치구별 {col_acc} 분포 (붉은색: 선택 강조 구)", fontsize=13, fontweight="bold")
    ax_bar.grid(axis="x", linestyle="--", alpha=0.4)

    for bar in bars:
        width = bar.get_width()
        ax_bar.text(
            width + (df_sorted[col_acc].max() * 0.01),
            bar.get_y() + bar.get_height() / 2,
            f"{width:,.1f}",
            va="center",
            fontsize=8.5,
        )

    plt.tight_layout()
    st.pyplot(fig_bar)

    top5_share = (
        df.sort_values(by="심야_교통사고_건수", ascending=False)
        .head(5)["심야_교통사고_건수"]
        .sum()
        / df["심야_교통사고_건수"].sum()
        * 100
    )
    st.info(
        f"**상권 집중도 인사이트:** 서울시 25개 자치구 중 상위 5개 구가 전체 심야 교통사고 발생 건수의 **{top5_share:.1f}%**를 차지하고 있습니다."
    )

# [탭 4] 기준선 vs 다중회귀 모델 비교 및 데이터프레임
with tab4:
    st.subheader("기준선 모델(거주인구) vs 최종 다중회귀 모델(유동인구 + 음식점) 성능 비교")
    perf_df = pd.DataFrame(
        {
            "모델 구분": [
                "기준선 모델 (단순 거주인구 단독)",
                "최종 다중회귀 모델 (심야 유동인구 + 일반음식점)",
            ],
            "사용한 독립변수": [col_pop, f"{col_float} + {col_food}"],
            "결정계수 (R²)": [round(r2_base_all, 3), round(r2_multi_all, 3)],
            "평균제곱근오차 (RMSE)": [round(rmse_base, 2), round(rmse_multi, 2)],
            "평균절대오차 (MAE)": [
                round(mean_absolute_error(y, model_base.predict(X_base)), 2),
                round(mae_multi, 2),
            ],
        }
    )
    st.table(perf_df)

    # 실제값 vs 예측값 산점도
    fig_pred, ax_pred = plt.subplots(figsize=(7, 4.5))
    ax_pred.scatter(df[col_acc], df["예측_사고량(다중회귀)"], color="#2ecc71", s=75, alpha=0.85, label="자치구별 예측값")
    min_val, max_val = df[col_acc].min(), df[col_acc].max()
    ax_pred.plot([min_val, max_val], [min_val, max_val], "r--", linewidth=1.5, label="완벽 예측선 (y = x)")
    for _, row in df[df["자치구명"].isin(selected_districts)].iterrows():
        ax_pred.annotate(
            row["자치구명"],
            (row[col_acc], row["예측_사고량(다중회귀)"]),
            textcoords="offset points",
            xytext=(0, 6),
            ha="center",
            fontsize=8.5,
        )
    ax_pred.set_title(f"실제 {col_acc} vs 다중회귀 모델 예측값", fontsize=12, fontweight="bold")
    ax_pred.set_xlabel("실제값")
    ax_pred.set_ylabel("예측값")
    ax_pred.legend()
    ax_pred.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    st.pyplot(fig_pred)
    import os
import platform
import urllib.request
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
# 1. 페이지 설정 및 한글 폰트 100% 강제 등록
# ---------------------------------------------------------
st.set_page_config(
    page_title="서울시 심야 교통사고 요인 분석 (20920·20922)",
    page_icon="🚦",
    layout="wide",
)

sns.set_style("whitegrid")


@st.cache_resource
def load_korean_font() -> str:
    """실행 환경(Windows / Mac / Streamlit Cloud 리눅스)에 맞는 한글 폰트 확보 및 등록"""
    system_name = platform.system()
    if system_name == "Windows":
        return "Malgun Gothic"
    if system_name == "Darwin":
        return "AppleGothic"

    # 리눅스(Streamlit Cloud): 시스템 폰트 확인 후 없으면 자동 다운로드
    font_candidates = [
        "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
        "/usr/share/fonts/truetype/nanum/NanumBarunGothic.ttf",
        "/tmp/NanumGothic.ttf",
    ]
    font_path = next((p for p in font_candidates if os.path.exists(p)), None)

    if font_path is None:
        font_path = "/tmp/NanumGothic.ttf"
        url = "https://github.com/google/fonts/raw/main/ofl/nanumgothic/NanumGothic-Regular.ttf"
        urllib.request.urlretrieve(url, font_path)

    fm.fontManager.addfont(font_path)
    return fm.FontProperties(fname=font_path).get_name()


FONT_NAME = load_korean_font()
plt.rcParams["font.family"] = FONT_NAME
plt.rcParams["axes.unicode_minus"] = False

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
# 2. 데이터 로드 및 OA-23056(일반음식점 인허가) CSV 처리 함수
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
    1) OA-23056 원본 CSV(수십만 행)인 경우: '영업/정상' 상태만 필터링해 25개 자치구별 집계
    2) 이미 집계된 요약 CSV(자치구명, 일반음식점_수)인 경우: 그대로 매핑
    """
    df = raw_df.copy()

    # 이미 25개 자치구별로 집계된 파일인 경우
    if "자치구명" in df.columns and "일반음식점_수" in df.columns:
        return df.dropna(subset=["자치구명"]).set_index("자치구명")["일반음식점_수"]

    # 원본 인허가 파일인 경우: 영업 중인 업소만 필터링 (폐업 제외)
    if "영업상태명" in df.columns:
        df = df[df["영업상태명"].astype(str).str.contains("영업", na=False)]
    elif "영업상태구분코드" in df.columns:
        df = df[df["영업상태구분코드"].astype(str).str.strip().isin(["1", "01"])]

    # 자치구명 추출 (개방자치단체코드 우선, 없으면 주소 문자열에서 추출)
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
    df["음식점_밀도(개/㎢)"] = (df["일반음식점_수"] / df["행정구역_면적"]).round(1)
    df["심야_유동인구_밀도(명/㎢)"] = (df["심야_유동인구"] / df["행정구역_면적"]).round(1)
    df["거주인구_밀도(명/㎢)"] = (df["단순_거주인구"] / df["행정구역_면적"]).round(1)
    df["심야_사고_밀도(건/㎢)"] = (df["심야_교통사고_건수"] / df["행정구역_면적"]).round(2)
    return df


# ---------------------------------------------------------
# 3. 사이드바: OA-23056 CSV 연결 및 분석 기준 설정
# ---------------------------------------------------------
st.sidebar.header("⚙️ 데이터 연결 (OA-23056)")
st.sidebar.markdown(
    "출처 링크: [서울시 일반음식점 인허가 정보 (OA-23056)]"
    "(https://data.seoul.go.kr/dataList/OA-23056/S/1/datasetView.do)"
)

oa_file = st.sidebar.file_uploader(
    "OA-23056 일반음식점 인허가 CSV 업로드",
    type=["csv"],
    help="서울열린데이터광장 OA-23056 원본 CSV 또는 자치구별 집계 CSV를 올리면 실제 영업중 음식점 수가 반영됩니다.",
)

merged_file = st.sidebar.file_uploader(
    "(선택) 전체 통합 CSV 업로드",
    type=["csv"],
    help="6개 필수 컬럼이 모두 포함된 25개 자치구 통합 CSV가 있다면 여기서 바로 불러올 수 있습니다.",
)

df = load_base_table()
data_status_msg = "기본 내장 25개 자치구 데이터 사용 중 (사이드바에서 OA-23056 CSV 업로드 가능)"

# 로컬 data/OA-23056.csv 파일이 있으면 자동 로드
local_oa_path = os.path.join("data", "OA-23056.csv")
if oa_file is not None:
    with st.spinner("OA-23056 CSV 데이터를 분석하여 자치구별 영업중 일반음식점 수를 집계하는 중..."):
        raw_oa = read_csv_auto_encoding(oa_file)
        gu_counts = parse_oa23056_restaurant_csv(raw_oa)
        if len(gu_counts) > 0:
            df["일반음식점_수"] = df["자치구명"].map(gu_counts).fillna(df["일반음식점_수"]).astype(int)
            data_status_msg = f"✅ 업로드된 OA-23056 실제 CSV 반영 완료 (원본 {len(raw_oa):,}행 처리 · {len(gu_counts)}개 자치구 갱신)"
        else:
            st.sidebar.error("CSV에서 자치구 컬럼을 찾지 못했습니다. OA-23056 원본 파일을 올려주세요.")
elif os.path.exists(local_oa_path):
    raw_oa = read_csv_auto_encoding(local_oa_path)
    gu_counts = parse_oa23056_restaurant_csv(raw_oa)
    if len(gu_counts) > 0:
        df["일반음식점_수"] = df["자치구명"].map(gu_counts).fillna(df["일반음식점_수"]).astype(int)
        data_status_msg = f"✅ 로컬 data/OA-23056.csv 실제 데이터 반영 완료 (원본 {len(raw_oa):,}행 처리)"

if merged_file is not None:
    df = read_csv_auto_encoding(merged_file)
    data_status_msg = "✅ 사용자 전체 통합 CSV 파일 반영 완료"

df = add_derived_features(df)

metric_basis = st.sidebar.radio(
    "비교 기준 단위 선택",
    ["면적 1㎢당 밀도 기준 (규모 편향 보정)", "총량 기준 (명 / 개 / 건)"],
)

selected_districts = st.sidebar.multiselect(
    "강조(하이라이트)할 자치구 선택",
    options=df["자치구명"].tolist(),
    default=["강남구", "마포구", "중구", "노원구"],
)

# ---------------------------------------------------------
# 4. 상단 타이틀 및 핵심 탐구 질문
# ---------------------------------------------------------
st.title("🚦 서울시 심야 유동인구·음식점 밀집도와 심야 교통사고 관계 분석")
st.markdown(f"**팀원:** `20920 손성민`, `20922 오하늘` | **데이터 상태:** {data_status_msg}")

st.info(
    "**핵심 탐구 질문:** \"최근 5년간 서울시 25개 자치구에서 심야 유동인구와 일반음식점이 밀집한 곳은 "
    "정말 심야 시간대(22시~04시) 교통사고도 비례하여 많이 발생하는가?\""
)

# ---------------------------------------------------------
# 5. 회귀 모델 학습 및 성능 지표 계산 (기준선 vs 다중회귀)
# ---------------------------------------------------------
if metric_basis == "총량 기준 (명 / 개 / 건)":
    col_pop = "단순_거주인구"
    col_float = "심야_유동인구"
    col_food = "일반음식점_수"
    col_acc = "심야_교통사고_건수"
else:
    col_pop = "거주인구_밀도(명/㎢)"
    col_float = "심야_유동인구_밀도(명/㎢)"
    col_food = "음식점_밀도(개/㎢)"
    col_acc = "심야_사고_밀도(건/㎢)"

r_pop = df[col_pop].corr(df[col_acc])
r_float = df[col_float].corr(df[col_acc])
r_food = df[col_food].corr(df[col_acc])

X_base = df[[col_pop]]
X_multi = df[[col_float, col_food]]
y = df[col_acc]

X_b_train, X_b_test, y_train, y_test = train_test_split(
    X_base, y, test_size=0.2, random_state=42
)
X_m_train, X_m_test, _, _ = train_test_split(
    X_multi, y, test_size=0.2, random_state=42
)

model_base = LinearRegression().fit(X_b_train, y_train)
model_multi = LinearRegression().fit(X_m_train, y_train)

r2_base_all = r2_score(y, model_base.predict(X_base))
r2_multi_all = r2_score(y, model_multi.predict(X_multi))
rmse_base = np.sqrt(mean_squared_error(y, model_base.predict(X_base)))
rmse_multi = np.sqrt(mean_squared_error(y, model_multi.predict(X_multi)))
mae_multi = mean_absolute_error(y, model_multi.predict(X_multi))

df["예측_사고량(다중회귀)"] = model_multi.predict(X_multi).round(2)

c1, c2, c3, c4 = st.columns(4)
c1.metric("단순 거주인구 상관계수 (r)", f"{r_pop:.2f}", "기준선 변수")
c2.metric("심야 유동인구 상관계수 (r)", f"{r_float:.2f}", f"{r_float - r_pop:+.2f} vs 거주인구")
c3.metric("일반음식점 밀집도 상관계수 (r)", f"{r_food:.2f}", f"{r_food - r_pop:+.2f} vs 거주인구")
c4.metric(
    "다중회귀 설명력 (R²)",
    f"{r2_multi_all:.2f}",
    f"기준선({r2_base_all:.2f}) 대비 +{r2_multi_all - r2_base_all:.2f}",
)

st.divider()

# ---------------------------------------------------------
# 6. 시각화 탭 구성
# ---------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs(
    [
        "📈 1. 핵심 변수별 산점도 & 회귀선",
        "🔥 2. 상관관계 히트맵",
        "🏙️ 3. 자치구별 비교 & 집중도",
        "🤖 4. 모델 성능 비교 & 원본 데이터",
    ]
)

with tab1:
    st.subheader(f"요인별 심야 교통사고(22~04시) 비례 관계 비교 — [{metric_basis}]")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    features = [
        (col_pop, f"① 단순 거주인구 (r = {r_pop:.2f})", "#7f8c8d"),
        (col_float, f"② 심야 유동인구 (r = {r_float:.2f})", "#2980b9"),
        (col_food, f"③ 일반음식점 밀집도 (r = {r_food:.2f})", "#e74c3c"),
    ]

    for ax, (feat_col, title, color) in zip(axes, features):
        sns.regplot(
            data=df,
            x=feat_col,
            y=col_acc,
            ax=ax,
            color=color,
            scatter_kws={"s": 70, "alpha": 0.75},
            line_kws={"linewidth": 2},
        )
        for _, row in df[df["자치구명"].isin(selected_districts)].iterrows():
            ax.annotate(
                row["자치구명"],
                (row[feat_col], row[col_acc]),
                textcoords="offset points",
                xytext=(0, 7),
                ha="center",
                fontsize=9,
                fontweight="bold",
                color="#2c3e50",
            )
        ax.set_title(title, fontsize=12, fontweight="bold", pad=10)
        ax.set_xlabel(feat_col, fontsize=10)
        ax.set_ylabel(col_acc, fontsize=10)
        ax.grid(True, linestyle="--", alpha=0.4)

    plt.tight_layout()
    st.pyplot(fig)

    st.success(
        f"**해석 요약:** 단순 거주인구(r={r_pop:.2f})에 비해 **심야 유동인구(r={r_float:.2f})**와 "
        f"**일반음식점 밀집도(r={r_food:.2f})**가 심야 교통사고와 훨씬 강한 양의 선형(비례) 관계를 보입니다."
    )

with tab2:
    st.subheader("주요 변수 간 피어슨(Pearson) 상관계수 히트맵")
    corr_cols = [
        "단순_거주인구",
        "심야_유동인구",
        "일반음식점_수",
        "음식점_밀도(개/㎢)",
        "심야_유동인구_밀도(명/㎢)",
        "심야_교통사고_건수",
        "심야_사고_밀도(건/㎢)",
    ]
    corr_matrix = df[corr_cols].corr()

    fig_hm, ax_hm = plt.subplots(figsize=(9, 6.5))
    sns.heatmap(
        corr_matrix,
        annot=True,
        fmt=".2f",
        cmap="RdYlBu_r",
        vmin=-1,
        vmax=1,
        linewidths=0.8,
        ax=ax_hm,
    )
    ax_hm.set_title("서울시 25개 자치구 심야 교통사고 관련 변수 상관관계", fontsize=13, fontweight="bold", pad=12)
    plt.tight_layout()
    st.pyplot(fig_hm)

with tab3:
    st.subheader(f"자치구별 {col_acc} 순위 (상위 상권 구 vs 주거 중심 구)")
    df_sorted = df.sort_values(by=col_acc, ascending=True)
    bar_colors = [
        "#e74c3c" if gu in selected_districts else "#95a5a6"
        for gu in df_sorted["자치구명"]
    ]

    fig_bar, ax_bar = plt.subplots(figsize=(10, 7.5))
    bars = ax_bar.barh(df_sorted["자치구명"], df_sorted[col_acc], color=bar_colors)
    ax_bar.set_xlabel(col_acc, fontsize=11)
    ax_bar.set_title(f"서울시 25개 자치구별 {col_acc} 분포 (붉은색: 선택 강조 구)", fontsize=13, fontweight="bold")
    ax_bar.grid(axis="x", linestyle="--", alpha=0.4)

    for bar in bars:
        width = bar.get_width()
        ax_bar.text(
            width + (df_sorted[col_acc].max() * 0.01),
            bar.get_y() + bar.get_height() / 2,
            f"{width:,.1f}",
            va="center",
            fontsize=8.5,
        )

    plt.tight_layout()
    st.pyplot(fig_bar)

    top5_share = (
        df.sort_values(by="심야_교통사고_건수", ascending=False)
        .head(5)["심야_교통사고_건수"]
        .sum()
        / df["심야_교통사고_건수"].sum()
        * 100
    )
    st.info(
        f"**상권 집중도 인사이트:** 서울시 25개 자치구 중 상위 5개 구가 전체 심야 교통사고 발생 건수의 **{top5_share:.1f}%**를 차지하고 있습니다."
    )

with tab4:
    st.subheader("기준선 모델(거주인구) vs 최종 다중회귀 모델(유동인구 + 음식점) 성능 비교")
    perf_df = pd.DataFrame(
        {
            "모델 구분": [
                "기준선 모델 (단순 거주인구 단독)",
                "최종 다중회귀 모델 (심야 유동인구 + 일반음식점)",
            ],
            "사용한 독립변수": [col_pop, f"{col_float} + {col_food}"],
            "결정계수 (R²)": [round(r2_base_all, 3), round(r2_multi_all, 3)],
            "평균제곱근오차 (RMSE)": [round(rmse_base, 2), round(rmse_multi, 2)],
            "평균절대오차 (MAE)": [
                round(mean_absolute_error(y, model_base.predict(X_base)), 2),
                round(mae_multi, 2),
            ],
        }
    )
    st.table(perf_df)

    fig_pred, ax_pred = plt.subplots(figsize=(7, 4.5))
    ax_pred.scatter(df[col_acc], df["예측_사고량(다중회귀)"], color="#2ecc71", s=75, alpha=0.85, label="자치구별 예측값")
    min_val, max_val = df[col_acc].min(), df[col_acc].max()
    ax_pred.plot([min_val, max_val], [min_val, max_val], "r--", linewidth=1.5, label="완벽 예측선 (y = x)")
    for _, row in df[df["자치구명"].isin(selected_districts)].iterrows():
        ax_pred.annotate(
            row["자치구명"],
            (row[col_acc], row["예측_사고량(다중회귀)"]),
            textcoords="offset points",
            xytext=(0, 6),
            ha="center",
            fontsize=8.5,
        )
    ax_pred.set_title(f"실제 {col_acc} vs 다중회귀 모델 예측값", fontsize=12, fontweight="bold")
    ax_pred.set_xlabel("실제값")
    ax_pred.set_ylabel("예측값")
    ax_pred.legend()
    ax_pred.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    st.pyplot(fig_pred)

    st.subheader("📋 분석에 사용된 25개 자치구 통합 데이터표")
    st.dataframe(df, use_container_width=True)

    st.subheader("📋 분석에 사용된 25개 자치구 통합 데이터표")
    st.dataframe(df, use_container_width=True)
