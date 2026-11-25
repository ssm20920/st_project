import os
import platform
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# 1. 페이지 설정
st.set_page_config(
    page_title="서울시 심야 교통사고 요인 분석 (20920·20922)",
    page_icon="🚦",
    layout="wide",
)

# 2. Seaborn 스타일 먼저 적용 후 한글 폰트 강제 고정
sns.set_style("whitegrid")

def set_korean_font():
    try:
        import koreanize_matplotlib  # noqa: F401
        return
    except Exception:
        pass

    system_name = platform.system()
    if system_name == "Windows":
        plt.rcParams["font.family"] = "Malgun Gothic"
    elif system_name == "Darwin":
        plt.rcParams["font.family"] = "AppleGothic"
    else:
        # 리눅스(Streamlit Cloud 등) 환경용 나눔고딕 탐색
        nanum_paths = [
            "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
            "/usr/share/fonts/truetype/nanum/NanumBarunGothic.ttf",
        ]
        for path in nanum_paths:
            if os.path.exists(path):
                font_prop = fm.FontProperties(fname=path)
                plt.rcParams["font.family"] = font_prop.get_name()
                break
        else:
            plt.rcParams["font.family"] = "NanumGothic"

    plt.rcParams["axes.unicode_minus"] = False

set_korean_font()
