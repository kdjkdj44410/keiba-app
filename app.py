# ==========================================
# Streamlit版：両対数(Log-Log)妙味分析システム
# 現在の予想コードのロジックを維持
# ==========================================

import io
import os
import warnings
from datetime import datetime

warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler


# ==================================================
# 日本語フォント設定（Streamlit Cloud / Linux最適化）
# ※以前のStreamlit版で使用していた設定を採用
# ==================================================
plt.rcParams["font.sans-serif"] = [
    "Noto Sans CJK JP",
    "Noto Sans JP",
    "IPAPGothic",
    "IPAexGothic",
    "TakaoPGothic",
    "DejaVu Sans",
]
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["axes.unicode_minus"] = False


# ==================================================
# ページ基本設定
# ==================================================
st.set_page_config(
    page_title="両対数 妙味分析システム",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.title("🏇 両対数(Log-Log) 妙味分析システム")
st.caption("AI Top3予測率 × 直前リアルタイムオッズ")


# ==================================================
# 設定と基準値
# ==================================================
TRAIN_FILE = "train_data3.xlsx"
TEST_FILE = "today_race2.xlsx"
TARGET = "Top3"
GROUP = "レースID"

XX_DEFAULT = 0.26
YY_DEFAULT = 0.15
Z_THRESHOLD_DEFAULT = -1.0
C_DEFAULT = 0.03


# ==================================================
# 共通関数
# ==================================================
def normalize_race_probs(df, prob_col, group_col, target_sum=3.0):
    df_norm = df.copy()
    group_sums = df_norm.groupby(group_col)[prob_col].transform("sum")
    res = np.where(
        group_sums > 0,
        df_norm[prob_col] / group_sums * target_sum,
        df_norm[prob_col],
    )
    return np.clip(res, 0.0, 1.0)


def calc_z(x):
    std = x.std()
    return x - x.mean() if std == 0 or pd.isna(std) else (x - x.mean()) / std


def make_best_features(df):
    X_raw = df.copy()
    groups = X_raw[GROUP]

    num_cols = [
        "斤量",
        "能力",
        "旧",
        "レース評価1",
        "レース評価2",
        "タイム指数1",
        "タイム指数2",
    ]

    for col in num_cols:
        if col in X_raw.columns:
            X_raw[col] = (
                pd.to_numeric(X_raw[col], errors="coerce").fillna(
                    X_raw[col].median()
                )
                if not X_raw[col].dropna().empty
                else 0
            )

    X_raw["レース評価_平均"] = (
        X_raw.get("レース評価1", 0) + X_raw.get("レース評価2", 0)
    ) / 2

    X_raw["タイム指数_最大"] = X_raw[
        [c for c in ["タイム指数1", "タイム指数2"] if c in X_raw.columns]
    ].max(axis=1)

    race_counts = groups.map(groups.value_counts())

    X = pd.DataFrame()
    X["斤量"] = X_raw.get("斤量", 0)

    X["コース"] = pd.to_numeric(
        X_raw.get("コース", 0), errors="coerce"
    ).fillna(0)

    X["パドック"] = pd.to_numeric(
        X_raw.get("パドック", 0), errors="coerce"
    ).fillna(0)

    X["バイアス"] = pd.to_numeric(
        X_raw.get("バイアス", 0), errors="coerce"
    ).fillna(0)

    X["近走欠損フラグ"] = (
        df["タイム指数1"].isna().astype(int)
        if "タイム指数1" in df.columns
        else 0
    )

    X["能力_生"] = X_raw.get("能力", 0)

    X["旧_パーセンタイル"] = (
        X_raw.groupby(groups)["旧"].rank(ascending=False) / race_counts
        if "旧" in X_raw.columns
        else 0
    )

    X["レース評価_平均_Z"] = X_raw.groupby(groups)[
        "レース評価_平均"
    ].transform(calc_z)

    X["タイム指数_最大_Z"] = X_raw.groupby(groups)[
        "タイム指数_最大"
    ].transform(calc_z)

    return df, X


# ==================================================
# netkeiba オッズ取得
# 現在の予想コードのAPIロジックを維持
# ==================================================
def fetch_netkeiba_odds(race_id: str) -> pd.DataFrame:
    clean_id = str(race_id).split(".")[0].strip()

    url = (
        "https://race.netkeiba.com/api/api_get_jra_odds.html"
        f"?race_id={clean_id}&type=1&action=init"
    )

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        ),
        "Referer": (
            "https://race.netkeiba.com/odds/index.html"
            f"?type=b1&race_id={clean_id}"
        ),
        "X-Requested-With": "XMLHttpRequest",
    }

    try:
        res = requests.get(url, headers=headers, timeout=10)
        res.raise_for_status()
        json_data = res.json()

        raw_odds = json_data.get("data", {}).get("odds", {}).get("1", {})

        if not raw_odds:
            return pd.DataFrame()

        formatted_list = []

        for umaban_str, val_list in raw_odds.items():
            umaban = int(umaban_str)

            odds_val = (
                float(val_list[0])
                if (val_list[0] != "---.-" and val_list[0] != "---")
                else np.nan
            )

            ninki_val = (
                int(val_list[2])
                if len(val_list) > 2 and str(val_list[2]).isdigit()
                else np.nan
            )

            formatted_list.append(
                {
                    "馬番": umaban,
                    "単勝": odds_val,
                    "人気": ninki_val,
                }
            )

        return pd.DataFrame(formatted_list)

    except Exception as e:
        st.warning(f"オッズAPI通信エラー（RaceID: {clean_id}）: {e}")
        return pd.DataFrame()


# ==================================================
# サイドバー
# ==================================================
st.sidebar.header("⚙️ システム設定")

XX = st.sidebar.number_input(
    "【防波堤】ライン (XX)",
    value=XX_DEFAULT,
    step=0.01,
    format="%.2f",
)

YY = st.sidebar.number_input(
    "【相手候補】ライン (YY)",
    value=YY_DEFAULT,
    step=0.01,
    format="%.2f",
)

Z_THRESHOLD = st.sidebar.number_input(
    "【妙味判定】閾値 (Z)",
    value=Z_THRESHOLD_DEFAULT,
    step=0.1,
    format="%.1f",
)

C_PARAM = st.sidebar.number_input(
    "ロジスティック回帰 C値",
    value=C_DEFAULT,
    step=0.01,
    format="%.2f",
)

USE_LIVE_ODDS = st.sidebar.checkbox(
    "netkeibaから直前オッズを取得",
    value=True,
)

st.sidebar.markdown("---")
st.sidebar.subheader("📂 ファイル設定")

train_upload = st.sidebar.file_uploader(
    "学習データ (train_data3.xlsx)",
    type=["xlsx", "csv"],
)

test_upload = st.sidebar.file_uploader(
    "予想レースデータ (today_race2.xlsx)",
    type=["xlsx", "csv"],
)


# ==================================================
# ファイル読み込み
# ==================================================
def read_uploaded_file(uploaded_file):
    if uploaded_file is None:
        return None

    if uploaded_file.name.lower().endswith(".csv"):
        return pd.read_csv(uploaded_file)

    return pd.read_excel(uploaded_file)


train_df = None
test_df = None

try:
    if train_upload is not None:
        train_df = read_uploaded_file(train_upload)
    elif os.path.exists(TRAIN_FILE):
        train_df = pd.read_excel(TRAIN_FILE)
except Exception as e:
    st.error(f"学習データの読み込みに失敗しました: {e}")

try:
    if test_upload is not None:
        test_df = read_uploaded_file(test_upload)
    elif os.path.exists(TEST_FILE):
        test_df = pd.read_excel(TEST_FILE)
except Exception as e:
    st.error(f"予想レースデータの読み込みに失敗しました: {e}")


if train_df is None or test_df is None:
    st.info(
        "左側のサイドバーから「train_data3.xlsx」と "
        "「today_race2.xlsx」をアップロードしてください。"
    )
    st.stop()


# ==================================================
# 必須列チェック
# ==================================================
required_train = [
    GROUP,
    TARGET,
    "斤量",
    "能力",
    "旧",
    "レース評価1",
    "レース評価2",
    "タイム指数1",
    "タイム指数2",
]

required_test = [
    GROUP,
    "斤量",
    "能力",
    "旧",
    "レース評価1",
    "レース評価2",
    "タイム指数1",
    "タイム指数2",
    "馬番",
    "馬名",
]

missing_train = [c for c in required_train if c not in train_df.columns]
missing_test = [c for c in required_test if c not in test_df.columns]

if missing_train:
    st.error(
        "学習データに必要な列がありません："
        + ", ".join(missing_train)
    )
    st.stop()

if missing_test:
    st.error(
        "予想レースデータに必要な列がありません："
        + ", ".join(missing_test)
    )
    st.stop()


# ==================================================
# メイン処理：モデル学習
# ==================================================
with st.spinner("AIモデルを学習しています..."):
    train_proc, X_train_raw = make_best_features(train_df)

    y_train = train_proc[TARGET]

    scaler = StandardScaler()

    X_train = pd.DataFrame(
        scaler.fit_transform(X_train_raw),
        columns=X_train_raw.columns,
    )

    model = LogisticRegression(
        C=C_PARAM,
        max_iter=5000,
        random_state=42,
    )

    model.fit(X_train, y_train)

    train_pred_prob = model.predict_proba(X_train)[:, 1]
    train_pred_class = model.predict(X_train)

    score_logloss = log_loss(y_train, train_pred_prob)
    score_auc = roc_auc_score(y_train, train_pred_prob)
    score_acc = accuracy_score(y_train, train_pred_class)


# ==================================================
# AIモデル評価
# ==================================================
with st.expander(
    "📊 AIモデル評価スコア（現在の学習データ）",
    expanded=False,
):
    c1, c2, c3 = st.columns(3)
    c1.metric("LogLoss", f"{score_logloss:.4f}")
    c2.metric("ROC-AUC", f"{score_auc:.4f}")
    c3.metric("正解率", f"{score_acc:.4f}")


# ==================================================
# テストデータ予測
# ==================================================
with st.spinner("レース予測を計算しています..."):
    test_proc, X_test_raw = make_best_features(test_df)

    X_test = pd.DataFrame(
        scaler.transform(X_test_raw),
        columns=X_test_raw.columns,
    )

    raw_prob_test = model.predict_proba(X_test)[:, 1]

    test_proc["Top3率"] = normalize_race_probs(
        pd.DataFrame(
            {
                GROUP: test_proc[GROUP],
                "prob": raw_prob_test,
            }
        ),
        "prob",
        GROUP,
    )


# ==================================================
# レース選択
# ==================================================
race_ids = test_proc[GROUP].unique()

if len(race_ids) == 0:
    st.error("予想対象のレースが見つかりません。")
    st.stop()

selected_race = st.selectbox(
    "🎯 レースを選択してください",
    race_ids,
)

selected_race_str = str(selected_race).split(".")[0].strip()


# ==================================================
# レースごとのセッション状態
# ==================================================
state_key = f"race_data_{selected_race_str}"
version_key = f"version_{selected_race_str}"
last_update_key = f"last_updated_{selected_race_str}"

if state_key not in st.session_state:
    st.session_state[state_key] = test_proc[
        test_proc[GROUP] == selected_race
    ].copy()

if version_key not in st.session_state:
    st.session_state[version_key] = 0


# ==================================================
# オッズ取得・編集
# ==================================================
st.markdown("---")
st.subheader("1. 直前オッズの取得・編集")

if last_update_key in st.session_state:
    st.caption(
        f"🕒 最終オッズ更新時刻: "
        f"**{st.session_state[last_update_key]}**"
    )

btn_col1, btn_col2 = st.columns(2)

with btn_col1:
    if st.button(
        "🌐 netkeibaから最新オッズを自動取得",
        type="primary",
        use_container_width=True,
    ):
        with st.spinner("最新オッズを取得中..."):
            live_df = fetch_netkeiba_odds(selected_race_str)

        if live_df is not None and not live_df.empty:
            df_temp = st.session_state[state_key].copy()

            for _, row in live_df.iterrows():
                umaban = int(row["馬番"])
                mask = df_temp["馬番"] == umaban

                if "単勝" in row and not pd.isna(row["単勝"]):
                    df_temp.loc[mask, "単勝"] = row["単勝"]

                if "人気" in row and not pd.isna(row["人気"]):
                    df_temp.loc[mask, "人気"] = row["人気"]

            st.session_state[state_key] = df_temp
            st.session_state[version_key] += 1
            st.session_state[last_update_key] = datetime.now().strftime(
                "%H:%M:%S"
            )

            st.success("✅ 最新オッズの反映に成功しました！")
            st.rerun()

        else:
            st.error(
                "⚠️ オッズの自動取得に失敗しました。"
                "レースID（netkeibaの12桁ID）を確認してください。"
            )

with btn_col2:
    if st.button(
        "🔄 初期データに戻す",
        use_container_width=True,
    ):
        st.session_state[state_key] = test_proc[
            test_proc[GROUP] == selected_race
        ].copy()

        st.session_state[version_key] += 1

        if last_update_key in st.session_state:
            del st.session_state[last_update_key]

        st.rerun()


st.caption(
    "※以下の表で単勝オッズ・人気をタップして手動微調整できます。"
)

editor_key = (
    f"editor_{selected_race_str}_"
    f"{st.session_state[version_key]}"
)

edited_df = st.data_editor(
    st.session_state[state_key][
        ["馬番", "馬名", "Top3率", "人気", "単勝"]
    ],
    key=editor_key,
    num_rows="fixed",
    use_container_width=True,
    column_config={
        "Top3率": st.column_config.NumberColumn(
            "Top3率",
            format="%.3f",
        ),
        "単勝": st.column_config.NumberColumn(
            "単勝オッズ",
            format="%.1f",
        ),
        "人気": st.column_config.NumberColumn(
            "人気",
            format="%d",
        ),
    },
)

# 編集結果をセッション状態へ反映
for col in ["単勝", "人気"]:
    st.session_state[state_key][col] = edited_df[col].values


# ==================================================
# 両対数分析
# ==================================================
st.markdown("---")
st.subheader("2. 両対数乖離分析 ＆ 買い目判定")

plot_df = edited_df[
    (edited_df["単勝"] > 0)
    & (~edited_df["単勝"].isna())
].copy()

if len(plot_df) < 3:
    st.error(
        "⚠️ 単勝オッズが設定されている馬が3頭未満のため、"
        "分析を実行できません。"
    )
    st.stop()


plot_df["単勝"] = pd.to_numeric(
    plot_df["単勝"],
    errors="coerce",
)

plot_df = plot_df[
    (plot_df["単勝"] > 0)
    & (~plot_df["単勝"].isna())
].copy()

if len(plot_df) < 3:
    st.error(
        "⚠️ 有効な単勝オッズが3頭未満のため、"
        "分析を実行できません。"
    )
    st.stop()


# --------------------------------------------------
# 現在の予想コードと同じ両対数計算
# --------------------------------------------------
plot_df["推定複勝"] = (
    plot_df["単勝"]
    / (np.log(plot_df["単勝"]) + 1)
)

plot_df["log_prob"] = np.log(
    plot_df["Top3率"].clip(lower=1e-5)
)

plot_df["log_est_place"] = np.log(
    plot_df["推定複勝"]
)

coef_poly = np.polyfit(
    plot_df["log_prob"],
    plot_df["log_est_place"],
    1,
)

plot_df["予測log_odds"] = (
    coef_poly[0] * plot_df["log_prob"]
    + coef_poly[1]
)

plot_df["残差"] = (
    plot_df["log_est_place"]
    - plot_df["予測log_odds"]
)

std_val = plot_df["残差"].std()

plot_df["残差Z"] = (
    0.0
    if (std_val == 0 or pd.isna(std_val))
    else (
        plot_df["残差"]
        - plot_df["残差"].mean()
    ) / std_val
)


# ==================================================
# AI順位
# ==================================================
plot_df = plot_df.sort_values(
    "Top3率",
    ascending=False,
).reset_index(drop=True)

plot_df["AI順位"] = np.arange(
    1,
    len(plot_df) + 1,
)


# ==================================================
# グラフ
# ==================================================
fig, ax = plt.subplots(figsize=(10, 6))

display_x_min = np.log(0.10)
display_x_max = np.log(
    min(
        0.80,
        plot_df["Top3率"].max() + 0.10,
    )
)

# 10% ～ YY未満
ax.axvspan(
    display_x_min,
    np.log(YY),
    color="lightgray",
    alpha=0.30,
    label=f"10% ～ < {YY:.0%}",
)

# YY ～ XX未満
ax.axvspan(
    np.log(YY),
    np.log(XX),
    color="gold",
    alpha=0.22,
    label=f"{YY:.0%} ～ {XX:.0%}",
)

# XX以上
ax.axvspan(
    np.log(XX),
    display_x_max,
    color="lightgreen",
    alpha=0.22,
    label=f"≥ {XX:.0%}",
)

ax.scatter(
    plot_df["log_prob"],
    plot_df["log_est_place"],
    s=60,
    alpha=0.8,
)

x_line = np.linspace(
    display_x_min,
    display_x_max,
    200,
)

ax.plot(
    x_line,
    coef_poly[0] * x_line + coef_poly[1],
    color="red",
    linewidth=2,
    label="Regression (Est. Place)",
)

for _, r in plot_df.iterrows():
    if r["log_prob"] >= display_x_min:
        ax.text(
            r["log_prob"],
            r["log_est_place"],
            f' {int(r["馬番"])} {r["馬名"]}',
            fontsize=10,
            va="center",
        )

ax.axvline(
    np.log(YY),
    linestyle="--",
    linewidth=1.5,
    color="gray",
)

ax.axvline(
    np.log(XX),
    linestyle="--",
    linewidth=1.5,
    color="gray",
)

ticks = [
    0.10,
    0.12,
    0.15,
    0.20,
    0.26,
    0.35,
    0.50,
    0.70,
]

valid_ticks = [
    t
    for t in ticks
    if display_x_min <= np.log(t) <= display_x_max
]

ax.set_xticks(np.log(valid_ticks))
ax.set_xticklabels(
    [f"{t * 100:.0f}%" for t in valid_ticks]
)

ax.set_xlim(
    display_x_min,
    display_x_max,
)

ax.set_xlabel(
    "Top3率 (対数スケール / %表示)"
)

ax.set_ylabel(
    "log(推定複勝オッズ)"
)

ax.set_title(
    f"レースID: {selected_race_str} "
    "両対数乖離分析【自然対数モデル補正】"
)

ax.grid(alpha=0.3)
ax.legend()

fig.tight_layout()

st.pyplot(fig, clear_figure=True)


# ==================================================
# 買い目判定
# ==================================================
st.markdown("### 📋 買い目判定結果")


# --------------------------------------------------
# 軸候補
# --------------------------------------------------
st.markdown(
    f"#### 🎯 軸候補"
    f"（Top3率 {YY:.0%}以上 ＆ 妙味(残差)上位3頭）"
)

axis_df = (
    plot_df[plot_df["Top3率"] >= YY]
    .sort_values("残差", ascending=False)
    [
        [
            "馬番",
            "馬名",
            "Top3率",
            "人気",
            "単勝",
            "推定複勝",
            "AI順位",
            "残差",
            "残差Z",
        ]
    ]
    .head(3)
)

if not axis_df.empty:
    st.dataframe(
        axis_df.style.format(
            {
                "Top3率": "{:.1%}",
                "単勝": "{:.1f}",
                "推定複勝": "{:.1f}",
                "残差": "{:.3f}",
                "残差Z": "{:.2f}",
            }
        ),
        use_container_width=True,
    )
else:
    st.info("該当する軸候補馬はいません。")


# --------------------------------------------------
# 防波堤
# --------------------------------------------------
st.markdown(
    f"#### 🛡️ 無条件採用【防波堤】"
    f"（Top3率 {XX:.0%}以上）"
)

always_df = (
    plot_df[plot_df["Top3率"] >= XX]
    .sort_values("Top3率", ascending=False)
    [
        [
            "馬番",
            "馬名",
            "Top3率",
            "人気",
            "単勝",
            "推定複勝",
            "AI順位",
        ]
    ]
)

if not always_df.empty:
    st.dataframe(
        always_df.style.format(
            {
                "Top3率": "{:.1%}",
                "単勝": "{:.1f}",
                "推定複勝": "{:.1f}",
            }
        ),
        use_container_width=True,
    )
else:
    st.info("防波堤ラインを超える馬はいません。")


# --------------------------------------------------
# 相手候補
# --------------------------------------------------
st.markdown(
    f"#### 🔍 相手候補"
    f"（{YY:.0%} ≦ Top3率 ＜ {XX:.0%}）"
)

candidate = plot_df[
    (plot_df["Top3率"] >= YY)
    & (plot_df["Top3率"] < XX)
].copy()

if not candidate.empty:
    candidate["判定"] = "×"

    candidate.loc[
        candidate["残差Z"] > Z_THRESHOLD,
        "判定",
    ] = "◎"

    candidate = candidate.sort_values(
        "残差",
        ascending=False,
    )

    st.dataframe(
        candidate[
            [
                "判定",
                "馬番",
                "馬名",
                "Top3率",
                "人気",
                "単勝",
                "推定複勝",
                "AI順位",
                "残差",
                "残差Z",
            ]
        ].style.format(
            {
                "Top3率": "{:.1%}",
                "単勝": "{:.1f}",
                "推定複勝": "{:.1f}",
                "残差": "{:.3f}",
                "残差Z": "{:.2f}",
            }
        ),
        use_container_width=True,
    )
else:
    st.info("該当する相手候補馬はいません。")
