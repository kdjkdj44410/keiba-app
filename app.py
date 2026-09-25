import os
import urllib.request
import warnings

warnings.filterwarnings("ignore")

import io
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import streamlit as st
from bs4 import BeautifulSoup
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler

# ==================================================
# 日本語フォント自動ダウンロード＆設定 (Python 3.12+ 対応)
# ==================================================
FONT_NAME = "NotoSansJP-Regular.ttf"
FONT_URL = (
    f"https://github.com/google/fonts/raw/main/ofl/notosansjp/{FONT_NAME}"
)

if not os.path.exists(FONT_NAME):
    try:
        urllib.request.urlretrieve(FONT_URL, FONT_NAME)
    except Exception:
        pass

if os.path.exists(FONT_NAME):
    fm.fontManager.addfont(FONT_NAME)
    plt.rcParams["font.family"] = "Noto Sans JP"
else:
    plt.rcParams["font.family"] = ["Meiryo", "DejaVu Sans", "sans-serif"]


# ==================================================
# ページ基本設定（スマホ表示最適化）
# ==================================================
st.set_page_config(
    page_title="両対数 妙味分析システム",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.title("🏇 両対数(Log-Log) 妙味分析システム")
st.caption("AI Top3予測率 × 直前リアルタイムオッズ")


# ==================================================
# ロジック関数
# ==================================================
def calc_z(x):
    std = x.std()
    return (
        x - x.mean()
        if std == 0 or pd.isna(std)
        else (x - x.mean()) / std
    )


def normalize_race_probs(df, prob_col, group_col, target_sum=3.0):
    df_norm = df.copy()
    group_sums = df_norm.groupby(group_col)[prob_col].transform("sum")
    res = np.where(
        group_sums > 0,
        df_norm[prob_col] / group_sums * target_sum,
        df_norm[prob_col],
    )
    return np.clip(res, 0.0, 1.0)


def make_best_features(df, group_col="レースID"):
    X_raw = df.copy()
    groups = X_raw[group_col]

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
            X_raw[col] = pd.to_numeric(X_raw[col], errors="coerce")
            med_val = (
                X_raw[col].median() if not X_raw[col].dropna().empty else 0
            )
            X_raw[col] = X_raw[col].fillna(med_val)

    r_eval1 = X_raw["レース評価1"] if "レース評価1" in X_raw.columns else 0
    r_eval2 = X_raw["レース評価2"] if "レース評価2" in X_raw.columns else 0
    X_raw["レース評価_平均"] = (r_eval1 + r_eval2) / 2

    t_cols = [c for c in ["タイム指数1", "タイム指数2"] if c in X_raw.columns]
    X_raw["タイム指数_最大"] = (
        X_raw[t_cols].max(axis=1) if len(t_cols) > 0 else 0
    )

    race_counts = groups.map(groups.value_counts())

    X = pd.DataFrame()
    X["斤量"] = (
        pd.to_numeric(X_raw["斤量"], errors="coerce").fillna(0)
        if "斤量" in X_raw.columns
        else 0
    )
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

    X["能力_生"] = (
        pd.to_numeric(X_raw["能力"], errors="coerce").fillna(0)
        if "能力" in X_raw.columns
        else 0
    )
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


def fetch_netkeiba_odds(race_id):
    """過去DB(db.netkeiba) & 当日出馬表(race.netkeiba) ハイブリッドオッズ取得"""
    clean_id = str(race_id).split(".")[0].strip()
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            " (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        )
    }

    # パターン①: db.netkeiba.com (過去DB)
    db_url = f"https://db.netkeiba.com/race/{clean_id}/"
    try:
        res = requests.get(db_url, headers=headers, timeout=5)
        if res.status_code == 200:
            soup = BeautifulSoup(
                res.content, "html.parser", from_encoding="euc-jp"
            )
            table = soup.find("table", class_="race_table_01")
            if table:
                headers_text = [th.text.strip() for th in table.find_all("th")]
                umaban_idx = next(
                    (i for i, h in enumerate(headers_text) if "馬番" in h), None
                )
                tan_idx = next(
                    (i for i, h in enumerate(headers_text) if "単勝" in h), None
                )
                ninki_idx = next(
                    (i for i, h in enumerate(headers_text) if "人気" in h), None
                )

                odds_dict = {}
                for tr in table.find_all("tr")[1:]:
                    tds = tr.find_all("td")
                    if len(tds) > max(
                        filter(lambda x: x is not None, [umaban_idx, tan_idx])
                    ):
                        try:
                            umaban = int(tds[umaban_idx].text.strip())
                            tan_str = (
                                tds[tan_idx].text.strip().replace(",", "")
                            )
                            odds_val = (
                                float(tan_str)
                                if tan_str and tan_str != "---"
                                else np.nan
                            )
                            ninki_str = (
                                tds[ninki_idx].text.strip()
                                if ninki_idx is not None
                                else ""
                            )
                            ninki_val = (
                                int(ninki_str) if ninki_str.isdigit() else np.nan
                            )

                            if not pd.isna(odds_val):
                                odds_dict[umaban] = {
                                    "単勝": odds_val,
                                    "人気": ninki_val,
                                }
                        except (ValueError, IndexError):
                            continue
                if odds_dict:
                    return pd.DataFrame.from_dict(odds_dict, orient="index")
    except Exception:
        pass

    # パターン②: race.netkeiba.com (当日出馬表)
    shutuba_url = (
        f"https://race.netkeiba.com/race/shutuba.html?race_id={clean_id}"
    )
    try:
        res = requests.get(shutuba_url, headers=headers, timeout=5)
        if res.status_code == 200:
            soup = BeautifulSoup(
                res.content, "html.parser", from_encoding="euc-jp"
            )
            rows = soup.select("tr.HorseList")
            if not rows:
                rows = soup.select("div.RaceTableArea tr")

            odds_dict = {}
            for row in rows:
                umaban_td = row.select_one("td.Umaban, td[class*='Umaban']")
                if not umaban_td or not umaban_td.text.strip().isdigit():
                    continue
                umaban = int(umaban_td.text.strip())

                odds_td = row.select_one(
                    "td.Odds, span[id^='odds-'], td[class*='Odds']"
                )
                odds_val = np.nan
                if odds_td:
                    try:
                        odds_val = float(
                            odds_td.text.strip().replace(",", "")
                        )
                    except ValueError:
                        odds_val = np.nan

                ninki_td = row.select_one(
                    "td.Popular, span[id^='ninki-'], td[class*='Popular']"
                )
                ninki_val = np.nan
                if ninki_td and ninki_td.text.strip().isdigit():
                    ninki_val = int(ninki_td.text.strip())

                if not pd.isna(odds_val):
                    odds_dict[umaban] = {"単勝": odds_val, "人気": ninki_val}

            if odds_dict:
                return pd.DataFrame.from_dict(odds_dict, orient="index")
    except Exception:
        pass

    return None


# ==================================================
# サイドバー：設定＆ファイルアップロード
# ==================================================
st.sidebar.header("⚙️ システム設定")

# パラメータ設定
XX = st.sidebar.number_input(
    "【防波堤】ライン (XX)", value=0.26, step=0.01, format="%.2f"
)
YY = st.sidebar.number_input(
    "【相手候補】ライン (YY)", value=0.15, step=0.01, format="%.2f"
)
Z_THRESHOLD = st.sidebar.number_input(
    "【妙味判定】閾値 (Z)", value=-1.0, step=0.1, format="%.1f"
)
C_PARAM = st.sidebar.number_input(
    "ロジスティック回帰 C値", value=0.03, step=0.01, format="%.2f"
)

st.sidebar.markdown("---")
st.sidebar.subheader("📂 ファイル設定")
train_upload = st.sidebar.file_uploader(
    "学習データ (train_data3.xlsx)", type=["xlsx", "csv"]
)
test_upload = st.sidebar.file_uploader(
    "予想レースデータ (today_race2.xlsx)", type=["xlsx", "csv"]
)

# ファイル読み込み処理 (アップロードなしの場合はローカルのデフォルトファイルを参照)
train_df, test_df = None, None

try:
    if train_upload:
        train_df = (
            pd.read_csv(train_upload)
            if train_upload.name.endswith(".csv")
            else pd.read_excel(train_upload)
        )
    else:
        train_df = pd.read_excel("train_data3.xlsx")
except Exception:
    st.warning(
        "⚠️ 学習データ (train_data3.xlsx) が読み込めません。サイドバーから指定してください。"
    )

try:
    if test_upload:
        test_df = (
            pd.read_csv(test_upload)
            if test_upload.name.endswith(".csv")
            else pd.read_excel(test_upload)
        )
    else:
        test_df = pd.read_excel("today_race2.xlsx")
except Exception:
    st.warning(
        "⚠️ 予想レースデータ (today_race2.xlsx) が読み込めません。サイドバーから指定してください。"
    )


# ==================================================
# メイン処理：モデル学習 ＆ 予測
# ==================================================
if train_df is not None and test_df is not None:
    # 1. モデル学習
    train_proc, X_train_raw = make_best_features(train_df, group_col="レースID")
    y_train = train_proc["Top3"]

    scaler = StandardScaler()
    X_train = pd.DataFrame(
        scaler.fit_transform(X_train_raw), columns=X_train_raw.columns
    )

    model = LogisticRegression(C=C_PARAM, max_iter=5000, random_state=42)
    model.fit(X_train, y_train)

    # 評価スコア計算
    train_pred_prob = model.predict_proba(X_train)[:, 1]
    train_pred_class = model.predict(X_train)
    score_logloss = log_loss(y_train, train_pred_prob)
    score_auc = roc_auc_score(y_train, train_pred_prob)
    score_acc = accuracy_score(y_train, train_pred_class)

    with st.expander("📊 AIモデル評価スコア（学習データ）を確認する", expanded=False):
        c1, c2, c3 = st.columns(3)
        c1.metric("LogLoss", f"{score_logloss:.4f}")
        c2.metric("ROC-AUC", f"{score_auc:.4f}")
        c3.metric("正解率", f"{score_acc:.4f}")

    # 2. テストデータ予測
    test_proc, X_test_raw = make_best_features(test_df, group_col="レースID")
    X_test = pd.DataFrame(
        scaler.transform(X_test_raw), columns=X_test_raw.columns
    )
    raw_prob = model.predict_proba(X_test)[:, 1]

    test_proc["Top3率"] = normalize_race_probs(
        pd.DataFrame({"レースID": test_proc["レースID"], "prob": raw_prob}),
        "prob",
        "レースID",
    )

    # ==================================================
    # 画面：レース選択 ＆ オッズ自動更新
    # ==================================================
    race_ids = test_proc["レースID"].unique()
    selected_race = st.selectbox("🎯 レースを選択してください", race_ids)

    selected_race_str = str(selected_race).split(".")[0].strip()

    # セッション状態管理（手動修正やオッズ更新の保持）
    state_key = f"race_data_{selected_race_str}"
    if state_key not in st.session_state:
        st.session_state[state_key] = test_proc[
            test_proc["レースID"] == selected_race
        ].copy()

    current_race_df = st.session_state[state_key]

    st.markdown("---")
    st.subheader("1. 直前オッズの取得・編集")

    btn_col1, btn_col2 = st.columns([1, 1])

    with btn_col1:
        if st.button(
            "🌐 netkeibaから最新オッズを自動取得",
            type="primary",
            use_container_width=True,
        ):
            with st.spinner("最新オッズを取得中..."):
                live_df = fetch_netkeiba_odds(selected_race_str)
                if live_df is not None and not live_df.empty:
                    df_temp = current_race_df.copy()
                    for umaban, row in live_df.iterrows():
                        mask = df_temp["馬番"] == umaban
                        if not pd.isna(row["単勝"]):
                            df_temp.loc[mask, "単勝"] = row["単勝"]
                        if not pd.isna(row["人気"]):
                            df_temp.loc[mask, "人気"] = row["人気"]
                    st.session_state[state_key] = df_temp
                    st.success("✅ 最新オッズの反映に成功しました！")
                    st.rerun()
                else:
                    st.error("⚠️ オッズの自動取得に失敗しました。")

    with btn_col2:
        if st.button("🔄 初期データに戻す", use_container_width=True):
            st.session_state[state_key] = test_proc[
                test_proc["レースID"] == selected_race
            ].copy()
            st.rerun()

    # データ編集テーブル
    st.caption("※以下の表で単勝オッズ・人気をタップして手動微調整が可能です。")
    edited_df = st.data_editor(
        current_race_df[["馬番", "馬名", "Top3率", "人気", "単勝"]],
        num_rows="fixed",
        use_container_width=True,
        column_config={
            "Top3率": st.column_config.NumberColumn("Top3率", format="%.3f"),
            "単勝": st.column_config.NumberColumn("単勝オッズ", format="%.1f"),
            "人気": st.column_config.NumberColumn("人気", format="%d"),
        },
    )

    # 編集結果をセッションに同期
    for col in ["単勝", "人気"]:
        st.session_state[state_key][col] = edited_df[col]

    st.markdown("---")
    # ==================================================
    # 両対数分析 ＆ グラフ描画 ＆ 買い目表示
    # ==================================================
    st.subheader("2. 両対数乖離分析 ＆ 買い目判定")

    plot_df = edited_df[(edited_df["単勝"] > 0) & (~edited_df["単勝"].isna())].copy()

    if len(plot_df) < 3:
        st.error("⚠️ 単勝オッズが設定されている馬が3頭未満のため、分析を実行できません。")
    else:
        plot_df["単勝"] = pd.to_numeric(plot_df["単勝"])
        plot_df["推定複勝"] = plot_df["単勝"] / (np.log(plot_df["単勝"]) + 1)

        plot_df["log_prob"] = np.log(plot_df["Top3率"].clip(lower=1e-5))
        plot_df["log_est_place"] = np.log(plot_df["推定複勝"])

        coef_poly = np.polyfit(plot_df["log_prob"], plot_df["log_est_place"], 1)
        plot_df["予測log_odds"] = (
            coef_poly[0] * plot_df["log_prob"] + coef_poly[1]
        )

        plot_df["残差"] = plot_df["log_est_place"] - plot_df["予測log_odds"]
        std_val = plot_df["残差"].std()
        plot_df["残差Z"] = (
            0.0
            if (std_val == 0 or pd.isna(std_val))
            else (plot_df["残差"] - plot_df["残差"].mean()) / std_val
        )

        # AI順位の付与
        plot_df = plot_df.sort_values("Top3率", ascending=False).reset_index(
            drop=True
        )
        plot_df["AI順位"] = np.arange(1, len(plot_df) + 1)

        # --- グラフ描画 ---
        fig, ax = plt.subplots(figsize=(10, 6))

        display_x_min = np.log(0.10)
        display_x_max = np.log(min(0.80, plot_df["Top3率"].max() + 0.10))

        ax.axvspan(
            display_x_min,
            np.log(YY),
            color="lightgray",
            alpha=0.30,
            label=f"10% ～ < {YY:.0%}",
        )
        ax.axvspan(
            np.log(YY),
            np.log(XX),
            color="gold",
            alpha=0.22,
            label=f"{YY:.0%} ～ {XX:.0%}",
        )
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
            s=70,
            alpha=0.8,
            color="#1f77b4",
        )

        x_line = np.linspace(display_x_min, display_x_max, 200)
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

        ax.axvline(np.log(YY), linestyle="--", linewidth=1.5, color="gray")
        ax.axvline(np.log(XX), linestyle="--", linewidth=1.5, color="gray")

        ticks = [0.10, 0.12, 0.15, 0.20, 0.26, 0.35, 0.50, 0.70]
        valid_ticks = [
            t for t in ticks if display_x_min <= np.log(t) <= display_x_max
        ]
        ax.set_xticks(np.log(valid_ticks))
        ax.set_xticklabels([f"{t*100:.0f}%" for t in valid_ticks])

        ax.set_xlim(display_x_min, display_x_max)
        ax.set_xlabel("Top3率 (対数スケール / %表示)")
        ax.set_ylabel("log(推定複勝オッズ)")
        ax.set_title(f"レースID: {selected_race_str} 妙味分析")
        ax.grid(alpha=0.3)
        ax.legend()

        st.pyplot(fig)

        # --- 買い目判定一覧 ---
        st.markdown("### 📋 買い目判定結果")

        # 軸候補
        st.markdown(
            f"#### 🎯 軸候補（Top3率 {YY:.0%}以上 ＆ 妙味(残差)上位3頭）"
        )
        axis_df = (
            plot_df[plot_df["Top3率"] >= YY]
            .sort_values("残差", ascending=False)
            .head(3)
        )
        if not axis_df.empty:
            st.dataframe(
                axis_df[
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
                ].style.format({
                    "Top3率": "{:.1%}",
                    "単勝": "{:.1f}",
                    "推定複勝": "{:.1f}",
                    "残差": "{:.3f}",
                    "残差Z": "{:.2f}",
                }),
                use_container_width=True,
            )
        else:
            st.info("該当する軸候補馬はいません。")

        # 防波堤
        st.markdown(f"#### 🛡️ 無条件採用【防波堤】（Top3率 {XX:.0%}以上）")
        always_df = plot_df[plot_df["Top3率"] >= XX].sort_values(
            "Top3率", ascending=False
        )
        if not always_df.empty:
            st.dataframe(
                always_df[
                    [
                        "馬番",
                        "馬名",
                        "Top3率",
                        "人気",
                        "単勝",
                        "推定複勝",
                        "AI順位",
                    ]
                ].style.format({
                    "Top3率": "{:.1%}",
                    "単勝": "{:.1f}",
                    "推定複勝": "{:.1f}",
                }),
                use_container_width=True,
            )
        else:
            st.info("防波堤ラインを超える馬はいません。")

        # 相手候補
        st.markdown(f"#### 🔍 相手候補（{YY:.0%} ≦ Top3率 ＜ {XX:.0%}）")
        candidate = plot_df[
            (plot_df["Top3率"] >= YY) & (plot_df["Top3率"] < XX)
        ].copy()
        if not candidate.empty:
            candidate["判定"] = "×"
            candidate.loc[candidate["残差Z"] > Z_THRESHOLD, "判定"] = "◎"
            candidate = candidate.sort_values("残差", ascending=False)
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
                ].style.format({
                    "Top3率": "{:.1%}",
                    "単勝": "{:.1f}",
                    "推定複勝": "{:.1f}",
                    "残差": "{:.3f}",
                    "残差Z": "{:.2f}",
                }),
                use_container_width=True,
            )
        else:
            st.info("該当する相手候補馬はいません。")
