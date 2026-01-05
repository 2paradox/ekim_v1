#!/usr/bin/env python3
import pandas as pd
import numpy as np
from functools import reduce
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import joblib
import json


RAW_PATH = "raw_data.csv"
MATCH_PATH = "match_info.csv"

STYLE_SCALER_PATH = "style_scaler.joblib"
STYLE_KMEANS_PATH = "style_kmeans.joblib"
STYLE_FEATURE_COLS_PATH = "style_feature_cols.json"
PLAYER_FEATURES_PATH = "player_features_with_cluster.csv"
CLUSTER_LABELS_PATH = "cluster_labels.json"


def safe_rate(numer, denom):
    numer = numer.astype(float)
    denom = denom.astype(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        rate = np.where(denom > 0, numer / denom, 0.0)
    return rate


def build_player_features():
    print("데이터 로딩 중...")
    raw = pd.read_csv(RAW_PATH)
    match = pd.read_csv(MATCH_PATH)

    # 기본 전처리
    raw = raw.dropna(subset=["player_id", "team_id", "type_name"])
    match_small = match[["game_id", "season_id",
                         "home_team_id", "away_team_id"]].copy()

    df = raw.merge(match_small, on="game_id", how="left")
    df["is_home_team"] = (df["team_id"] == df["home_team_id"]).astype(int)
    df["minute"] = df["time_seconds"] / 60.0

    # 경기 수
    print("선수별 경기 수 계산...")
    games_per_player = df.groupby("player_id")["game_id"].nunique().rename("games")

    # 패스 관련
    print("패스 피처 계산...")
    is_pass = df["type_name"].isin(["Pass",
                                    "Pass_Corner",
                                    "Pass_Freekick",
                                    "Cross"])
    pass_df = df[is_pass].copy()
    pass_df["pass_success"] = (pass_df["result_name"] == "Successful").astype(int)
    pass_df["distance"] = np.sqrt(pass_df["dx"] ** 2 + pass_df["dy"] ** 2)
    pass_df["is_forward"] = (pass_df["dx"] > 0).astype(int)
    pass_df["is_long"] = (pass_df["distance"] > 30).astype(int)  # 임의 기준

    pass_agg = pass_df.groupby("player_id").agg(
        pass_attempts=("action_id", "count"),
        pass_successes=("pass_success", "sum"),
        forward_passes=("is_forward", "sum"),
        long_passes=("is_long", "sum"),
    ).reset_index()

    pass_agg["pass_success_rate"] = safe_rate(
        pass_agg["pass_successes"], pass_agg["pass_attempts"]
    )
    pass_agg["forward_pass_ratio"] = safe_rate(
        pass_agg["forward_passes"], pass_agg["pass_attempts"]
    )
    pass_agg["long_pass_ratio"] = safe_rate(
        pass_agg["long_passes"], pass_agg["pass_attempts"]
    )

    # Carry
    print("Carry 피처 계산...")
    carry_df = df[df["type_name"] == "Carry"].copy()
    carry_df["carry_distance"] = np.sqrt(carry_df["dx"] ** 2 +
                                         carry_df["dy"] ** 2)
    carry_agg = carry_df.groupby("player_id").agg(
        carry_count=("action_id", "count"),
        avg_carry_distance=("carry_distance", "mean"),
    ).reset_index()

    # Take On
    print("Take On 피처 계산...")
    takeon_df = df[df["type_name"] == "Take-On"].copy()
    takeon_df["takeon_success"] = (
        takeon_df["result_name"] == "Successful"
    ).astype(int)
    takeon_agg = takeon_df.groupby("player_id").agg(
        takeon_attempts=("action_id", "count"),
        takeon_successes=("takeon_success", "sum"),
    ).reset_index()
    takeon_agg["takeon_success_rate"] = safe_rate(
        takeon_agg["takeon_successes"], takeon_agg["takeon_attempts"]
    )

    # 슈팅
    print("슈팅 피처 계산...")
    is_shot = df["type_name"].isin([
        "Shot", "Shot_Corner", "Shot_Freekick", "Penalty Kick",
    ])
    shot_df = df[is_shot].copy()
    shot_df["shot_on_target"] = shot_df["result_name"].isin(
        ["On Target", "Goal"]
    ).astype(int)

    # 매우 단순한 박스 근처 정의
    shot_df["is_in_box"] = (
        (shot_df["start_x"] > 80)
        & (shot_df["start_y"] > 20)
        & (shot_df["start_y"] < 48)
    ).astype(int)

    shot_agg = shot_df.groupby("player_id").agg(
        shots=("action_id", "count"),
        shots_on_target=("shot_on_target", "sum"),
        shots_in_box=("is_in_box", "sum"),
    ).reset_index()

    shot_agg["shot_on_target_rate"] = safe_rate(
        shot_agg["shots_on_target"], shot_agg["shots"]
    )
    shot_agg["inbox_shot_ratio"] = safe_rate(
        shot_agg["shots_in_box"], shot_agg["shots"]
    )

    # 수비
    print("수비 피처 계산...")
    tackle_df = df[df["type_name"] == "Tackle"].copy()
    tackle_df["tackle_success"] = (
        tackle_df["result_name"] == "Successful"
    ).astype(int)

    tackle_agg = tackle_df.groupby("player_id").agg(
        tackles=("action_id", "count"),
        tackle_successes=("tackle_success", "sum"),
    ).reset_index()
    tackle_agg["tackle_success_rate"] = safe_rate(
        tackle_agg["tackle_successes"], tackle_agg["tackles"]
    )

    intercept_df = df[df["type_name"] == "Interception"].copy()
    intercept_agg = intercept_df.groupby("player_id").agg(
        interceptions=("action_id", "count"),
    ).reset_index()

    block_df = df[df["type_name"] == "Block"].copy()
    block_agg = block_df.groupby("player_id").agg(
        blocks=("action_id", "count"),
    ).reset_index()

    # 하나로 합치기
    print("피처 병합 중...")
    dfs = [
        pass_agg,
        carry_agg,
        takeon_agg,
        shot_agg,
        tackle_agg,
        intercept_agg,
        block_agg,
    ]
    player_features = reduce(
        lambda left, right: pd.merge(left, right, on="player_id", how="outer"),
        dfs,
    )

    player_features = player_features.merge(
        games_per_player.reset_index(), on="player_id", how="left"
    )

    player_features = player_features.fillna(0)

    # 경기당 수치 추가
    print("경기당 수치 계산...")
    count_cols = [
        "pass_attempts",
        "carry_count",
        "takeon_attempts",
        "shots",
        "tackles",
        "interceptions",
        "blocks",
    ]
    games_safe = player_features["games"].replace(0, 1)
    for col in count_cols:
        if col in player_features.columns:
            player_features[col + "_per_game"] = (
                player_features[col] / games_safe
            )

    # 이름, 포지션, 팀 정보 추가
    print("선수 정보 병합...")
    player_names = df[["player_id", "player_name_ko",
                       "main_position", "team_name_ko"]].drop_duplicates(
        subset="player_id"
    )
    player_features = player_features.merge(
        player_names, on="player_id", how="left"
    )

    # 스타일 피처 목록
    feature_cols = [
        "pass_attempts_per_game",
        "pass_success_rate",
        "forward_pass_ratio",
        "long_pass_ratio",
        "carry_count_per_game",
        "avg_carry_distance",
        "takeon_attempts_per_game",
        "takeon_success_rate",
        "shots_per_game",
        "shot_on_target_rate",
        "inbox_shot_ratio",
        "tackles_per_game",
        "tackle_success_rate",
        "interceptions_per_game",
        "blocks_per_game",
    ]
    for col in feature_cols:
        if col not in player_features.columns:
            player_features[col] = 0.0

    print("정규화와 군집화 진행...")
    X = player_features[feature_cols].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    k = 5  # 필요하면 바꿀 수 있음
    kmeans = KMeans(n_clusters=k, random_state=42, n_init="auto")
    clusters = kmeans.fit_predict(X_scaled)

    player_features["cluster"] = clusters

    # 파일 저장
    print("모델 및 피처 저장...")
    joblib.dump(scaler, STYLE_SCALER_PATH)
    joblib.dump(kmeans, STYLE_KMEANS_PATH)
    player_features.to_csv(PLAYER_FEATURES_PATH, index=False)

    with open(STYLE_FEATURE_COLS_PATH, "w") as f:
        json.dump(feature_cols, f, ensure_ascii=False, indent=2)

    # 군집 라벨 초안 생성 예시 (원하면 나중에 직접 수정)
    default_labels = {
        int(i): f"스타일 타입 {i}"
        for i in sorted(player_features["cluster"].unique())
    }
    with open(CLUSTER_LABELS_PATH, "w") as f:
        json.dump(default_labels, f, ensure_ascii=False, indent=2)

    print("완료")
    print("생성 파일 목록")
    print(" ", PLAYER_FEATURES_PATH)
    print(" ", STYLE_SCALER_PATH)
    print(" ", STYLE_KMEANS_PATH)
    print(" ", STYLE_FEATURE_COLS_PATH)
    print(" ", CLUSTER_LABELS_PATH)


if __name__ == "__main__":
    build_player_features()