import os
import pandas as pd
import joblib
import json
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

STYLE_SCALER_PATH = os.path.join(BASE_DIR, "style_scaler.joblib")
STYLE_KMEANS_PATH = os.path.join(BASE_DIR, "style_kmeans.joblib")
STYLE_FEATURE_COLS_PATH = os.path.join(BASE_DIR, "style_feature_cols.json")
PLAYER_FEATURES_PATH = os.path.join(BASE_DIR, "player_features_with_cluster.csv")
CLUSTER_LABELS_PATH = os.path.join(BASE_DIR, "cluster_labels.json")

# 로딩
scaler = joblib.load(STYLE_SCALER_PATH)
kmeans = joblib.load(STYLE_KMEANS_PATH)
player_df = pd.read_csv(PLAYER_FEATURES_PATH)

with open(STYLE_FEATURE_COLS_PATH, "r") as f:
    feature_cols = json.load(f)

with open(CLUSTER_LABELS_PATH, "r") as f:
    cluster_labels = json.load(f)


def get_player_list():
    """선수 목록을 반환한다."""
    cols = ["player_id", "player_name_ko", "team_name_ko", "main_position"]

    df = player_df[cols].drop_duplicates().copy()

    # player_id가 float으로 되어 있을 수 있으니, NaN 제거 후 int로 변환
    df = df[df["player_id"].notna()]
    df["player_id"] = df["player_id"].astype(int)

    return df.sort_values("player_name_ko")

def get_player_profile(player_id):
    """특정 선수의 스타일 프로필과 비슷한 선수 목록을 반환한다."""
    try:
        player_id = int(player_id)
    except Exception:
        return None

    row_df = player_df[player_df["player_id"] == player_id]
    if row_df.empty:
        return None

    row = row_df.iloc[0]

    # 피처 벡터
    x = row[feature_cols].values.reshape(1, -1)
    x_scaled = scaler.transform(x)

    cluster = int(kmeans.predict(x_scaled)[0])
    label = cluster_labels.get(str(cluster),
                               cluster_labels.get(cluster, f"클러스터 {cluster}"))

    # 모든 선수 벡터
    X_all = scaler.transform(player_df[feature_cols].values)
    target_vec = x_scaled[0]
    distances = np.linalg.norm(X_all - target_vec, axis=1)

    tmp_df = player_df.copy()
    tmp_df["distance_tmp"] = distances

    similar = (
        tmp_df[tmp_df["player_id"] != player_id]
        .nsmallest(5, "distance_tmp")
        [["player_id", "player_name_ko",
          "team_name_ko", "main_position"]]
    )

    # 레이더 차트용 값 (리그 평균 대비 비율)
    mean_values = player_df[feature_cols].mean()
    mean_values = mean_values.replace(0, 1)  # 나누기 0 방지
    ratio_series = row[feature_cols] / mean_values
    radar_values = [float(v) for v in ratio_series.to_list()]

    profile = {
        "player_id": int(player_id),
        "name": row.get("player_name_ko", ""),
        "team": row.get("team_name_ko", ""),
        "position": row.get("main_position", ""),
        "cluster": cluster,
        "cluster_label": label,
        "radar_labels": feature_cols,
        "radar_values": radar_values,
        "similar_players": similar.to_dict(orient="records"),
    }
    return profile