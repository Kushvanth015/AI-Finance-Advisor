import os
import requests

# ==========================================================
# CREATE FOLDERS
# ==========================================================

os.makedirs("models", exist_ok=True)
os.makedirs("scalers", exist_ok=True)

# ==========================================================
# FILES TO DOWNLOAD
# ==========================================================

FILES = {

    # Models
    "models/isolation_forest_model.pkl":
    "https://huggingface.co/Kushvanth05/AI-Finance-Advisor-Models/resolve/main/models/isolation_forest_model.pkl",

    "models/fraud_xgb_model.pkl":
    "https://huggingface.co/Kushvanth05/AI-Finance-Advisor-Models/resolve/main/models/fraud_xgb_model.pkl",

    "models/goal_prediction_model.pkl":
    "https://huggingface.co/Kushvanth05/AI-Finance-Advisor-Models/resolve/main/models/goal_prediction_model.pkl",

    # Scalers
    "scalers/anomaly_scaler.pkl":
        "https://huggingface.co/Kushvanth05/AI-Finance-Advisor-Models/resolve/main/scalers/anomaly_scaler.pkl",

    "scalers/goal_scaler.pkl":
        "https://huggingface.co/Kushvanth05/AI-Finance-Advisor-Models/resolve/main/scalers/goal_scaler.pkl"

}

# ==========================================================
# DOWNLOAD FUNCTION
# ==========================================================

def download_file(url, save_path):

    print(f"Downloading: {save_path}")

    response = requests.get(
        url,
        stream=True
    )

    response.raise_for_status()

    with open(
        save_path,
        "wb"
    ) as file:

        for chunk in response.iter_content(
            chunk_size=8192
        ):

            file.write(chunk)

    print(
        f"Downloaded: {save_path}"
    )


# ==========================================================
# DOWNLOAD ALL FILES
# ==========================================================

for path, url in FILES.items():

    if os.path.exists(path):

        print(
            f"Already Exists: {path}"
        )

    else:

        download_file(
            url,
            path
        )

print("\nAll Models Downloaded Successfully!")