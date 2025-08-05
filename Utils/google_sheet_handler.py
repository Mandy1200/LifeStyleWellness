import gspread
from oauth2client.service_account import ServiceAccountCredentials
import pandas as pd

GOOGLE_SHEET_NAME = "RecoveryGoals"
GOOGLE_CREDS_PATH = "./service_account.json"

def get_sheet(sheet_name=GOOGLE_SHEET_NAME):
    scope = [
        "https://spreadsheets.google.com/feeds",
        "https://www.googleapis.com/auth/drive"
    ]
    creds = ServiceAccountCredentials.from_json_keyfile_name(GOOGLE_CREDS_PATH, scope)
    client = gspread.authorize(creds)
    return client.open(sheet_name).sheet1

def save_goal_to_sheet(user_id: str, goal_data: dict):
    sheet = get_sheet()
    row = [
        user_id,
        goal_data["start"],
        goal_data["target"],
        goal_data["days"],
        goal_data["start_time"],
        goal_data["timestamp"]
    ]
    sheet.append_row(row)

def load_latest_goal_from_sheet(user_id: str):
    sheet = get_sheet()
    records = sheet.get_all_records()

    # 🛠 Normalize keys
    normalized_records = [
        {str(k).strip().lower(): v for k, v in row.items()}
        for row in records
    ]

    df = pd.DataFrame(normalized_records)

    # 🔎 Debug column names
    print("🔎 Sheet columns:", df.columns.tolist())

    if "user_id" not in df.columns:
        raise KeyError(f"Column 'user_id' still not found — columns are: {df.columns.tolist()}")

    user_df = df[df["user_id"] == user_id]
    if user_df.empty:
        return None

    user_df["start_time"] = pd.to_datetime(user_df["start_time"])
    return user_df.sort_values("start_time", ascending=False).iloc[0].to_dict()





def load_all_goals_for_user(user_id: str):
    sheet = get_sheet()
    df = pd.DataFrame(sheet.get_all_records())
    df = df[df["user_id"] == user_id]
    if df.empty:
        return pd.DataFrame()
    df["start_time"] = pd.to_datetime(df["start_time"])
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df.sort_values("start_time", ascending=False)

def delete_latest_goal_for_user(user_id: str):
    sheet = get_sheet()
    df = pd.DataFrame(sheet.get_all_records())
    df["row_number"] = range(2, len(df) + 2)  # +2 because Google Sheets starts at 1 and row 1 is header
    user_df = df[df["user_id"] == user_id]
    if user_df.empty:
        return False
    latest_row = user_df.sort_values("start_time", ascending=False).iloc[0]["row_number"]
    sheet.delete_rows(latest_row)
    return True
