import schedule
import time
import subprocess
import os
from datetime import datetime
import pytz


def run_all_files(directory):
    files = [f for f in os.listdir(directory) if f.endswith(".py")]
    for file in files:
        file_path = os.path.join(directory, file)
        print(f"Running {file_path}...")
        subprocess.run(["python", file_path])
        print(f"Finished {file_path} at {datetime.now()}")


def is_market_close():
    azerbaijan_time = pytz.timezone("Asia/Baku")
    eastern_time = pytz.timezone("US/Eastern")
    now_in_azerbaijan = datetime.now(azerbaijan_time)
    now_in_eastern = now_in_azerbaijan.astimezone(eastern_time)
    if now_in_eastern.weekday() < 5 and now_in_eastern.hour == 16 and now_in_eastern.minute == 0:
        return True
    return False


def check_and_run(directory):
    if is_market_close():
        run_all_files(directory)


directory_to_run: str = "/home/recabet/Coding/Stock-Predictor/create_STOCK_models"
schedule.every(1).minutes.do(check_and_run, directory=directory_to_run)
print("Monitoring stock market close...")

while True:
    schedule.run_pending()
    time.sleep(1)
