import tkinter as tk
import time
import threading
import datetime
import os
import csv

# 文件名
data_file = "work_data.csv"

def close_reminder_window(window):
    window.destroy()

def save_data(work_duration, water_intake, situps):
    with open(data_file, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), work_duration, water_intake, situps])

def submit_data(window, work_duration_var, water_intake_var, situps_var):
    work_duration = work_duration_var.get()
    water_intake = water_intake_var.get()
    situps = situps_var.get()

    save_data(work_duration, water_intake, situps)
    close_reminder_window(window)

def show_reminder():
    reminder_window = tk.Toplevel(root)
    reminder_window.title("休息时间")

    message = tk.Label(reminder_window, text="休息一下吧，可以去卫生间、活动一下或运动。")
    message.pack(padx=20, pady=20)

    work_duration_var = tk.StringVar(value="0")
    water_intake_var = tk.StringVar(value="0")
    situps_var = tk.StringVar(value="0")

    work_duration_label = tk.Label(reminder_window, text="有效工作时长(分钟)：")
    work_duration_label.pack()
    work_duration_entry = tk.Entry(reminder_window, textvariable=work_duration_var)
    work_duration_entry.pack()

    water_intake_label = tk.Label(reminder_window, text="喝水量(杯)：")
    water_intake_label.pack()
    water_intake_entry = tk.Entry(reminder_window, textvariable=water_intake_var)
    water_intake_entry.pack()

    situps_label = tk.Label(reminder_window, text="仰卧起坐数量(个)：")
    situps_label.pack()
    situps_entry = tk.Entry(reminder_window, textvariable=situps_var)
    situps_entry.pack()

    submit_button = tk.Button(reminder_window, text="提交数据", command=lambda: submit_data(reminder_window, work_duration_var, water_intake_var, situps_var))
    submit_button.pack(pady=10)

    # 一分钟后自动关闭提醒窗口
    root.after(60000, close_reminder_window, reminder_window)

def schedule_reminder():
    while True:
        current_time = datetime.datetime.now()
        minutes = current_time.minute

        if minutes == 0 or minutes == 30:
            root.after(0, show_reminder)
            time.sleep(60)
        else:
            time.sleep(60)

# 创建文件
if not os.path.exists(data_file):
    with open(data_file, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "work_duration", "water_intake", "situps"])

root = tk.Tk()
root.withdraw()

reminder_thread = threading.Thread(target=schedule_reminder)
reminder_thread.daemon = True
reminder_thread.start()

root.mainloop()

