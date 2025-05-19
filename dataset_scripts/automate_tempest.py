import re
import pyautogui
import os

pattern = re.compile(r"TSDR.+\.png")


def rename_last_screenshot(directory, index):
    for filename in os.listdir(directory):
        if pattern.match(filename):
            new_name = f"{index}.png"

            src = os.path.join(directory, filename)
            dst = os.path.join(directory, new_name)

            os.rename(src, dst)

            print(f"Renamed {filename} to {new_name}.")


def take_screenshot(directory, index):
    pyautogui.moveTo()  # "Tweaks" dropdown position
    pyautogui.click()
    pyautogui.moveTo()  # "Take snapshot" button position
    pyautogui.click()

    rename_last_screenshot(directory, index)

    pyautogui.moveTo()  # Reference photo location
    pyautogui.press('right')


path_to_directory = ""

for i in range(1, 3001):
    take_screenshot(path_to_directory, i)
