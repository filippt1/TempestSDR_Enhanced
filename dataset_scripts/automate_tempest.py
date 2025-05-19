import re
import pyautogui
import os

# Script that automates taking snapshots within TempestSDR.

path_to_directory = ""  # Set path for TempestSDR.

pattern = re.compile(r"TSDR.+\.png")


# Main function that takes screenshot. Define the (x, y) positions of TempestSDR buttons in the script.
# Images should be browsed using software "feh" with argument "-F" (fullscreen).
def take_screenshot(directory, index):
    pyautogui.moveTo()  # "Tweaks" dropdown position
    pyautogui.click()
    pyautogui.moveTo()  # "Take snapshot" button position
    pyautogui.click()

    rename_last_screenshot(directory, index)

    pyautogui.moveTo()  # Reference photo location
    pyautogui.press('right')


# Helper function that renames created screenshot with desired name.
def rename_last_screenshot(directory, index):
    for filename in os.listdir(directory):
        if pattern.match(filename):
            new_name = f"{index}.png"

            src = os.path.join(directory, filename)
            dst = os.path.join(directory, new_name)

            os.rename(src, dst)

            print(f"Renamed {filename} to {new_name}.")


for i in range(1, 3001):
    take_screenshot(path_to_directory, i)
