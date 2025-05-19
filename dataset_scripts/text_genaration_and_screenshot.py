import string
import time
import random
import os
from selenium import webdriver
from selenium.webdriver.firefox.options import Options
from PIL import Image

# Script that generates a page with random characters and numbers, then takes a screenshot of it with desired
# resolution and saves it.

# Set directory to save to and the number of generated images (iterations).
save_to_dir = ""
iterations = 1500
os.makedirs(os.path.join(save_to_dir, "screenshots"), exist_ok=True)

characters = string.ascii_lowercase + string.ascii_uppercase
special_characters = "!@#$%^&*(){}_+|><?\"[];',./\\"
numbers = "0123456789"

COMMON_FONTS = [
    "Arial, sans-serif", "Verdana, sans-serif", "Times New Roman, serif", "Georgia, serif",
    "Courier New, monospace", "Lucida Console, monospace", "Trebuchet MS, sans-serif", "Tahoma, sans-serif",
    "Impact, sans-serif", "Comic Sans MS, sans-serif"
]

# Main function that performs the generation using Selenium.
def generate_random_html():
    html_template = """
    <html>
    <head>
        <style>
            body {{}}
            .text-block {{}}
        </style>
    </head>
    <body>
    """

    num_sections = random.randint(10, 60)
    for _ in range(num_sections):
        text_or_gap = random.choices(["text", "gap"], weights=(90, 10))[0]
        if text_or_gap == "text":
            text = ""
            for _ in range(random.randint(15, 30)):
                word = "".join(random.choices(characters, k=random.randint(1, 15)))
                special_word = "".join(random.choices(special_characters, k=random.randint(1, 10)))
                number = "".join(random.choices(numbers, k=random.randint(3, 8)))
                text = text + " " + "".join(random.choices([word, number, special_word], weights=(85, 10, 5)))

            font = random.choice(COMMON_FONTS)

            font_size = random.randint(14, 38)
            padding = random.randint(0, 10)
            margin_left = random.randint(0, 100)
            margin_right = random.randint(0, 100)

            style = f"font-family:{font}; font-size:{font_size}px; padding:{padding}px; margin-left:{margin_left}px; margin-right:{margin_right}px"
            html_template += f'<div class="text-block" style="{style}">{text}</div>'
        else:
            html_template += '<div class="text-block" style="visibility:hidden;">&nbsp;</div>'

    html_template += "</body></html>"

    return html_template

# Helper function that creates html file.
def create_html_file(filename):
    html_content = generate_random_html()
    with open(filename, "w", encoding="utf-8") as f:
        f.write(html_content)

# Helper function that defines the desired resolution.
def set_viewport_size(d, width, height):
    window_size = d.execute_script("""
        return [window.outerWidth - window.innerWidth + arguments[0],
          window.outerHeight - window.innerHeight + arguments[1]];
        """, width, height)
    d.set_window_size(*window_size)

# Helper function that takes screenshot of desired resolution.
def capture_screenshot(driver, file_path):
    set_viewport_size(driver, 1024, 768)
    driver.get_screenshot_as_file(file_path)
    with Image.open(file_path) as img:
        img_resized = img.resize((1024, 768))
        img_resized.save(file_path)
    print(f"Saved screenshot: {file_path}")


options = Options()
driver = webdriver.Firefox(options=options)
html_file = "temp_page.html"

for index in range(1, iterations + 1):
    create_html_file(html_file)
    driver.get("file://" + os.path.abspath(html_file))
    screenshot_path = os.path.join(save_to_dir, "screenshots", f"random_text_{index:04d}.png")
    capture_screenshot(driver, screenshot_path)

driver.quit()
os.remove(html_file)
