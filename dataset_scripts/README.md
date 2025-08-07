# Dataset Scripts
This subdirectory contains the scripts used during the dataset creation:
- [text_generation_and_screenshot.py](text_generation_and_screenshot.py) – generates synthetic images with random text,
- [automate_tempest.py](automate_tempest.py) – automates TempestSDR and image capturing,
- [remove_borders.py](remove_borders.py) – removes padding from the captured images,
- [divide_dataset.py](divide_dataset.py) – splits the created dataset into three disjoint subsets: *train*, *val*, and *test* in a ratio of 80:10:10.
