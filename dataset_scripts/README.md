# Dataset skripty
Tento podadresár obsahuje skripty, ktoré sme použili pri tvorbe datasetu:
- [text_generation_and_screenshot.py](dataset_scripts/text_generation_and_screenshot.py) - vygeneruje syntetické snímky s náhodným textom,
- [automate_tempest.py](dataset_scripts/automate_tempest.py) - automatizuje TempestSDR a zachytávanie snímok,
- [remove_borders.py](dataset_scripts/remove_borders.py) - zo zachytených snímok odstráni „vatu“,
- [divide_dataset.py](dataset_scripts/divide_dataset.py) - rozdelí vytvorený dataset na tri disjunktné podmnožiny *train*, *val* a *test* v pomere 80:10:10.
