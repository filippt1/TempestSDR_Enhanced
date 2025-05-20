# Dataset skripty
Tento podadresár obsahuje skripty, ktoré sme použili pri tvorbe datasetu:
- [text_generation_and_screenshot.py](text_generation_and_screenshot.py) - vygeneruje syntetické snímky s náhodným textom,
- [automate_tempest.py](automate_tempest.py) - automatizuje TempestSDR a zachytávanie snímok,
- [remove_borders.py](remove_borders.py) - zo zachytených snímok odstráni „vatu“,
- [divide_dataset.py](divide_dataset.py) - rozdelí vytvorený dataset na tri disjunktné podmnožiny *train*, *val* a *test* v pomere 80:10:10.
