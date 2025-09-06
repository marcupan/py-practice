# Py Practice

A collection of small to mid-size Python practice projects across different difficulty levels (junior, middle, senior) and domains (algorithms, web, data science, ML/DL). Each folder contains a self-contained example with a script and, where needed, sample datasets.

## Structure

- junior_*: Introductory Python tasks
  - junior_1_guess_number/ — simple number guessing game.
  - junior_2_simple_calc/ — command-line calculator.
  - junior_3_generate_password/ — random password generator.
- middle_*: Intermediate tasks
  - middle_4_todo_list/ — CLI to-do list.
  - middle_5_weather_forecast/ — fetch weather data (likely via an API).
  - middle_6_scrape_news/ — basic web scraping example.
- senior_*: Advanced topics
  - senior_7_simple_flask_blog/ — minimal Flask blog with templates.
  - senior_8_telegram_bot/ — simple Telegram bot example script.
  - senior_9_data_analysis/ — example of data analysis workflow.
- data_science_*: ML and data analysis projects by level
  - data_science_junior_1_titanic/ — basic EDA on Titanic dataset.
  - data_science_junior_2_movies/ — TMDB movie dataset exploration.
  - data_science_junior_3_retail/ — retail transactions analysis.
  - data_science_middle_4_titanic_prediction/ — Titanic ML prediction baseline.
  - data_science_middle_5_price_prediction/ — regression/price prediction.
  - data_science_middle_6_image_classification/ — image classification (TensorFlow/Keras). Includes `commands.txt` and `test_tf.py`.
  - data_science_senior_7_spam_classification/ — spam/ham text classification.
  - data_science_senior_8_customer_segmentation/ — clustering/segmentation.
  - data_science_senior_9_stock_price_forecasting/ — time-series forecasting.

## Requirements

Most scripts target Python 3.9+ and rely on common libraries. Consider creating a virtual environment.

A consolidated dependency list is provided in `requirements.txt` at the repository root. You can install everything commonly needed with:

```
python -m venv .venv
. .venv\Scripts\activate
python -m pip install -U pip
pip install -r requirements.txt
```

Typical dependencies you may need (install as needed per script):

- Core: numpy, pandas, matplotlib, seaborn
- ML: scikit-learn
- NLP: scikit-learn, nltk (for some text tasks)
- DL: tensorflow (for image classification)
- Web: flask (for the blog), requests, beautifulsoup4 (for scraping)

Windows note (pandas build error on Python 3.14):
- If you see an error mentioning Meson and `vswhere.exe` while installing pandas (building from source), it likely means a prebuilt wheel isn’t available for your Python version and Windows setup yet.
- Recommended solutions:
  1) Use Python 3.12 in a virtual environment, where wheels are typically available:
     - `py -3.12 -m venv .venv`
     - `. .venv\Scripts\activate`
     - `python -m pip install -U pip`
     - `pip install -r requirements.txt`
  2) Or use Conda (which often has ready-made binaries):
     - `conda create -n pypractice python=3.12`
     - `conda activate pypractice`
     - `pip install -r requirements.txt`
  3) Advanced: Install Microsoft C++ Build Tools and Meson/Ninja to build from source (slower, not recommended for beginners).

Note: Some datasets are included as CSV/XLSX files. For Excel files, `openpyxl` may be required.

## Running Examples

Each folder contains a `script.py` (or `app.py` for Flask) you can run directly.

- Command-line scripts:
  - Windows PowerShell:
    ``
    cd path\to\project\<folder>
    .\.venv\Scripts\activate  # if using venv
    python script.py
    ``

- Flask blog:
  - ``
    cd senior_7_simple_flask_blog
    python app.py
    # Open http://127.0.0.1:5000 in your browser
    ``

- TensorFlow check (optional):
  - ``
    cd data_science_middle_6_image_classification
    python test_tf.py
    ``

Some scripts may expect the current working directory to be the script's folder so they can find the accompanying dataset files.

## Data Notes

- Data files are stored alongside their corresponding scripts (e.g., `train.csv`, `tmdb_5000_movies.csv`, `Mall_Customers.csv`).
- Ensure you do not move datasets without updating paths in the scripts.

## Contributing / Extending

- Keep each example self-contained.
- Use relative paths to reference data within the example folder.
- Prefer small, well-commented scripts that are easy to read and run.

## License

This repository is for educational purposes. If you plan to publish or reuse code/data, verify dataset licensing and attribution requirements.