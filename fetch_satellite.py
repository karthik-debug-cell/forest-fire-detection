import pandas as pd

def get_fire_data():
    api_key = "f3fac5e638c9d20758c3ffb1b43c8e8b"  # 🔥 paste your key here

    url = f"https://firms.modaps.eosdis.nasa.gov/api/area/csv/{api_key}/VIIRS_SNPP_NRT/world/1"

    try:
        data = pd.read_csv(url)
        return data
    except Exception as e:
        print("API ERROR:", e)
        return None