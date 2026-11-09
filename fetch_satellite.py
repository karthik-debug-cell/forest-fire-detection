import os
import pandas as pd

try:
    import streamlit as st
    cache_data = st.cache_data(ttl=600, show_spinner="Fetching NASA satellite live data...")
except Exception:
    def cache_data(f):
        return f

DEFAULT_API_KEY = "f3fac5e638c9d20758c3ffb1b43c8e8b"

@cache_data
def get_fire_data():
    api_key = os.getenv("NASA_FIRMS_API_KEY")
    if not api_key:
        try:
            import streamlit as st
            api_key = st.secrets.get("NASA_FIRMS_API_KEY", DEFAULT_API_KEY)
        except Exception:
            api_key = DEFAULT_API_KEY

    url = f"https://firms.modaps.eosdis.nasa.gov/api/area/csv/{api_key}/VIIRS_SNPP_NRT/world/1"

    try:
        data = pd.read_csv(url)
        return data
    except Exception as e:
        print("API ERROR:", e)
        return None
