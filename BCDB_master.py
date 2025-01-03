import os
import re
import streamlit as st
from google.oauth2 import service_account
from google.cloud import storage
import base64
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import pandas as pd
from collections import defaultdict
from io import StringIO

credentials = service_account.Credentials.from_service_account_info(
    st.secrets["gcp_service_account"]
)
client = storage.Client(credentials=credentials)

st.set_page_config(page_title="Blank Check Database", page_icon=":mag_right:")

if 'reset' not in st.session_state:
    st.session_state['reset'] = False
if 'button_clicked' not in st.session_state:
    st.session_state['button_clicked'] = False

def download_csv(data_frame, file_name):
    csv_buffer = StringIO()
    data_frame.to_csv(csv_buffer, sep=";", index=False)
    b64 = base64.b64encode(csv_buffer.getvalue().encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{file_name}" target="_blank">Download</a>'
    return href

term_styles = {
    'comedy points': {'color': '#D4AF37', 'emoji': ''},
    'night eggs': {'color': '#AE88E1', 'emoji': '🥚'},
    'river of ham': {'color': '#AE88E1', 'emoji': '🐷'},
    'david dog': {'color': '#AE88E1', 'emoji': '🐶'},
    'burger report': {'color': '#AE88E1', 'emoji': '🍔'}
}

def highlight_term(text, term):
    def replace(match):
        matched_text = match.group(0)
        style = term_styles.get(term.lower(), {'color': '#AE88E1', 'emoji': ''})
        return f'<span style="font-weight: bold; color: {style["color"]};">{matched_text}{style["emoji"]}</span>'
    return re.sub(rf'\b{re.escape(term)}\b', replace, text, flags=re.IGNORECASE)

def process_blob(blob, search_term, folder_name):
    matching_rows = defaultdict(list)

    if blob.name.endswith('.csv') and (folder_name == "all" or blob.name.startswith(folder_name)):
        data = pd.read_csv(blob.open("rt"), delimiter=";", engine="python")

        if not data.empty:
            url1, url2, url3 = data.iloc[0, 0], data.iloc[1, 0], data.iloc[2, 0]
            youtube_url = url1 if isinstance(url1, str) and 'youtube' in url1.lower() else url2 if isinstance(url2, str) and 'youtube' in url2.lower() else url3 if isinstance(url3, str) and 'youtube' in url3.lower() else None
            patreon_url = url1 if isinstance(url1, str) and 'patreon' in url1.lower() else url2 if isinstance(url2, str) and 'patreon' in url2.lower() else url3 if isinstance(url3, str) and 'patreon' in url3.lower() else None

        matches = data[data.iloc[:, 2].apply(lambda x: bool(re.search(rf'\b{search_term}\b', str(x), re.IGNORECASE)))].values.tolist()

        for match in matches:
            col4 = match[3] if len(match) > 3 else None
            col5 = match[4] if len(match) > 4 else None
            matching_rows[blob.name].append((youtube_url, patreon_url, match[0], match[2], col4, col5))

    return matching_rows

def get_csv_data(bucket, search_term, folder_name):
    matching_rows = defaultdict(list)
    blobs = list(bucket.list_blobs())

    with ThreadPoolExecutor() as executor:
        results = executor.map(process_blob, blobs, [search_term]*len(blobs), [folder_name]*len(blobs))

        for result in results:
            matching_rows.update(result)

    return matching_rows

def get_csv_dataframe(bucket, file_name):
    for blob in bucket.list_blobs():
        if blob.name == file_name:
            data = pd.read_csv(blob.open("rt"), delimiter=";", engine="python")
            return data
    return None

logo_url = f"https://storage.googleapis.com/bcdb_images/BCDb_logo_apr10.png"
st.markdown(f'<div style="text-align: center;"><img src="{logo_url}" width="300"></div>', unsafe_allow_html=True)
st.markdown("<h1 style='text-align: center;'><span style='color: #AE88E1;'>Blank Check </span><span style='color: #8E3497;'>Database</span></h1>", unsafe_allow_html=True)

bucket_name_mapping = {
    'Main Feed': 'bcdb_episodes',
    'Patreon': 'bcdb_patreon',
}
display_names = list(bucket_name_mapping.keys())
selected_display_name = st.selectbox("Select a feed:", display_names)
bucket_name = bucket_name_mapping[selected_display_name]

bucket = client.get_bucket(bucket_name)
blobs = bucket.list_blobs()
unique_folder_names = sorted(set(os.path.dirname(blob.name) for blob in blobs if blob.name.endswith(('.csv'))))
folder_names = {"All Miniseries": "all", **{os.path.basename(folder)[4:].replace('_', ' '): folder for folder in unique_folder_names if folder}}
folder_name = st.selectbox("Select a Miniseries:", list(folder_names.keys()))

search_term = st.text_input("Enter search term:", value="", key="search_box", max_chars=None, type="default", help=None, placeholder="e.g. Star Wars", on_change=lambda: st.session_state.update({'button_clicked': True}))
highlight_color = "#E392EA"
button_clicked = st.button("Search") or st.session_state.get('button_clicked', False)

def time_to_seconds(time_str):
    h, m, s = map(int, time_str.split(':'))
    return h * 3600 + m * 60 + s

youtube_icon_url = "https://storage.googleapis.com/bcdb_images/Youtube_logo.png"
patreon_icon_url = "https://storage.googleapis.com/bcdb_images/patreon_logo.png"

def get_image_url(bucket, file_name):
    image_url = None

    file_name_without_ext, _ = os.path.splitext(file_name)
    file_name_only = os.path.basename(file_name_without_ext)
    folder_name = os.path.basename(os.path.dirname(file_name))

    prefix = f"episode_art/{file_name_only}"
    for blob in bucket.list_blobs(prefix=prefix):
        if blob.name.endswith(('.jpg', '.jpeg', '.png', '.gif')):
            image_url = f"https://storage.googleapis.com/{bucket.name}/{blob.name}"
            return image_url

    prefix = f"episode_art/{folder_name}"
    for blob in bucket.list_blobs(prefix=prefix):
        if blob.name.endswith(('.jpg', '.jpeg', '.png', '.gif')):
            image_url = f"https://storage.googleapis.com/{bucket.name}/{blob.name}"
            return image_url

    return image_url

def extract_number(file_name):
    match = re.search(r'\d+', file_name)
    return int(match.group()) if match else float('inf')

image_bucket = client.get_bucket('bcdb_images')

if st.button('Reset', type="primary"):
    st.session_state['button_clicked'] = False
    st.experimental_rerun()

if button_clicked:
    if not search_term.strip():
        st.write("Please enter a search term")
    else:

        folder_path = folder_names[folder_name]
        results = get_csv_data(bucket, search_term, folder_path)

        if results:
            total_results = sum(len(file_results) for file_results in results.values())
            st.write(f"({total_results}) {'result' if total_results == 1 else 'results'} found")

            sorted_results = sorted(results.items(), key=lambda x: (x[0].rsplit('/', 1)[-1], extract_number(x[0])))

            for index, (file_name, file_results) in enumerate(sorted_results):
                patreon_icon_displayed = False
                file_name_without_ext, _ = os.path.splitext(file_name)
                file_folder = os.path.dirname(file_name)
                file_name_only = os.path.basename(file_name_without_ext)[4:].replace('_', ' ')
                result_word = "result" if len(file_results) == 1 else "results"

                patreon_url = next((result[1] for result in file_results if result[1]), None)
                patreon_icon = f'<a href="{patreon_url}" target="_blank"><img src="{patreon_icon_url}" width="20"></a>' if patreon_url else ""

                if folder_name == "All Miniseries":
                    image_url = get_image_url(image_bucket, file_name)
                    if image_url:
                        st.markdown(f'<div style="text-align: left;"><img src="{image_url}" width="200"></div>', unsafe_allow_html=True)

                data_frame = get_csv_dataframe(bucket, file_name)
                if data_frame is not None:
                    download_button = download_csv(data_frame, file_name)
                    public_url = f'https://storage.googleapis.com/{bucket_name}/{file_name}'
                    view_button = f'<a href="{public_url}" target="_blank">View</a>'

                    st.markdown(f"<span style='font-size: 25px; color: #AE88E1; font-weight: bold;'>{patreon_icon} {file_name_only} <span style='font-size: 15pt; color: #8E3497;'>({len(file_results)} {result_word})</span>:</span><br>Transcript:  {view_button} | {download_button}", unsafe_allow_html=True)

                for youtube_url, patreon_url, col1, col3, col4, col5 in file_results:
                    if pd.notna(col5):
                        st.markdown(f'<div style="margin-bottom: 2px;"><span style="color: #8E3497; font-weight: bold;">[{col1}]:</span> <span style="color: red;">{highlight_term(col3, search_term)}</span></div>', unsafe_allow_html=True)
                    else:
                        st.markdown(f'<div style="margin-bottom: 2px;"><span style="color: #8E3497; font-weight: bold;">[{col1}]:</span> {highlight_term(col3, search_term)}</div>', unsafe_allow_html=True)

                    time_in_seconds = time_to_seconds(col1)
                    youtube_icon = f'<a href="{youtube_url}&t={time_in_seconds}" target="_blank"><img src="{youtube_icon_url}" width="30"></a>' if youtube_url else ""

                    movie_info = f'<span style="color: #FF424D; font-weight: bold;">Movie Timecode:</span> {col4}' if pd.notna(col4) else ""

                    combined_html = f'{youtube_icon} {movie_info}'

                    st.markdown(combined_html, unsafe_allow_html=True)

                st.write("---")
        else:
            st.write("No bits found.")

st.markdown(
    "<h1 style='text-align: center; font-size: 16pt;'>TIP: For faster searching, choose a miniseries first.</h1>",
    unsafe_allow_html=True
)

google_form_text = "This buffoonery is not officially sanctioned by the <a href='https://www.blankcheckpod.com'>Blank Check</a> podcast."
st.markdown(f'<div style="text-align: center; font-size: 12px;">{google_form_text}</div>', unsafe_allow_html=True)

footer_text = "For inquiries, email us at <a href='mailto:blankcheckdb@gmail.com'>blankcheckdb@gmail.com</a>"
st.write(f'<div style="text-align: center;font-size: 12px;">{footer_text}</div>', unsafe_allow_html=True)

footer_text = "<a href='https://www.youtube.com/watch?v=MNLTgUZ8do4&t=3928s'>Beta</a> build December 23, 2024"
st.write(f'<div style="text-align: center;font-size: 12px;">{footer_text}</div>', unsafe_allow_html=True)