import os
import re
import streamlit as st
import glob
from google.oauth2 import service_account
from google.cloud import storage
import base64
from concurrent.futures import ThreadPoolExecutor, as_completed

credentials = service_account.Credentials.from_service_account_info(
    st.secrets["gcp_service_account"]
)
client = storage.Client(credentials=credentials)

bucket_name = 'bcdb_episodes'

st.set_page_config(page_title="Blank Check Database", page_icon=":mag_right:")
logo_url = "https://storage.googleapis.com/bcdb_images/BCDb_logo_apr10.png"
st.markdown(f'<div style="text-align: center;"><img src="{logo_url}" width="300"></div>', unsafe_allow_html=True)

st.markdown("<h1 style='text-align: center;'><span style='color: #AE88E1;'>Blank Check </span><span style='color: #8E3497;'>Database</span></h1>", unsafe_allow_html=True)

bucket = client.get_bucket(bucket_name)
blobs = bucket.list_blobs()
unique_folder_names = sorted(set(os.path.dirname(blob.name) for blob in blobs if blob.name.endswith('.txt')))
folder_names = {"All Miniseries": "all", **{os.path.basename(folder)[4:].replace('_', ' '): folder for folder in unique_folder_names if folder}}
folder_name = st.selectbox("Select a Miniseries:", list(folder_names.keys()))

search_term = st.text_input("Enter search term:", value="", key="search_box", max_chars=None, type="default", help=None, placeholder="e.g. Star Wars")
highlight_color = "#E392EA"
button_clicked = st.button("Search")

def search_file(blob, search_term):
    file_path = blob.name
    if not file_path.endswith('.txt'):
        return None

    content = blob.download_as_text()
    lines = content.split('\n')

    matches = []
    YouTube_url = None
    Soundcloud_url = None

    for line in lines:
        if 'www.youtube.com' in line:
            YouTube_url = line
        elif 'soundcloud.com' in line:
            Soundcloud_url = line
        else:
            if not any(tag in line for tag in ['<span', '</span', '<color', '</color']):
                if re.search(rf'\b{re.escape(search_term)}\b', line, flags=re.IGNORECASE):
                    matches.append(line)

    return file_path, matches, YouTube_url, Soundcloud_url

CACHE = {}

def search_files(search_term, bucket_name, folder_name):
    cache_key = (search_term, folder_name)
    if cache_key in CACHE:
        return CACHE[cache_key]

    matches_dict, YouTube_urls, Soundcloud_urls = {}, {}, {}

    bucket = client.get_bucket(bucket_name)
    blobs = bucket.list_blobs() if folder_name == "all" else bucket.list_blobs(prefix=folder_name)

    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(search_file, blob, search_term) for blob in blobs]

        for future in as_completed(futures):
            result = future.result()
            if result:
                file_path, matches, YouTube_url, Soundcloud_url = result
                if matches:
                    matches_dict[file_path] = matches
                if YouTube_url:
                    YouTube_urls[file_path] = YouTube_url
                if Soundcloud_url:
                    Soundcloud_urls[file_path] = Soundcloud_url

    result = (matches_dict, YouTube_urls, Soundcloud_urls)
    CACHE[cache_key] = result
    return result

def convert_timecode(timecode):
    # Extract hours, minutes, and seconds from the timecode
    match = re.search(r'\[(\d{2}):(\d{2}):(\d{2})\]', timecode)
    if not match:
        return None
    hours, minutes, seconds = match.groups()

    # Convert hours, minutes, and seconds to a single integer
    time_in_seconds = int(hours) * 3600 + int(minutes) * 60 + int(seconds)
    return time_in_seconds

def extract_number(file_name):
    match = re.search(r'\d+', file_name)
    return int(match.group()) if match else float('inf')

if button_clicked:
    search_term = search_term.strip()  # Remove leading/trailing whitespaces
    if len(search_term) < 2:
        st.write("Please enter a search term with at least 2 characters.")
    else:
        matches_dict, YouTube_urls, Soundcloud_urls = search_files(search_term, bucket_name, folder_names[folder_name])
        if matches_dict:
            st.write(f"Search results ({sum(len(lines) for lines in matches_dict.values())}):")
            for file_path, lines in sorted(matches_dict.items(), key=lambda x: (x[0].rsplit('/', 1)[-1], extract_number(x[0]))):
                file_name = os.path.splitext(os.path.basename(file_path))[0].replace('_', ' ')
                formatted_file_name = file_name.replace('_', ' ')
                file_content = client.get_bucket(bucket_name).get_blob(file_path).download_as_text()
                file_data = base64.b64encode(file_content.encode('utf-8')).decode('utf-8')
                download_button = f'<a href="data:text/plain;base64,{file_data}" download="{formatted_file_name}.txt">Download Transcript</a>'

                public_url = f'https://storage.googleapis.com/{bucket_name}/{file_path}'
                view_button = f'<a href="{public_url}" target="_blank">View Transcript</a>'

                st.markdown(f"<span style='font-size: 25px; color: #AE88E1; font-weight: bold;'>{formatted_file_name[4:]}:</span><br>{download_button} | {view_button}", unsafe_allow_html=True)

                for line in lines:
                    line = re.sub(f'({re.escape(search_term)})', f"<span style='background-color: {highlight_color}'>\\1</span>", line, flags=re.IGNORECASE)
                    if file_path in YouTube_urls and file_path in Soundcloud_urls:
                        timecode = convert_timecode(line[:10])
                        YouTube_URL = YouTube_urls[file_path].strip() + "&t=" + str(timecode)
                        Soundcloud_url = Soundcloud_urls[file_path].strip() + "#t=" + str(timecode)
                        line = line[:10] + line[10:] + f"<br><a href='{YouTube_URL}' target='_blank'><img src='https://storage.googleapis.com/bcdb_images/Youtube_logo.png' alt='YouTube' width='30' height='20'></a>&nbsp;<a href='{Soundcloud_url}' target='_blank'><img src='https://storage.googleapis.com/bcdb_images/soundcloud_logo.png' alt='SoundCloud' width='20' height='20'></a>"
                    elif file_path in YouTube_urls:
                        timecode = convert_timecode(line[:10])
                        YouTube_URL = YouTube_urls[file_path].strip() + "&t=" + str(timecode)
                        line = line[:10] + line[10:] + f"<br><a href='{YouTube_URL}' target='_blank'><img src='https://storage.googleapis.com/bcdb_images/Youtube_logo.png' alt='YouTube' width='30' height='20'></a>"
                    elif file_path in Soundcloud_urls:
                        timecode = convert_timecode(line[:10])
                        Soundcloud_url = Soundcloud_urls[file_path].strip() + "#t=" + str(timecode)
                        line = line[:10] + line[10:] + f"<br><a href='{Soundcloud_url}' target='_blank'><img src='https://storage.googleapis.com/bcdb_images/soundcloud_logo.png' alt='SoundCloud' width='20' height='20'></a>"
                    st.markdown(line, unsafe_allow_html=True)
        else:
            st.write("No bits found.")

google_form_text = "Not affiliated with the <a href='https://www.blankcheckpod.com'>Blank Check</a> podcast. Bugs or feature request? Fill out this <a href='https://docs.google.com/forms/d/1pxJjxpV_vBE9__YRnlzXOo9cK6mrXgwgmfR2-sd8hds/edit'>form</a>."
st.markdown(f'<div style="text-align: center; font-size: 12px;">{google_form_text}</div>', unsafe_allow_html=True)

email_address = "blankcheckdb@gmail.com"
email_link = f'<a href="mailto:{email_address}">{email_address}</a>'
email_text = "If you would like to help with this project, email here:"
st.markdown(f'<div style="text-align: center;font-size: 12px;">{email_text} {email_link}</div>', unsafe_allow_html=True)

footer_text = "<a href='https://www.youtube.com/watch?v=MNLTgUZ8do4&t=3928s'>Beta</a> build April 19, 2023"
st.write(f'<div style="text-align: center;font-size: 12px;">{footer_text}</div>', unsafe_allow_html=True)