import os
import re
import base64
import math
from io import StringIO
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict

import pandas as pd
import streamlit as st
from google.cloud import storage
from google.oauth2 import service_account

from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

import random
import streamlit.components.v1 as components

from datetime import datetime, timezone, timedelta
import uuid, json

credentials = service_account.Credentials.from_service_account_info(
    st.secrets["gcp_service_account"]
)
client = storage.Client(credentials=credentials)

st.set_page_config(page_title="Blank Check Database", page_icon=":mag_right:")

SHEETS_SPREADSHEET_ID = "185kZdzb2_mzjqHyPTux1MuhatZuX3SzwnGcW4VMkbYk"
SHEETS_TAB_NAME = "search_log" 

AUTO_AGGREGATE_EVERY_MINUTES = 60 

SOUND_TRIGGERS = [
    {
        "id": "uk",
        "patterns": [r"\blondon\b", r"\bengland\b", r"\bgreat britain\b", r"\buk\b"],
        "url": "https://storage.googleapis.com/bcdb_audio/big_ben_chimes.mp3",
        "type": "regex_any",
    },
    {
         "id": "twisted",
         "patterns": [r"\btwisted\b"],
         "url": "https://storage.googleapis.com/bcdb_audio/twisted.mp3",
         "type": "regex_any",
    },
    {
         "id": "unbreakable",
         "patterns": [r"\bunbreakable\b"],
         "url": "https://storage.googleapis.com/bcdb_audio/unbreakable.mp3",
         "type": "regex_any",
    },    
    {
         "id": "comedy_point",
         "patterns": [r"\bcomedy points\b"],
         "url": "https://storage.googleapis.com/bcdb_audio/comedy_point.mp3",
         "type": "regex_any",
    },
]

for trig in SOUND_TRIGGERS:
    if trig.get("type", "regex_any").startswith("regex"):
        trig["_compiled"] = [re.compile(p, re.IGNORECASE) for p in trig["patterns"]]

def _first_matching_trigger(term: str):
    t = term.strip()
    tl = t.lower()
    for trig in SOUND_TRIGGERS:
        mtype = trig.get("type", "regex_any")
        if mtype == "regex_any":
            if any(rx.search(t) for rx in trig["_compiled"]):
                return trig
        elif mtype == "exact":
            if any(tl == p.lower() for p in trig["patterns"]):
                return trig
        elif mtype == "substring_any":
            if any(p.lower() in tl for p in trig["patterns"]):
                return trig
    return None

def _play_sound_once_per_term(trigger_id: str, term: str, sound_url: str):
    key = f"{trigger_id}::{term.lower()}"
    already = st.session_state.get("played_sounds", [])
    aset = set(already)
    if key in aset:
        return
    components.html(
        f"""
        <audio autoplay>
            <source src="{sound_url}">
            Your browser does not support the audio element.
        </audio>
        """,
        height=0,
    )
    aset.add(key)
    st.session_state["played_sounds"] = list(aset)

@st.cache_resource
def get_gcs_client():
    return storage.Client(credentials=credentials)

@st.cache_resource
def get_sheets_service():
    scoped = credentials.with_scopes(["https://www.googleapis.com/auth/spreadsheets"])
    return build("sheets", "v4", credentials=scoped, cache_discovery=False)

@st.cache_data(show_spinner=False, ttl=3600)
def list_csv_blobs(bucket_name: str, prefix: str | None):
    client = get_gcs_client()
    bucket = client.bucket(bucket_name)
    return [b.name for b in bucket.list_blobs(prefix=prefix) if b.name.lower().endswith('.csv')]

@st.cache_data(show_spinner=False, ttl=3600)
def list_all_miniseries(bucket_name: str):
    names = list_csv_blobs(bucket_name, prefix=None)
    folders = sorted({os.path.dirname(n) for n in names if os.path.dirname(n)})
    mapping = { (os.path.basename(f)[4:].replace('_',' ')): f for f in folders }
    return {"All Miniseries": "all", **mapping}

@st.cache_data(show_spinner=False, ttl=3600)
def cached_image_url(bucket_name: str, file_name: str):
    client = get_gcs_client()
    bucket = client.bucket('bcdb_images') if bucket_name != 'bcdb_images' else client.bucket(bucket_name)
    stem = os.path.splitext(os.path.basename(file_name))[0]
    guess_paths = [
        f"episode_art/{stem}.jpg",
        f"episode_art/{stem}.jpeg",
        f"episode_art/{stem}.png",
        f"episode_art/{stem}.gif",
    ]
    for gp in guess_paths:
        blob = bucket.blob(gp)
        if blob.exists(client):
            return f"https://storage.googleapis.com/{bucket.name}/{gp}"
    folder_name = os.path.basename(os.path.dirname(file_name))
    for b in bucket.list_blobs(prefix=f"episode_art/{folder_name}"):
        if b.name.lower().endswith(('.jpg','.jpeg','.png','.gif')):
            return f"https://storage.googleapis.com/{bucket.name}/{b.name}"
    return None

def get_youtube_url_for_blob(bucket_name: str, blob_name: str) -> str | None:
    client = get_gcs_client()
    blob = client.bucket(bucket_name).blob(blob_name)
    try:
        head = pd.read_csv(
            blob.open("rt"),
            sep=";",
            engine="c",
            dtype=str,
            nrows=3,
            header=None,
            on_bad_lines="skip",
        )
    except Exception:
        return None
    head0 = head.iloc[:, 0].astype(str).tolist() + ["", "", ""]
    youtube, _ = _first_urls_from_head0(head0)
    return youtube

def _get_session_id() -> str:
    if "session_id" not in st.session_state:
        st.session_state["session_id"] = str(uuid.uuid4())
    return st.session_state["session_id"]

def _ensure_sheet_header():
    try:
        svc = get_sheets_service()
        rng = f"{SHEETS_TAB_NAME}!A1:G1"
        res = svc.spreadsheets().values().get(
            spreadsheetId=SHEETS_SPREADSHEET_ID, range=rng
        ).execute()
        vals = res.get("values", [])
        if not vals:
            header = [["ts_utc", "session_id", "feed", "folder", "term", "results_count", "hour_utc"]]
            svc.spreadsheets().values().update(
                spreadsheetId=SHEETS_SPREADSHEET_ID,
                range=rng,
                valueInputOption="RAW",
                body={"values": header},
            ).execute()
    except HttpError as e:
        st.warning(f"Could not ensure header for {SHEETS_TAB_NAME}: {e}")

def log_search_event(term: str, feed: str, folder: str, results_count: int):
    try:
        _ensure_sheet_header()
        now = datetime.now(timezone.utc)
        svc = get_sheets_service()
        row = [
            now.isoformat(),
            _get_session_id(),
            feed,
            folder,
            term,
            int(results_count),
            now.strftime("%H"),
        ]
        svc.spreadsheets().values().append(
            spreadsheetId=SHEETS_SPREADSHEET_ID,
            range=f"{SHEETS_TAB_NAME}!A1",
            valueInputOption="RAW",
            insertDataOption="INSERT_ROWS",
            body={"values": [row]},
        ).execute()
    except Exception as e:
        print(f"[search logging to Sheets skipped] {e}")

def maybe_auto_aggregate():
    pass

term_styles = {
    'comedy points': {'color': '#D4AF37', 'emoji': ''},
    'night eggs': {'color': '#AE88E1', 'emoji': '🥚'},
    'river of ham': {'color': '#AE88E1', 'emoji': '🐷'},
    'david dog': {'color': '#AE88E1', 'emoji': '🐶'},
    'London': {'color': '#AE88E1', 'emoji': '🇬🇧'},
    'England': {'color': '#AE88E1', 'emoji': '🇬🇧'},
    'Great Britain': {'color': '#AE88E1', 'emoji': '🇬🇧'},
    'burger report': {'color': '#AE88E1', 'emoji': '🍔'},
}

def highlight_term_html(text: str, pattern: re.Pattern, raw_term_for_style: str) -> str:
    style = term_styles.get(raw_term_for_style.lower(), {'color': '#AE88E1', 'emoji': ''})
    color = style['color']
    emoji = style['emoji']
    def repl(m: re.Match):
        return f'<span style="font-weight:bold;color:{color};">{m.group(0)}{emoji}</span>'
    return pattern.sub(repl, text)

YOUTUBE_ICON = "https://storage.googleapis.com/bcdb_images/Youtube_logo.png"
PATREON_ICON = "https://storage.googleapis.com/bcdb_images/patreon_logo.png"
LOGO_URL = "https://storage.googleapis.com/bcdb_images/BCDb_logo_2025.png"

def _first_urls_from_head0(head0: list[str]):
    youtube = next((u for u in head0 if isinstance(u, str) and 'youtube' in u.lower()), None)
    patreon = next((u for u in head0 if isinstance(u, str) and 'patreon' in u.lower()), None)
    return youtube, patreon

def scan_csv_for_matches(blob: storage.Blob, pattern: re.Pattern):
    out = []
    head_done = False
    youtube = patreon = None
    for chunk in pd.read_csv(
        blob.open("rt"),
        sep=";",
        engine="c",
        dtype=str,
        chunksize=50_000,
        header=None,
        on_bad_lines='skip'
    ):
        chunk = chunk.fillna('')
        if not head_done and not chunk.empty:
            head0 = chunk.iloc[:3, 0].astype(str).tolist() + ['', '', '']
            youtube, patreon = _first_urls_from_head0(head0)
            head_done = True
        if chunk.shape[1] < 3:
            continue
        mask = chunk.iloc[:, 2].astype(str).str.contains(pattern, na=False)
        if not mask.any():
            continue
        for _, row in chunk.loc[mask].iterrows():
            tc = row.iloc[0] if len(row) > 0 else ''
            line = row.iloc[2] if len(row) > 2 else ''
            movie_time = row.iloc[3] if len(row) > 3 else ''
            red_flag = row.iloc[4] if len(row) > 4 else ''
            out.append((youtube, patreon, tc, line, movie_time, red_flag))
    return out

@st.cache_data(show_spinner=False, ttl=600)
def searching(bucket_name: str, search_term: str, folder_path: str):
    if not search_term:
        return {}
    client = get_gcs_client()
    bucket = client.bucket(bucket_name)
    names = list_csv_blobs(bucket_name, None if folder_path == 'all' else folder_path)
    if not names:
        return {}
    pattern = re.compile(rf"\b{re.escape(search_term)}\b", re.IGNORECASE)
    results: dict[str, list[tuple]] = {}
    max_workers = min(32, max(1, len(names)))
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        future_to_name = {ex.submit(scan_csv_for_matches, bucket.blob(n), pattern): n for n in names}
        for fut in as_completed(future_to_name):
            name = future_to_name[fut]
            rows = fut.result()
            if rows:
                results[name] = rows
    return results

def extract_number(file_name: str):
    m = re.search(r"\d+", file_name)
    return int(m.group()) if m else float('inf')

def time_to_seconds(time_str: str) -> int:
    try:
        h, m, s = map(int, time_str.split(':'))
        return h * 3600 + m * 60 + s
    except Exception:
        return 0

def build_view_and_download_links(bucket_name: str, blob_name: str):
    public_url = f"https://storage.googleapis.com/{bucket_name}/{blob_name}"
    view_btn = f'<a href="{public_url}" target="_blank">View</a>'
    download_btn = f'<a href="{public_url}" download target="_blank">Download</a>'
    return view_btn, download_btn

def ensure_random_url(bucket_name: str, folder_path: str):
    ctx = (bucket_name, folder_path)
    if st.session_state.get("random_ctx") == ctx and st.session_state.get("random_url"):
        return 

    names = list_csv_blobs(bucket_name, None if folder_path == "all" else folder_path)
    yt_url = None
    if names:
        attempts = min(20, len(names))
        for _ in range(attempts):
            candidate = random.choice(names)
            yt_url = get_youtube_url_for_blob(bucket_name, candidate)
            if yt_url:
                break

    st.session_state["random_ctx"] = ctx
    st.session_state["random_url"] = yt_url

st.markdown(f'<div style="text-align:center;"><img src="{LOGO_URL}" width="300"></div>', unsafe_allow_html=True)
st.markdown("<h1 style='text-align:center;'><span style='color:#AE88E1;'>Blank Check </span><span style='color:#8E3497;'>Database</span></h1>", unsafe_allow_html=True)

bucket_name_mapping = {
    'Main Feed': 'bcdb_episodes',
    'Patreon': 'bcdb_patreon',
}

try:
    feed = st.segmented_control(
        "Select feed:",
        options=list(bucket_name_mapping.keys()),
        selection_mode="single",
        default="Main Feed",
    )
except Exception:
    feed = st.radio(
        "Select feed:",
        options=list(bucket_name_mapping.keys()),
        horizontal=True,
        index=0,
    )

bucket_name = bucket_name_mapping[feed]

maybe_auto_aggregate()

folder_names = list_all_miniseries(bucket_name)
folder_choice = st.selectbox("Select a Miniseries:", list(folder_names.keys()))
folder_path = folder_names[folder_choice]

if 'button_clicked' not in st.session_state:
    st.session_state['button_clicked'] = False
if 'page' not in st.session_state:
    st.session_state['page'] = 1
if 'played_sounds' not in st.session_state:
    st.session_state['played_sounds'] = []

def _trigger_search():
    st.session_state['button_clicked'] = True
    st.session_state['page'] = 1

PLACEHOLDER_OPTIONS = [
    "e.g. Star Wars",
    "e.g. Beta Cuck Movement",
    "e.g. Big Chicago",
    "e.g. Chip Smith",
    "e.g. Retired Bit",
    "e.g. Watto",
    "e.g. Owns bones",
    "e.g. Night Eggs",
    "e.g. Hello, Fennel",
    "e.g. Buried Jeans",
    "e.g. Burger Report",  
    "e.g. Humblebrag", 
    "e.g. BLANK IT",  
    "e.g. Sully",  
    "e.g. Wild Hogs",  
    "e.g. Blarp",  
    "e.g. Boss Baby",
    "e.g. Box office game",    
    "e.g. Decade of dreams", 
]

if "search_placeholder" not in st.session_state:
    st.session_state["search_placeholder"] = random.choice(PLACEHOLDER_OPTIONS)

if "search_box" not in st.session_state:
    st.session_state["search_box"] = ""

search_term = st.text_input(
    "Enter search term:",
    key="search_box",
    placeholder=st.session_state["search_placeholder"],
    on_change=_trigger_search,
)

ensure_random_url(bucket_name, folder_path)

c1, c2, c3 = st.columns(3)
with c1:
    search_clicked = st.button("Search", use_container_width=True)
with c2:
    reset_clicked = st.button("Reset", type="primary", use_container_width=True)
with c3:
    url = st.session_state.get("random_url")
    if url:
        try:
            st.link_button("▶ Play random episode", url, use_container_width=True)
        except Exception:
            st.markdown(
                f'<a href="{url}" target="_blank" rel="noopener">▶ Play random episode</a>',
                unsafe_allow_html=True,
            )
    else:
        st.button("▶ Play random episode", disabled=True, help="No YouTube link found", use_container_width=True)

run_search = search_clicked or st.session_state.get('button_clicked', False)

if reset_clicked:
    st.session_state['button_clicked'] = False
    st.session_state['page'] = 1
    st.session_state.pop("random_url", None)
    st.session_state.pop("random_ctx", None)
    st.rerun()

if run_search:
    term = search_term.strip()

    matched = _first_matching_trigger(term)
    if matched:
        _play_sound_once_per_term(matched["id"], term, matched["url"])

    if not term:
        st.write("Please enter a search term")
    else:
        with st.spinner("Searching episodes…"):
            results = searching(bucket_name, term, folder_path)

        total = sum(len(v) for v in results.values()) if results else 0

        log_search_event(term=term, feed=feed, folder=folder_choice, results_count=total)

        if not results:
            st.write("No bits found.")
        else:
            st.write(f"({total}) {'result' if total == 1 else 'results'} found")

            sorted_items = sorted(results.items(), key=lambda x: (x[0].rsplit('/',1)[-1], extract_number(x[0])))
            compiled = re.compile(rf"\b{re.escape(term)}\b", re.IGNORECASE)

            total_files = len(sorted_items)
            page_size = 20
            total_pages = max(1, math.ceil(total_files / page_size))
            st.session_state['page'] = max(1, min(st.session_state.get('page', 1), total_pages))

            start = (st.session_state['page'] - 1) * page_size
            end = start + page_size
            page_items = sorted_items[start:end]

            col_prev, col_info, col_next = st.columns([1, 2, 1])
            with col_prev:
                if st.button("◀ Prev", disabled=st.session_state['page'] <= 1, key="prev_btn"):
                    st.session_state['page'] -= 1
                    st.rerun()
            with col_info:
                st.markdown(
                    f"<div style='text-align:center;'>Page {st.session_state['page']} of {total_pages} &nbsp;•&nbsp; "
                    f"Showing files {start+1}-{min(end, total_files)} of {total_files}</div>",
                    unsafe_allow_html=True
                )
            with col_next:
                if st.button("Next ▶", disabled=st.session_state['page'] >= total_pages, key="next_btn"):
                    st.session_state['page'] += 1
                    st.rerun()

            for file_name, rows in page_items:
                file_title = os.path.basename(os.path.splitext(file_name)[0])[4:].replace('_',' ')
                view_btn, download_btn = build_view_and_download_links(bucket_name, file_name)

                if folder_choice == "All Miniseries":
                    img_url = cached_image_url('bcdb_images', file_name)
                    if img_url:
                        st.markdown(f'<div style="text-align:left;"><img src="{img_url}" width="200"></div>', unsafe_allow_html=True)

                patreon_url_any = next((r[1] for r in rows if r[1]), None)
                patreon_html = f'<a href="{patreon_url_any}" target="_blank"><img src="{PATREON_ICON}" width="20"></a>' if patreon_url_any else ''
                header_html = (
                    f"<span style='font-size:25px;color:#AE88E1;font-weight:bold;'>{patreon_html} {file_title} "
                    f"<span style='font-size:15pt;color:#8E3497;'>({len(rows)} {'result' if len(rows)==1 else 'results'})</span>:</span><br>"
                    f"Transcript: {view_btn} | {download_btn}"
                )
                st.markdown(header_html, unsafe_allow_html=True)

                with st.expander(f"Matches: {len(rows)}", expanded=True):
                    parts = []
                    for (youtube_url, patreon_url, timecode, line, movie_time, red_flag) in rows:
                        compiled_term = re.compile(rf"\b{re.escape(term)}\b", re.IGNORECASE)
                        line_html = highlight_term_html(line, compiled_term, term)
                        tsec = time_to_seconds(timecode)
                        yt_icon = f'<a href="{youtube_url}&t={tsec}" target="_blank"><img src="{YOUTUBE_ICON}" width="20" style="vertical-align:middle;"></a>' if youtube_url else ''

                        tc_html = f"<span style=\"color:#8E3497;font-weight:bold;\">[{timecode}]:</span>"

                        line_block = f"<div style=\"margin-bottom:6px;\">{yt_icon} {tc_html} "
                        if isinstance(red_flag, str) and red_flag.strip():
                            line_block += f"<span style=\"color:red;\">{line_html}</span>"
                        else:
                            line_block += f"{line_html}"
                        line_block += "</div>"

                        movie_info = f'<span style=\"color:#FF424D;font-weight:bold;\">Movie Timecode:</span> {movie_time}' if isinstance(movie_time, str) and movie_time.strip() else ''
                        if movie_info:
                            line_block += f"<div>{movie_info}</div>"

                        parts.append(line_block)
                    st.markdown("".join(parts), unsafe_allow_html=True)

                st.write("---")

            col_prev2, col_info2, col_next2 = st.columns([1, 2, 1])
            with col_prev2:
                if st.button("◀ Prev", disabled=st.session_state['page'] <= 1, key="prev_btn_bottom"):
                    st.session_state['page'] -= 1
                    st.rerun()
            with col_info2:
                st.markdown(f"<div style='text-align:center;'>Page {st.session_state['page']} of {total_pages}</div>", unsafe_allow_html=True)
            with col_next2:
                if st.button("Next ▶", disabled=st.session_state['page'] >= total_pages, key="next_btn_bottom"):
                    st.session_state['page'] += 1
                    st.rerun()

st.markdown(
    "<h1 style='text-align:center;font-size:16pt;'>TIP: For faster searching, choose a miniseries first.</h1>",
    unsafe_allow_html=True,
)

info_text = "This buffoonery is not officially sanctioned by the <a href='https://www.blankcheckpod.com'>Blank Check</a> podcast."
st.markdown(f'<div style=\"text-align:center;font-size:12px;\">{info_text}</div>', unsafe_allow_html=True)

footer_text = "<a href='https://www.youtube.com/watch?v=MNLTgUZ8do4&t=3928s'>Beta</a> build August 2025"
st.write(f'<div style=\"text-align:center;font-size:12px;\">{footer_text}</div>', unsafe_allow_html=True)