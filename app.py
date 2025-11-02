import os
import re
import io
import json
import time
import uuid
import base64
import zipfile
import random
import string
import textwrap
import requests
import boto3
import nltk
from urllib.parse import urlparse
from pathlib import Path
from collections import OrderedDict
from datetime import datetime, timezone
from dotenv import load_dotenv
import streamlit as st
from openai import AzureOpenAI

# =========================
# Base Config & Utilities
# =========================
load_dotenv()

# NLTK once
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# ---------- Constants (edit to your CDN paths) ----------
DEFAULT_COVER_URL = "https://media.suvichaar.org/upload/covers/default_news.png"
DEFAULT_SLIDE_IMAGE_URL = "https://media.suvichaar.org/upload/covers/default_news_slide.png"
DEFAULT_CTA_AUDIO = "https://cdn.suvichaar.org/media/tts_cta_default.mp3"

# Enforce total slides (Headline + main slides + Hookline). CTA is extra.
MIN_TOTAL_SLIDES = 8
MAX_TOTAL_SLIDES = 10

# ---------- Secrets ----------
client = AzureOpenAI(
    azure_endpoint=st.secrets["azure_api"]["AZURE_OPENAI_ENDPOINT"],
    api_key=st.secrets["azure_api"]["AZURE_OPENAI_API_KEY"],
    api_version="2025-01-01-preview",
)

AZURE_TTS_URL = st.secrets["azure"]["AZURE_TTS_URL"]
AZURE_API_KEY = st.secrets["azure"]["AZURE_API_KEY"]

AWS_ACCESS_KEY = st.secrets["aws"]["AWS_ACCESS_KEY"]
AWS_SECRET_KEY = st.secrets["aws"]["AWS_SECRET_KEY"]
AWS_REGION     = st.secrets["aws"]["AWS_REGION"]
AWS_BUCKET     = st.secrets["aws"]["AWS_BUCKET"]
S3_PREFIX      = "media/"
CDN_BASE       = st.secrets["aws"]["CDN_BASE"]
CDN_PREFIX_MEDIA = "https://media.suvichaar.org/"

s3_client = boto3.client(
    "s3",
    aws_access_key_id=AWS_ACCESS_KEY,
    aws_secret_access_key=AWS_SECRET_KEY,
    region_name=AWS_REGION,
)

voice_options = {
    "1": "alloy",
    "2": "echo",
    "3": "fable",
    "4": "onyx",
    "5": "nova",
    "6": "shimmer",
}

def clamp_main_slides(requested_main: int) -> int:
    """
    requested_main = slides between Headline and Hookline.
    We clamp TOTAL slides (Headline + main + Hookline) into [8..10].
    CTA is not counted in this min/max.
    """
    total = 2 + int(requested_main)  # +Headline +Hookline
    total = max(MIN_TOTAL_SLIDES, min(MAX_TOTAL_SLIDES, total))
    return max(0, total - 2)

def generate_slug_and_urls(title: str):
    if not title or not isinstance(title, str):
        raise ValueError("Invalid title")
    slug = ''.join(
        c for c in title.lower().replace(" ", "-").replace("_", "-")
        if c in string.ascii_lowercase + string.digits + '-'
    ).strip('-')
    nano = ''.join(random.choices(string.ascii_letters + string.digits + '_-', k=10)) + '_G'
    slug_nano = f"{slug}_{nano}"
    return nano, slug_nano, f"https://suvichaar.org/stories/{slug_nano}", f"https://stories.suvichaar.org/{slug_nano}.html"

# =========================
# Extract / Analyze
# =========================
def extract_article(url):
    # lazy import to avoid installing when not needed
    import newspaper
    from newspaper import Article
    try:
        article = Article(url)
        article.download()
        article.parse()
        try:
            article.nlp()
        except Exception:
            pass
        title = (article.title or "Untitled Article").strip()
        text = (article.text or "No article content available.").strip()
        summary = (article.summary or text[:300]).strip()
        return title, summary, text
    except Exception as e:
        st.error(f"❌ Failed to extract article: {e}")
        return "Untitled Article", "No summary available.", "No article content available."

def get_sentiment(text: str) -> str:
    """
    Dependency-free lightweight heuristic sentiment.
    Returns: 'positive' | 'negative' | 'neutral'
    """
    if not text or not str(text).strip():
        return "neutral"
    t = str(text).lower()

    pos_words = {
        "growth","improved","record","gain","surge","soar","win","wins","victory","milestone",
        "boost","increase","expansion","positive","rise","strong","beat","beats","achieve","achieved"
    }
    neg_words = {
        "decline","drop","fall","fell","loss","losses","downturn","crash","fail","failed","failure",
        "delay","delayed","negative","fraud","scam","ban","banned","risk","risks","cut","cuts","layoff",
        "layoffs","debt","default","crisis","crises","probe","investigation"
    }

    pos = sum(w in t for w in pos_words)
    neg = sum(w in t for w in neg_words)
    if pos - neg > 1:
        return "positive"
    if neg - pos > 1:
        return "negative"
    return "neutral"

def detect_category_and_subcategory(text, content_language="English"):
    if not text or len(text.strip()) < 50:
        return {"category": "Unknown", "subcategory": "General", "emotion": "Neutral"}

    if content_language == "Hindi":
        prompt = f"""
आप एक समाचार विश्लेषण विशेषज्ञ हैं।
इस लेख का विश्लेषण करके JSON दें:

{{
  "category": "...",
  "subcategory": "...",
  "emotion": "..."
}}

लेख:
\"\"\"{text[:3000]}\"\"\""""
    else:
        prompt = f"""
You are a news analysis expert. Return JSON only:

{{
  "category": "...",
  "subcategory": "...",
  "emotion": "..."
}}

Article:
\"\"\"{text[:3000]}\"\"\""""

    try:
        resp = client.chat.completions.create(
            model="gpt-5-chat",
            messages=[
                {"role": "system", "content": "Classify the news article."},
                {"role": "user", "content": prompt.strip()},
            ],
            max_tokens=150,
        )
        content = resp.choices[0].message.content.strip()
        content = content.strip("```json").strip("```").strip()
        data = json.loads(content)
        if all(k in data for k in ("category", "subcategory", "emotion")):
            return data
    except Exception:
        pass
    return {"category": "Unknown", "subcategory": "General", "emotion": "Neutral"}

# =========================
# Slide Generator (No Polaris)
# =========================
def title_script_generator(category, subcategory, emotion, article_text, content_language="English"):
    """Enforces:
    - Slide 1 = Headline
    - Slide 2 = connected 'what/why' context
    - Remaining slides = concise points
    (Narrations created later)
    """
    sys = f"""
You are a digital editor.
Create a structured web story outline in {content_language}:
- Slide 1: the news headline (short)
- Slide 2: a connected one-liner (what/why context)
- Slides 3..N: concise points derived from the article
Return JSON only with:
{{ "slides": [{{"title":"..." , "prompt":"instruction to narrate"}}, ...] }}
"""
    usr = f"""
Category: {category}
Subcategory: {subcategory}
Emotion: {emotion}

Article:
\"\"\"{article_text[:3000]}\"\"\""""

    resp = client.chat.completions.create(
        model="gpt-5-chat",
        messages=[
            {"role": "system", "content": sys.strip()},
            {"role": "user", "content": usr.strip()},
        ],
        temperature=0.5,
    )
    raw = resp.choices[0].message.content.strip()
    raw = raw.strip("```json").strip("```").strip()

    try:
        slides = json.loads(raw)["slides"]
    except Exception:
        slides = []

    return {"category": category, "subcategory": subcategory, "emotion": emotion, "slides": slides}

def restructure_slide_output(final_output, main_slide_count):
    """Return dict with s1..sNparagraph1 placeholders (content only).
    We'll generate narrations per slide later."""
    slides = final_output.get("slides", [])[:main_slide_count]  # clamp early
    structured = OrderedDict()
    for idx, slide in enumerate(slides, start=1):
        key = f"s{idx}paragraph1"
        # Prefer the slide title as the visible line; narration will be separate audio
        text = (slide.get("title") or slide.get("prompt") or "").strip()
        structured[key] = text or f"Slide {idx}"
    return structured

# =========================
# Language helpers (optional Hindi)
# =========================
def transliterate_to_devanagari(json_data):
    """Transliterate only sXparagraph1 keys if content is Latin-script Hindi"""
    updated = {}
    for k, v in json_data.items():
        if k.startswith("s") and "paragraph1" in k and isinstance(v, str) and v.strip():
            prompt = f"Transliterate this Hindi sentence (Latin) into Devanagari. Return only the transliteration:\n\n{v}"
            try:
                r = client.chat.completions.create(
                    model="gpt-5-chat",
                    messages=[
                        {"role": "system", "content": "You are a Hindi transliteration expert."},
                        {"role": "user", "content": prompt.strip()},
                    ],
                )
                updated[k] = r.choices[0].message.content.strip()
            except Exception:
                updated[k] = v
        else:
            updated[k] = v
    return updated

def generate_hookline(title, summary, content_language="English"):
    if content_language == "Hindi":
        prompt = f"""
'सुविचार' चैनल के लिए इस समाचार का एक छोटा, ध्यान आकर्षित करने वाला हुकलाइन वाक्य लिखें।
- एक वाक्य
- हैशटैग/इमोजी/अधिक विराम चिह्न नहीं
- 120 वर्णों से कम
शीर्षक: {title}
सारांश: {summary}
आउटपुट: केवल हुकलाइन
"""
    else:
        prompt = f"""
Write a short, attention-grabbing hookline for this news story.
- One sentence
- No hashtags/emojis/excess punctuation
- Under 120 characters
Title: {title}
Summary: {summary}
Return only the hookline.
"""
    try:
        r = client.chat.completions.create(
            model="gpt-5-chat",
            messages=[
                {"role": "system", "content": "You create crisp hooklines for news."},
                {"role": "user", "content": prompt.strip()},
            ],
            temperature=0.5,
        )
        return r.choices[0].message.content.strip().strip('"')
    except Exception:
        return "This story might surprise you!"

def generate_storytitle(title, summary, content_language="English"):
    if content_language == "Hindi":
        prompt = f"""
अंग्रेज़ी शीर्षक और सारांश पढ़कर एक सरल, आकर्षक हिंदी शीर्षक बनाएं।
- एक पंक्ति
- उद्धरण न लगाएं
शीर्षक: {title}
सारांश: {summary}
आउटपुट: केवल नया शीर्षक
"""
        try:
            r = client.chat.completions.create(
                model="gpt-5-chat",
                messages=[
                    {"role": "system", "content": "You generate clear Hindi news headlines."},
                    {"role": "user", "content": prompt.strip()},
                ],
            )
            return r.choices[0].message.content.strip().strip('"')
        except Exception:
            return title.strip()
    else:
        return title.strip()

# =========================
# TTS + Upload (order: title -> s1..sN -> hookline -> CTA)
# =========================
def _tts_bytes(text: str, voice: str) -> bytes:
    r = requests.post(
        AZURE_TTS_URL,
        headers={"Content-Type": "application/json", "api-key": AZURE_API_KEY},
        json={"model": "tts-1-hd", "input": text, "voice": voice},
        timeout=60,
    )
    r.raise_for_status()
    return r.content

def _upload_bytes_to_s3(data: bytes, ext: str = ".mp3") -> str:
    filename = f"tts_{uuid.uuid4().hex}{ext}"
    key = f"{S3_PREFIX}{filename}"
    s3_client.put_object(Bucket=AWS_BUCKET, Key=key, Body=data, ContentType="audio/mpeg")
    return f"{CDN_BASE}{key}"

def synthesize_and_upload(paragraphs: dict, voice: str, add_cta=True) -> OrderedDict:
    """
    paragraphs must include:
      - storytitle
      - s1paragraph1..sNparagraph1
      - hookline
    Output format:
      slide1: { storytitle, audio_url, voice }
      slide2..: { sXparagraph1: "...", audio_url, voice }
      last content slide: hookline
      optional CTA final slide: fixed text + audio
    """
    result = OrderedDict()
    os.makedirs("temp", exist_ok=True)

    slide_index = 1

    # Slide 1 — title
    if "storytitle" in paragraphs and str(paragraphs["storytitle"]).strip():
        audio = _upload_bytes_to_s3(_tts_bytes(paragraphs["storytitle"], voice))
        result[f"slide{slide_index}"] = {
            "storytitle": paragraphs["storytitle"],
            "audio_url": audio,
            "voice": voice,
        }
        slide_index += 1

    # Slides 2..N — sXparagraph1 in numeric order
    s_keys = sorted(
        [k for k in paragraphs.keys() if k.startswith("s") and k.endswith("paragraph1")],
        key=lambda x: int(re.findall(r"s(\d+)paragraph1", x)[0])
    )
    for k in s_keys:
        txt = paragraphs[k]
        if not isinstance(txt, str) or not txt.strip():
            continue
        audio = _upload_bytes_to_s3(_tts_bytes(txt, voice))
        result[f"slide{slide_index}"] = {
            k: txt,
            "audio_url": audio,
            "voice": voice,
        }
        slide_index += 1

    # Last content slide — hookline
    if "hookline" in paragraphs and isinstance(paragraphs["hookline"], str) and paragraphs["hookline"].strip():
        audio = _upload_bytes_to_s3(_tts_bytes(paragraphs["hookline"], voice))
        result[f"slide{slide_index}"] = {
            "hookline": paragraphs["hookline"],
            "audio_url": audio,
            "voice": voice,
        }
        slide_index += 1

    # Final CTA (fixed text & your canned audio url)
    if add_cta:
        result[f"slide{slide_index}"] = {
            f"s{slide_index}paragraph1": "For Such Content Stay Connected with Suvichar Live\n\nRead | Share | Inspire",
            "audio_url": DEFAULT_CTA_AUDIO,
            "voice": voice,
        }

    return result

# =========================
# AMP helpers
# =========================
def generate_amp_slide(paragraph: str, audio_url: str):
    # Minimal, clean slide w/ background audio and one big text block
    return f"""
<amp-story-page auto-advance-after="audio-bg">
  <amp-story-grid-layer template="fill">
    <amp-video id="audio-bg" autoplay width="1" height="1" layout="fixed">
      <source type="audio/mpeg" src="{audio_url}">
    </amp-video>
  </amp-story-grid-layer>
  <amp-story-grid-layer template="vertical">
    <h3 style="padding:16px; line-height:1.25">{paragraph}</h3>
  </amp-story-grid-layer>
</amp-story-page>
"""

# =========================
# Streamlit UI
# =========================
st.set_page_config(page_title="🧠 Web Story Content Generator", page_icon="📰", layout="wide")
st.title("📰 Suvichaar — Web Story Content Generator")

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["Step 1", "Step 2", "Step 3", "AMP Builder", "Publish", "Cover Image"])

# -------------------- Tab 1 --------------------
with tab1:
    st.subheader("Generalized Web Story Prompt Generator (No Polaris)")
    url = st.text_input("Enter a news article URL")
    persona = st.selectbox("Choose audience persona:", ["genz", "millenial", "working professionals", "creative thinkers", "spiritual explorers"])
    content_language = st.selectbox("Choose content language", ["English", "Hindi"])
    number = st.number_input("Enter # of main slides (excluding Headline & Hookline)", min_value=0, max_value=1000, value=6, step=1)
    st.caption("Total slides = Headline (1) + main slides + Hookline (1). Enforced between 8 and 10 total. CTA is extra.")

    if st.button("🚀 Submit and Generate JSON"):
        if url and persona:
            with st.spinner("Analyzing article and generating outline..."):
                try:
                    title, summary, full_text = extract_article(url)
                    sentiment = get_sentiment(summary or full_text)
                    result = detect_category_and_subcategory(full_text, content_language)
                    category, subcategory, emotion = result["category"], result["subcategory"], result["emotion"]

                    # clamp main slides based on total min/max
                    main_count = clamp_main_slides(number)

                    # Generate outline
                    outline = title_script_generator(category, subcategory, emotion, full_text, content_language)

                    # Build the flattened story structure
                    storytitle = generate_storytitle(title, summary, content_language)
                    hookline   = generate_hookline(title, summary, content_language)

                    structured = OrderedDict()
                    structured["storytitle"] = storytitle
                    # s1..sN (main slides)
                    structured.update(restructure_slide_output(outline, main_count))
                    structured["hookline"] = hookline

                    if content_language == "Hindi":
                        structured = transliterate_to_devanagari(structured)

                    # Save + offer download
                    ts = int(time.time())
                    fname = f"structured_slides_{ts}.json"
                    with open(fname, "w", encoding="utf-8") as f:
                        json.dump(structured, f, indent=2, ensure_ascii=False)

                    with open(fname, "r", encoding="utf-8") as f:
                        st.success("✅ Prompt JSON ready")
                        st.download_button("⬇️ Download JSON", f.read(), file_name=fname, mime="application/json")

                except Exception as e:
                    st.error(f"❌ Error: {e}")
        else:
            st.warning("Please enter a URL and choose a persona.")

# -------------------- Tab 2 --------------------
with tab2:
    st.subheader("🎙️ Text-to-Speech → S3")
    uploaded_file = st.file_uploader("Upload structured slide JSON", type=["json"])
    voice_label = st.selectbox("Choose Voice", list(voice_options.values()))

    if uploaded_file and voice_label:
        paragraphs = json.load(uploaded_file)
        st.success(f"✅ Loaded {len(paragraphs)} fields")

        if st.button("🚀 Generate TTS + Upload to S3"):
            with st.spinner("Synthesizing & uploading..."):
                output = synthesize_and_upload(paragraphs, voice_label, add_cta=True)
                st.success("✅ Uploaded to S3")

                ts = int(time.time())
                out_name = f"tts_output_{ts}.json"
                with open(out_name, "w", encoding="utf-8") as f:
                    json.dump(output, f, indent=2, ensure_ascii=False)

                # fixed image for remotion/cover flows (configurable)
                fixed_image_url = DEFAULT_SLIDE_IMAGE_URL

                # Offer download
                with open(out_name, "r", encoding="utf-8") as f:
                    st.download_button("⬇️ Download TTS Output", f.read(), file_name=out_name, mime="application/json")

# -------------------- Tab 3 --------------------
with tab3:
    st.subheader("🧩 Save modified file (HTML + JSON Zip)")
    up_json = st.file_uploader("📤 Upload Full Slide JSON (with slide1..)", type=["json"])
    up_html = st.file_uploader("📄 Upload HTML template with placeholders", type=["html"])

    def replace_placeholders_in_html(html_text, json_data):
        storytitle = json_data.get("slide1", {}).get("storytitle", "")
        storytitle_url = json_data.get("slide1", {}).get("audio_url", "")
        hookline = ""
        hookline_url = ""
        # find last slide that has hookline
        for k in sorted(json_data.keys(), key=lambda x: int(x.replace("slide",""))):
            if "hookline" in json_data[k]:
                hookline = json_data[k]["hookline"]
                hookline_url = json_data[k].get("audio_url","")
        html_text = html_text.replace("{{storytitle}}", storytitle)
        html_text = html_text.replace("{{storytitle_audiourl}}", storytitle_url)
        html_text = html_text.replace("{{hookline}}", hookline)
        html_text = html_text.replace("{{hookline_audiourl}}", hookline_url)
        return html_text

    def modify_tab4_json(original_json):
        """Trim & re-map to a compact sX structure if needed (kept for backwards compat)."""
        updated_json = OrderedDict()
        slide_number = 2
        # start from slide3 to build a compact view
        for i in range(3, 100):
            old_key = f"slide{i}"
            if old_key not in original_json:
                break
            content = original_json[old_key]
            new_key = f"slide{slide_number}"
            for k, v in content.items():
                if k.endswith("paragraph1"):
                    para_key = f"s{slide_number}paragraph1"
                    audio_key = f"audio_url{slide_number}"
                    updated_json[new_key] = {
                        para_key: v,
                        audio_key: content.get("audio_url",""),
                        "voice": content.get("voice","")
                    }
                    break
            slide_number += 1
        return updated_json

    if up_json and up_html:
        json_data = json.load(up_json)
        html_template = up_html.read().decode("utf-8")

        updated_html = replace_placeholders_in_html(html_template, json_data)
        updated_json = modify_tab4_json(json_data)

        if st.button("🎯 Generate Final HTML + Trimmed JSON (ZIP)"):
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            zip_filename = f"Output_bundle_{ts}.zip"
            buf = io.BytesIO()
            with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
                zf.writestr(f"updated_{ts}.html", updated_html)
                zf.writestr(f"output_{ts}.json", json.dumps(updated_json, indent=2, ensure_ascii=False))
            buf.seek(0)
            st.download_button("⬇️ Download ZIP", buf, file_name=zip_filename, mime="application/zip")

# -------------------- Tab 4 (AMP Builder) --------------------
with tab4:
    st.subheader("🎞️ AMP Web Story Builder (Audio + Animation)")
    uploaded_html_file = st.file_uploader("📄 Upload AMP Template HTML (must contain <!--INSERT_SLIDES_HERE-->)", type=["html"], key="html_upload_tab3")
    uploaded_json_file = st.file_uploader("📦 Upload TTS Output JSON (from Step 2)", type=["json"], key="json_upload_tab3")

    if uploaded_html_file and uploaded_json_file:
        try:
            template_html = uploaded_html_file.read().decode("utf-8")
            output_data = json.load(uploaded_json_file)

            if "<!--INSERT_SLIDES_HERE-->" not in template_html:
                st.error("❌ Placeholder <!--INSERT_SLIDES_HERE--> not found in uploaded HTML.")
            else:
                # build slides in numeric order
                all_slides = ""
                keys_sorted = sorted(output_data.keys(), key=lambda x: int(x.replace("slide","")))
                for k in keys_sorted:
                    data = output_data[k]
                    # resolve paragraph field name
                    para = None
                    # storytitle
                    if "storytitle" in data:
                        para = data["storytitle"]
                    # hookline
                    elif "hookline" in data:
                        para = data["hookline"]
                    else:
                        # sXparagraph1
                        for kk in data.keys():
                            if kk.endswith("paragraph1"):
                                para = data[kk]
                                break
                    audio_url = data.get("audio_url", "")
                    if para and audio_url:
                        raw = str(para).replace("’", "'").replace('"', '&quot;')
                        paragraph = textwrap.shorten(raw, width=180, placeholder="...")
                        all_slides += generate_amp_slide(paragraph, audio_url)

                final_html = template_html.replace("<!--INSERT_SLIDES_HERE-->", all_slides)
                filename = f"pre-final_amp_story_{int(time.time())}.html"

                st.success("✅ Final AMP HTML generated!")
                st.download_button("📥 Download Final AMP HTML", final_html, file_name=filename, mime="text/html")

        except Exception as e:
            st.error(f"⚠️ Error: {e}")

# -------------------- Tab 5 (Publish) --------------------
with tab5:
    st.header("Publish to S3 (HTML + Metadata)")
    if "last_title" not in st.session_state:
        st.session_state.last_title = ""
        st.session_state.meta_description = ""
        st.session_state.meta_keywords = ""

    story_title = st.text_input("Story Title")
    # Auto meta (optional)
    if story_title.strip() and story_title != st.session_state.last_title:
        with st.spinner("Generating meta description/keywords/tags..."):
            messages = [{
                "role": "user",
                "content": f"Generate: 1) short SEO meta description 2) meta keywords (comma) 3) filter tags (comma) for '{story_title}'"
            }]
            try:
                r = client.chat.completions.create(
                    model="gpt-5-chat", messages=messages, max_tokens=300, temperature=0.5
                )
                out = r.choices[0].message.content or ""
                desc = re.search(r"[Dd]escription\s*[:\-]\s*(.+)", out)
                keys = re.search(r"[Kk]eywords\s*[:\-]\s*(.+)", out)
                tags = re.search(r"[Ff]ilter\s*[Tt]ags\s*[:\-]\s*(.+)", out)
                st.session_state.meta_description = desc.group(1).strip() if desc else ""
                st.session_state.meta_keywords = keys.group(1).strip() if keys else ""
                st.session_state.generated_filter_tags = tags.group(1).strip() if tags else ""
            except Exception as e:
                st.warning(f"Meta gen error: {e}")
            st.session_state.last_title = story_title

    meta_description = st.text_area("Meta Description", value=st.session_state.meta_description)
    meta_keywords    = st.text_input("Meta Keywords", value=st.session_state.meta_keywords)
    content_type = st.selectbox("Content type", ["News", "Article"])
    language     = st.selectbox("Language", ["en-US", "hi"])
    image_url    = st.text_input("Cover Image URL", value=DEFAULT_COVER_URL)
    uploaded_prefinal = st.file_uploader("💾 Upload pre-final AMP HTML", type=["html","htm"], key="prefinal_upload")

    categories = st.selectbox("Categories", ["Art","Travel","Entertainment","Literature","Books","Sports","History","Culture","Wildlife","Spiritual"])
    default_tags = ["News","Breaking","Update","Suvichaar Stories"]
    tag_input = st.text_input("Filter Tags (comma)", value=st.session_state.get("generated_filter_tags", ", ".join(default_tags)))
    use_custom_cover = st.radio("Custom cover image URL?", ("No","Yes"))
    cover_image_url = st.text_input("Custom Cover Image URL") if use_custom_cover == "Yes" else image_url

    with st.form("content_form"):
        submit_button = st.form_submit_button("Submit")

    if submit_button:
        missing = []
        if not story_title.strip(): missing.append("Story Title")
        if not meta_description.strip(): missing.append("Meta Description")
        if not meta_keywords.strip(): missing.append("Meta Keywords")
        if not content_type.strip(): missing.append("Content Type")
        if not language.strip(): missing.append("Language")
        if not image_url.strip(): missing.append("Image URL")
        if not tag_input.strip(): missing.append("Filter Tags")
        if not categories.strip(): missing.append("Category")
        if not uploaded_prefinal: missing.append("pre-final AMP HTML")
        if missing:
            st.error("❌ Please fill required fields:\n- " + "\n- ".join(missing))
        else:
            st.markdown("### Submitted Data")
            st.write(f"**Story Title:** {story_title}")
            st.write(f"**Meta Description:** {meta_description}")
            st.write(f"**Meta Keywords:** {meta_keywords}")
            st.write(f"**Content Type:** {content_type}")
            st.write(f"**Language:** {language}")

            try:
                nano, slug_nano, canurl, canurl1 = generate_slug_and_urls(story_title)
                page_title = f"{story_title} | Suvichaar"
            except Exception as e:
                st.error(f"Canonical URL error: {e}")
                nano = slug_nano = canurl = canurl1 = page_title = ""

            # Upload image if not already on CDN
            key_path = "media/default.png"
            uploaded_url = ""
            try:
                if image_url.startswith("https://media.suvichaar.org/") or image_url.startswith("http://media.suvichaar.org/"):
                    uploaded_url = image_url
                    key_path = urlparse(image_url).path.lstrip("/")
                else:
                    resp = requests.get(image_url, timeout=10)
                    resp.raise_for_status()
                    ext = os.path.splitext(urlparse(image_url).path)[1].lower() or ".jpg"
                    unique_filename = f"{uuid.uuid4().hex}{ext}"
                    s3_key = f"{S3_PREFIX}{unique_filename}"
                    s3_client.put_object(Bucket=AWS_BUCKET, Key=s3_key, Body=resp.content, ContentType=resp.headers.get("Content-Type","image/jpeg"))
                    uploaded_url = f"{CDN_BASE}{s3_key}"
                    key_path = s3_key
                    st.success("Image uploaded to CDN!")
            except Exception as e:
                st.warning(f"Cover upload failed, using provided URL. Error: {e}")
                uploaded_url = image_url

            # Build responsive variants via CloudFront lambda@edge encoding
            parsed_path = key_path
            resize_presets = {
                "potraitcoverurl": (640, 853),
                "msthumbnailcoverurl": (300, 300),
            }
            def encode_resize(w,h):
                template = {"bucket": AWS_BUCKET, "key": parsed_path, "edits":{"resize":{"width":w,"height":h,"fit":"cover"}}}
                return f"{CDN_PREFIX_MEDIA}{base64.urlsafe_b64encode(json.dumps(template).encode()).decode()}"

            # Use uploaded HTML as template
            try:
                html_template = uploaded_prefinal.read().decode("utf-8")
                html_template = html_template.replace("{{publishedtime}}", datetime.now(timezone.utc).isoformat(timespec='seconds'))
                html_template = html_template.replace("{{modifiedtime}}", datetime.now(timezone.utc).isoformat(timespec='seconds'))
                html_template = html_template.replace("{{storytitle}}", story_title)
                html_template = html_template.replace("{{metadescription}}", meta_description)
                html_template = html_template.replace("{{metakeywords}}", meta_keywords)
                html_template = html_template.replace("{{contenttype}}", content_type)
                html_template = html_template.replace("{{lang}}", language)
                html_template = html_template.replace("{{pagetitle}}", page_title)
                html_template = html_template.replace("{{canurl}}", canurl)
                html_template = html_template.replace("{{canurl1}}", canurl1)
                html_template = html_template.replace("{{image0}}", uploaded_url)
                html_template = html_template.replace("{{potraitcoverurl}}", encode_resize(640,853))
                html_template = html_template.replace("{{msthumbnailcoverurl}}", encode_resize(300,300))
                # cleanup {url}
                html_template = re.sub(r'href="\{(https://[^}]+)\}"', r'href="\\1"', html_template)
                html_template = re.sub(r'src="\{(https://[^}]+)\}"', r'src="\\1"', html_template)

                st.markdown("### Final Modified HTML")
                st.code(html_template[:10000], language="html")  # preview head

                # Metadata JSON
                category_mapping = {"Art":1,"Travel":2,"Entertainment":3,"Literature":4,"Books":5,"Sports":6,"History":7,"Culture":8,"Wildlife":9,"Spiritual":10}
                filternumber = category_mapping[categories]
                filter_tags = [t.strip() for t in tag_input.split(",") if t.strip()]
                metadata_dict = {
                    "story_title": story_title,
                    "categories": filternumber,
                    "filterTags": filter_tags,
                    "story_uid": nano,
                    "story_link": canurl,
                    "storyhtmlurl": canurl1,
                    "urlslug": slug_nano,
                    "cover_image_link": cover_image_url or uploaded_url,
                    "publisher_id": 1,
                    "story_logo_link": "https://media.suvichaar.org/filters:resize/96x96/media/brandasset/suvichaariconblack.png",
                    "keywords": meta_keywords,
                    "metadescription": meta_description,
                    "lang": language,
                }

                # Upload HTML to site bucket
                site_bucket = "suvichaarstories"
                s3_client.put_object(Bucket=site_bucket, Key=f"{slug_nano}.html", Body=html_template.encode("utf-8"), ContentType="text/html")
                final_story_url = f"https://suvichaar.org/stories/{slug_nano}"
                st.success("✅ HTML uploaded to S3")
                st.markdown(f"🔗 **Live Story URL:** [{final_story_url}]({final_story_url})")

                # ZIP for download
                zip_buffer = io.BytesIO()
                with zipfile.ZipFile(zip_buffer, "w") as zf:
                    zf.writestr(f"{slug_nano}.html", html_template)
                    zf.writestr(f"{slug_nano}_metadata.json", json.dumps(metadata_dict, indent=4))
                zip_buffer.seek(0)
                st.download_button("📦 Download HTML + Metadata ZIP", data=zip_buffer, file_name=f"{story_title}.zip", mime="application/zip")

            except Exception as e:
                st.error(f"Error processing HTML: {e}")

# -------------------- Tab 6 (Cover Image) --------------------
with tab6:
    st.title("Cover Image Request")
    s3 = boto3.client("s3", aws_access_key_id=AWS_ACCESS_KEY, aws_secret_access_key=AWS_SECRET_KEY, region_name=AWS_REGION)

    uploaded = st.file_uploader("📥 Upload Suvichaar TTS JSON", type=["json"])
    if not uploaded:
        st.info("Please upload TTS JSON to begin.")
        st.stop()

    try:
        data = json.load(uploaded)
        transformed = {}
        for slide_key, info in data.items():
            idx = int(slide_key.replace("slide", ""))
            # derive text
            if "storytitle" in info:
                text = info["storytitle"]
            elif "hookline" in info:
                text = info["hookline"]
            else:
                text = next((v for k, v in info.items() if "paragraph1" in k), "")
            audio = info.get("audio_url", "")
            transformed[slide_key] = {
                f"s{idx}paragraph1": text,
                f"s{idx}audio1": audio,
                f"s{idx}image1": DEFAULT_COVER_URL,
                f"s{idx}paragraph2": "Suvichaar",
            }
        st.success("✅ Transformation Complete")
        st.json(transformed)
    except Exception as e:
        st.error(f"❌ Error during transformation: {e}")
        st.stop()

    if st.button("Generate Thumbnail"):
        with st.spinner("Generating…"):
            try:
                resp = requests.post("https://remotion.suvichaar.org/api/generate-news-thumbnail", json=transformed, timeout=30)
                resp.raise_for_status()
            except requests.RequestException as err:
                st.error(f"Thumbnail API error: {err}")
                st.stop()

        img_bytes = resp.content
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        key = f"{S3_PREFIX}cover_{ts}.png"
        try:
            s3.put_object(Bucket=AWS_BUCKET, Key=key, Body=img_bytes, ContentType=resp.headers.get("Content-Type","image/png"))
        except Exception as s3_err:
            st.error(f"S3 upload failed: {s3_err}")
            st.stop()

        cdn_url = f"{CDN_PREFIX_MEDIA}{key}"
        st.success("🖼️ Thumbnail generated and uploaded!")
        st.markdown(f"[View on CDN]({cdn_url})")
        st.image(cdn_url, use_column_width=True)

        st.download_button(
            "⬇️ Download Transformed JSON",
            data=json.dumps(transformed, indent=2, ensure_ascii=False),
            file_name=f"CoverJSON_{ts}.json",
            mime="application/json",
        )
