# =========================
# 🔧 CONFIG: Azure + AWS
# =========================
import os, re, io, json, uuid, time, base64, random, string, textwrap
from pathlib import Path
from datetime import datetime, timezone
from collections import OrderedDict

import requests, boto3, nltk, zipfile
import streamlit as st
from dotenv import load_dotenv
from openai import AzureOpenAI
from urllib.parse import urlparse

load_dotenv()

# ── Secrets you must set in .streamlit/secrets.toml ─────────────────────
# [azure_openai]
# AZURE_API_KEY = "..."
# AZURE_ENDPOINT = "https://<your-openai>.cognitiveservices.azure.com"
# AZURE_DEPLOYMENT = "gpt-5-chat"
# AZURE_API_VERSION = "2025-01-01-preview"
#
# [azure_speech]
# AZURE_SPEECH_KEY = "..."
# AZURE_SPEECH_REGION = "eastus"
# VOICE_NAME = "hi-IN-AaravNeural"
#
# [azure]
# AZURE_TTS_URL = "https://tts.suvichaar.org/api/speak"
#
# [aws]
# AWS_ACCESS_KEY = "..."
# AWS_SECRET_KEY = "..."
# AWS_REGION     = "ap-south-1"
# AWS_BUCKET     = "suvichaarapp"
# S3_PREFIX      = "media/"
# CDN_BASE       = "https://media.suvichaar.org/"

# --- Azure OpenAI (from secrets) ---
AZURE_API_KEY     = st.secrets["azure_openai"]["AZURE_API_KEY"]
AZURE_ENDPOINT    = st.secrets["azure_openai"]["AZURE_ENDPOINT"]
AZURE_DEPLOYMENT  = st.secrets["azure_openai"]["AZURE_DEPLOYMENT"]
AZURE_API_VERSION = st.secrets["azure_openai"]["AZURE_API_VERSION"]

# --- Azure Speech (from secrets) ---
AZURE_SPEECH_KEY    = st.secrets["azure_speech"]["AZURE_SPEECH_KEY"]
AZURE_SPEECH_REGION = st.secrets["azure_speech"]["AZURE_SPEECH_REGION"]
DEFAULT_VOICE       = st.secrets["azure_speech"].get("VOICE_NAME", "en-IN-AaravNeural")

# --- Custom TTS microservice (optional) ---
AZURE_TTS_URL = st.secrets.get("azure", {}).get("AZURE_TTS_URL", "https://tts.suvichaar.org/api/speak")

# --- AWS (from secrets) ---
AWS_ACCESS_KEY = st.secrets["aws"]["AWS_ACCESS_KEY"]
AWS_SECRET_KEY = st.secrets["aws"]["AWS_SECRET_KEY"]
AWS_REGION     = st.secrets["aws"]["AWS_REGION"]
AWS_BUCKET     = st.secrets["aws"]["AWS_BUCKET"]
S3_PREFIX      = st.secrets["aws"].get("S3_PREFIX", "media/")
CDN_BASE       = st.secrets["aws"]["CDN_BASE"]
CDN_PREFIX_MEDIA = "https://media.suvichaar.org/"

s3_client = boto3.client(
    "s3",
    aws_access_key_id=AWS_ACCESS_KEY,
    aws_secret_access_key=AWS_SECRET_KEY,
    region_name=AWS_REGION,
)

# --- Azure OpenAI Client ---
client = AzureOpenAI(
    azure_endpoint=AZURE_ENDPOINT,
    api_key=AZURE_API_KEY,
    api_version=AZURE_API_VERSION
)

# =========================
# 🔊 Voice selection
# =========================
def pick_voice_for_language(lang_code: str, default_voice: str = DEFAULT_VOICE) -> str:
    """Map detected language → Azure voice name."""
    if not lang_code:
        return default_voice
    l = lang_code.lower()
    if l.startswith("hi"):
        return "hi-IN-AaravNeural"
    if l.startswith("en-in"):
        return "en-IN-NeerjaNeural"
    if l.startswith("en"):
        return "en-IN-AaravNeural"
    if l.startswith("bn"):
        return "bn-IN-BashkarNeural"
    if l.startswith("ta"):
        return "ta-IN-PallaviNeural"
    if l.startswith("te"):
        return "te-IN-ShrutiNeural"
    if l.startswith("mr"):
        return "mr-IN-AarohiNeural"
    if l.startswith("gu"):
        return "gu-IN-DhwaniNeural"
    if l.startswith("kn"):
        return "kn-IN-SapnaNeural"
    if l.startswith("pa"):
        return "pa-IN-GeetikaNeural"
    return default_voice

# =========================
# 🧱 Helpers
# =========================
def generate_slug_and_urls(title):
    if not title or not isinstance(title, str):
        raise ValueError("Invalid title")
    slug = ''.join(c for c in title.lower().replace(" ", "-").replace("_", "-")
                   if c in string.ascii_lowercase + string.digits + '-').strip('-')
    nano = ''.join(random.choices(string.ascii_letters + string.digits + '_-', k=10)) + '_G'
    slug_nano = f"{slug}_{nano}"
    return nano, slug_nano, f"https://suvichaar.org/stories/{slug_nano}", f"https://stories.suvichaar.org/{slug_nano}.html"

def extract_article(url):
    import newspaper
    from newspaper import Article
    try:
        article = Article(url)
        article.download(); article.parse()
        try:
            article.nlp()
        except:
            pass
        title   = (article.title or "Untitled Article").strip()
        text    = (article.text  or "No article content available.").strip()
        summary = (article.summary or text[:300]).strip()
        return title, summary, text
    except Exception as e:
        st.error(f"❌ Failed to extract article from URL. Error: {e}")
        return "Untitled Article", "No summary available.", "No article content available."

def detect_category_and_subcategory(text, content_language="English"):
    prompt = f"""
Analyze the following news article and return:
1. category
2. subcategory
3. emotion

Article:
\"\"\"{text[:3000]}\"\"\"
Return ONLY JSON with keys category, subcategory, emotion.
"""
    try:
        resp = client.chat.completions.create(
            model=AZURE_DEPLOYMENT,
            messages=[
                {"role":"system","content":"Classify category, subcategory, and emotion."},
                {"role":"user","content":prompt.strip()},
            ],
            max_tokens=150
        )
        content = resp.choices[0].message.content.strip().strip("```").strip("json").strip()
        data = json.loads(content)
        if all(k in data for k in ["category","subcategory","emotion"]):
            return data
    except Exception:
        pass
    return {"category":"Unknown","subcategory":"General","emotion":"Neutral"}

def get_sentiment(text):
    from textblob import TextBlob
    if not text or not text.strip(): return "neutral"
    pol = TextBlob(text.strip().replace("\n"," ")).sentiment.polarity
    return "positive" if pol>0.2 else "negative" if pol<-0.2 else "neutral"

# =========================
# 🖋️ Slide generation — no Polaris
# =========================
MIN_SLIDES = 8
MAX_SLIDES = 10

def make_connected_point(headline: str, summary: str, lang: str) -> str:
    """Slide 2 helper: one connected line to the news (no hashtags/emojis)."""
    if (lang or "").lower().startswith("hi"):
        up = f"शीर्षक: {headline}\nसारांश: {summary}\n\nएक वाक्य में, सरल हिंदी में, हेडलाइन से जुड़ा मुख्य बिंदु लिखें (120 वर्णों से कम)।"
    else:
        up = f"Headline: {headline}\nSummary: {summary}\n\nIn one short sentence (<120 chars), write a key connected point to the headline."
    try:
        r = client.chat.completions.create(
            model=AZURE_DEPLOYMENT,
            messages=[
                {"role":"system","content":"Write a single concise line, no emojis/hashtags."},
                {"role":"user","content": up}
            ],
            max_tokens=80, temperature=0.3
        )
        return r.choices[0].message.content.strip().strip('"')
    except Exception:
        return (summary or headline)[:110]

def split_article_into_chunks(article_text: str, desired_count: int, lang: str) -> list:
    """Simple chunking to fill slides 3..N-1. (Short ≈160 chars)"""
    text = (article_text or "").strip()
    if not text:
        return [""] * max(desired_count, 0)
    paras = [p.strip() for p in text.split("\n") if p.strip()]
    chunks, i = [], 0
    while len(chunks) < desired_count:
        part = paras[i % len(paras)]
        if len(part) > 160:
            part = textwrap.shorten(part, width=160, placeholder="…")
        chunks.append(part)
        i += 1
    return chunks

def build_story_struct(title, summary, article_text, content_language: str, total_slides: int):
    """
    Enforce:
      1 = Headline
      2 = Connected point
    3..(N-1) = Article points
      N = Fixed Hookline (CTA)
    """
    total_slides = max(MIN_SLIDES, min(MAX_SLIDES, total_slides))
    slides = []

    # Slide 1: Headline
    slides.append({
        "title": title[:80],
        "script": title.strip()
    })

    # Slide 2: Connected point
    slides.append({
        "title": "Key Update" if not content_language.lower().startswith("hi") else "मुख्य बिंदु",
        "script": make_connected_point(title, summary, content_language)
    })

    # Slides 3..N-1: points
    middle_needed = total_slides - 3  # because last slide is CTA
    middle_scripts = split_article_into_chunks(article_text, middle_needed, content_language)
    for s in middle_scripts:
        slides.append({ "title": "", "script": s })

    # Slide N: Fixed Hookline (CTA)
    slides.append({
        "title": "Stay Connected" if not content_language.lower().startswith("hi") else "जुड़े रहें",
        "script": "For Such Content Stay Connected with Suvichar Live\n\nRead|Share|Inspire"
    })

    return slides

def restructure_slide_output_for_tts(slides: list) -> OrderedDict:
    """
    Flatten slides → {"storytitle":..., "s1paragraph1":..., ..., "hookline": ...}
    where last slide text is Hookline/CTA.
    """
    out = OrderedDict()
    if not slides:
        return out

    out["storytitle"] = slides[0]["script"].strip()
    middle = slides[1:-1]
    for idx, s in enumerate(middle, start=1):
        out[f"s{idx}paragraph1"] = (s.get("script","") or "").strip()
    out["hookline"] = (slides[-1].get("script","") or "").strip()
    return out

# =========================
# 🔉 TTS uploader (uses your microservice)
# =========================
def synthesize_and_upload(paragraphs: dict, voice_name: str):
    """
    paragraphs structure:
      storytitle (string)
      s1paragraph1 ... sNparagraph1
      hookline
    POST → AZURE_TTS_URL : {"model":"tts-1-hd","input":<text>,"voice":<voice_name>}
    """
    result = OrderedDict()
    os.makedirs("temp", exist_ok=True)
    slide_index = 1

    # Prefer Speech key if your TTS service validates against it; otherwise fallback to OpenAI key.
    api_key_for_tts = AZURE_SPEECH_KEY or AZURE_API_KEY

    def _speak_and_upload(text: str, voice: str) -> str:
        resp = requests.post(
            AZURE_TTS_URL,
            headers={"Content-Type": "application/json", "api-key": api_key_for_tts},
            json={"model":"tts-1-hd","input":text,"voice":voice},
            timeout=60
        )
        resp.raise_for_status()
        filename = f"tts_{uuid.uuid4().hex}.mp3"
        local_path = os.path.join("temp", filename)
        with open(local_path, "wb") as f:
            f.write(resp.content)
        s3_key = f"{S3_PREFIX}{filename}"
        s3_client.upload_file(local_path, AWS_BUCKET, s3_key)
        os.remove(local_path)
        return f"{CDN_BASE}{s3_key}"

    # Slide 1: storytitle
    if "storytitle" in paragraphs:
        url = _speak_and_upload(paragraphs["storytitle"], voice_name)
        result[f"slide{slide_index}"] = {
            "storytitle": paragraphs["storytitle"], "audio_url": url, "voice": voice_name
        }
        slide_index += 1

    # Slide 2..N-1: s1..sK
    i = 1
    while f"s{i}paragraph1" in paragraphs:
        text = paragraphs[f"s{i}paragraph1"]
        url  = _speak_and_upload(text, voice_name)
        result[f"slide{slide_index}"] = {
            f"s{slide_index}paragraph1": text, f"audio_url{slide_index}": url, "voice": voice_name
        }
        slide_index += 1
        i += 1

    # Last: hookline
    if "hookline" in paragraphs and paragraphs["hookline"].strip():
        url = _speak_and_upload(paragraphs["hookline"], voice_name)
        result[f"slide{slide_index}"] = {
            f"s{slide_index}paragraph1": paragraphs["hookline"], f"audio_url{slide_index}": url, "voice": voice_name
        }

    return result

# =========================
# 🎬 Remotion input (CTA updated, neutral images)
# =========================
def generate_remotion_input(tts_output: dict, fixed_image_url: str, author_name: str = "Suvichaar"):
    remotion_data = OrderedDict()
    slide_index = 1

    # Slide 1: storytitle (if available)
    if "slide1" in tts_output and "storytitle" in tts_output["slide1"]:
        remotion_data[f"slide{slide_index}"] = {
            f"s{slide_index}paragraph1": tts_output["slide1"]["storytitle"],
            f"s{slide_index}audio1":     tts_output["slide1"].get("audio_url",""),
            f"s{slide_index}image1":     fixed_image_url,
            f"s{slide_index}paragraph2": f"- {author_name}"
        }
        slide_index += 1

    # The rest in order
    max_idx = max(int(k.replace("slide","")) for k in tts_output.keys())
    for i in range(2, max_idx+1):
        data = tts_output.get(f"slide{i}", {})
        para_val = ""
        for k,v in data.items():
            if "paragraph1" in k:
                para_val = v; break
        remotion_data[f"slide{slide_index}"] = {
            f"s{slide_index}paragraph1": para_val,
            f"s{slide_index}audio1":     data.get(f"audio_url{i}", data.get("audio_url","")),
            f"s{slide_index}image1":     fixed_image_url,
            f"s{slide_index}paragraph2": f"- {author_name}"
        }
        slide_index += 1

    # Save to file
    ts = int(time.time())
    filename = f"remotion_input_{ts}.json"
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(remotion_data, f, indent=2, ensure_ascii=False)
    return filename

# =========================
# 🔧 Tab helpers used later
# =========================
def modify_tab4_json(original_json):
    """Trim slide3.. to continuous slide2.., preserving audio."""
    updated_json = OrderedDict()
    slide_number = 2
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
                    audio_key: content.get("audio_url", "") or content.get(f"audio_url{i}", ""),
                    "voice": content.get("voice", "")
                }
                break
        slide_number += 1
    return updated_json

def replace_placeholders_in_html(html_text, json_data):
    storytitle = json_data.get("slide1", {}).get("storytitle", "")
    storytitle_url = json_data.get("slide1", {}).get("audio_url", "")
    hookline = json_data.get("slide2", {}).get("hookline", "")
    hookline_url = json_data.get("slide2", {}).get("audio_url", "")
    html_text = html_text.replace("{{storytitle}}", storytitle)
    html_text = html_text.replace("{{storytitle_audiourl}}", storytitle_url)
    html_text = html_text.replace("{{hookline}}", hookline)
    html_text = html_text.replace("{{hookline_audiourl}}", hookline_url)
    return html_text

# =========================
# 🧠 UI
# =========================
st.title("🧠 Web Story Content Generator")

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["Step:1", "Step:2", "Step:3","Step:4","Step:5","Step:6"])

# -------------------------
# Tab 1 — Build slides JSON
# -------------------------
with tab1:
    st.title("🧠 Generalized Web Story Prompt Generator")
    url = st.text_input("Enter a news article URL")
    content_language = st.selectbox("Choose content language", ["English (en)", "Hindi (hi)"])
    requested = st.number_input("Slides (min 8, max 10)", min_value=8, max_value=10, value=8, step=1)

    if st.button("🚀 Submit and Generate JSON"):
        if url:
            with st.spinner("Analyzing the article and generating slides..."):
                try:
                    title, summary, full_text = extract_article(url)
                    sentiment = get_sentiment(summary or full_text)
                    result = detect_category_and_subcategory(full_text, content_language)
                    _ = (result["category"], result["subcategory"], result["emotion"])  # not shown but kept

                    slides = build_story_struct(
                        title=title,
                        summary=summary,
                        article_text=full_text,
                        content_language="hi" if "hi" in content_language.lower() else "en",
                        total_slides=requested
                    )
                    structured = restructure_slide_output_for_tts(slides)

                    ts = int(time.time())
                    filename = f"structured_slides_{ts}.json"
                    with open(filename, "w", encoding="utf-8") as f:
                        json.dump(structured, f, indent=2, ensure_ascii=False)

                    with open(filename, "r", encoding="utf-8") as f:
                        st.success("✅ JSON ready. Download below:")
                        st.download_button(
                            label=f"⬇️ Download JSON ({ts})",
                            data=f.read(),
                            file_name=filename,
                            mime="application/json"
                        )
                except Exception as e:
                    st.error(f"❌ Error: {e}")
        else:
            st.warning("Please enter a valid URL.")

# -------------------------
# Tab 2 — TTS + S3
# -------------------------
with tab2:
    st.title("🎙️ Text-to-Speech → S3")
    uploaded_file = st.file_uploader("Upload structured slide JSON", type=["json"])

    voice_lang = st.selectbox("Narration Language", ["hi-IN", "en-IN", "en-US", "bn-IN", "ta-IN", "te-IN", "mr-IN", "gu-IN", "kn-IN", "pa-IN"])
    chosen_voice = pick_voice_for_language(voice_lang)

    if uploaded_file:
        paragraphs = json.load(uploaded_file)
        st.success(f"✅ Loaded {len(paragraphs)} items")

        if st.button("🚀 Generate TTS + Upload to S3"):
            with st.spinner("Synthesizing & uploading..."):
                tts_out = synthesize_and_upload(paragraphs, chosen_voice)

                ts = int(time.time())
                out_name = f"tts_output_{ts}.json"
                with open(out_name, "w", encoding="utf-8") as f:
                    json.dump(tts_out, f, indent=2, ensure_ascii=False)

                fixed_image_url = "https://media.suvichaar.org/upload/covers/default-cover.png"
                _remotion_file = generate_remotion_input(tts_out, fixed_image_url, author_name="Suvichaar")

                with open(out_name, "r", encoding="utf-8") as f:
                    st.download_button("⬇️ Download TTS JSON", data=f.read(), file_name=out_name, mime="application/json")

# -------------------------
# Tab 3 — Save modified file (ZIP)
# -------------------------
with tab3:
    st.title("🧩 Saving modified file")
    uploaded_file_tab3 = st.file_uploader("📤 Upload Full Slide JSON (with slide1..)", type=["json"], key="tab3_upl")

    if uploaded_file_tab3:
        json_data = json.load(uploaded_file_tab3)
        st.success("✅ JSON Loaded")
        try:
            with open("test.html", "r", encoding="utf-8") as f:
                html_template = f.read()
        except FileNotFoundError:
            st.error("❌ Could not find `test.html`. Please make sure it exists.")
        else:
            updated_html = replace_placeholders_in_html(html_template, json_data)
            updated_json = modify_tab4_json(json_data)

            if st.button("🎯 Generate Final HTML + Trimmed JSON (ZIP)"):
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                zip_filename = f"Output_bundle_{ts}.zip"

                buffer = io.BytesIO()
                with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zipf:
                    zipf.writestr(f"updated_test_{ts}.html", updated_html)
                    zipf.writestr(f"output_{ts}.json", json.dumps(updated_json, indent=2, ensure_ascii=False))
                buffer.seek(0)

                st.download_button(
                    label="⬇️ Download ZIP with HTML + JSON",
                    data=buffer,
                    file_name=zip_filename,
                    mime="application/zip"
                )

# -------------------------
# Tab 4 — AMP builder
# -------------------------
with tab4:
    st.title("🎞️ AMP Web Story Generator with Full Animation and Audio")
    TEMPLATE_PATH = Path("test.html")

    def generate_slide(paragraph: str, audio_url: str):
        # Neutral default slide image (no Polaris)
        return f"""
        <amp-story-page id="c29cbf94-847a-4bb7-a4eb-47d17d8c2d5a" auto-advance-after="page-c29cbf94-847a-4bb7-a4eb-47d17d8c2d5a-background-audio" class="i-amphtml-layout-container" i-amphtml-layout="container">
            <amp-story-grid-layer template="fill" class="i-amphtml-layout-container" i-amphtml-layout="container">
                <amp-video autoplay layout="fixed" width="1" height="1" poster="" id="page-c29cbf94-847a-4bb7-a4eb-47d17d8c2d5a-background-audio" cache="google" class="i-amphtml-layout-fixed i-amphtml-layout-size-defined" style="width:1px;height:1px" i-amphtml-layout="fixed">
                    <source type="audio/mpeg" src="{audio_url}">
                </amp-video>
            </amp-story-grid-layer>
            <amp-story-grid-layer template="vertical" aspect-ratio="412:618" class="grid-layer i-amphtml-layout-container" i-amphtml-layout="container" style="--aspect-ratio:412/618;">
                <div class="page-fullbleed-area"><div class="page-safe-area">
                    <div class="_c19e533"><div class="_89d52dd mask">
                        <div data-leaf-element="true" class="_8aed44c">
                            <amp-img layout="fill" src="https://media.suvichaar.org/upload/covers/default-slide.png" alt="cover" disable-inline-width="true" class="i-amphtml-layout-fill i-amphtml-layout-size-defined" i-amphtml-layout="fill"></amp-img>
                        </div></div></div>
                    <div class="_3d0c7a9">
                        <div class="_e559378">
                            <div class="_5342a26">
                                <h3 class="_d1a8d0d fill text-wrapper"><span><span class="_14af73e">{paragraph}</span></span></h3>
                            </div>
                        </div>
                    </div>
                </div></div>
            </amp-story-grid-layer>
        </amp-story-page>
        """

    uploaded_html_file = st.file_uploader("📄 Upload AMP Template HTML (with <!--INSERT_SLIDES_HERE-->)", type=["html"], key="html_upload_tab4")
    uploaded_json_file = st.file_uploader("📦 Upload Output JSON", type=["json"], key="json_upload_tab4")

    if uploaded_html_file and uploaded_json_file:
        try:
            template_html = uploaded_html_file.read().decode("utf-8")
            output_data = json.load(uploaded_json_file)

            if "<!--INSERT_SLIDES_HERE-->" not in template_html:
                st.error("❌ Placeholder <!--INSERT_SLIDES_HERE--> not found in uploaded HTML.")
            else:
                all_slides = ""
                for key in sorted(
                    [k for k in output_data.keys() if k.startswith("slide")],
                    key=lambda x: int(x.replace("slide", ""))
                ):
                    slide_num = key.replace("slide", "")
                    data = output_data[key]
                    para_key = f"s{slide_num}paragraph1"
                    audio_key = f"audio_url{slide_num}"

                    # support either audio_url{n} or audio_url
                    audio_url = data.get(audio_key, data.get("audio_url", ""))
                    if para_key in data and audio_url:
                        raw = str(data[para_key]).replace("’", "'").replace('"', '&quot;')
                        paragraph = textwrap.shorten(raw, width=180, placeholder="...")
                        all_slides += generate_slide(paragraph, audio_url)

                final_html = template_html.replace("<!--INSERT_SLIDES_HERE-->", all_slides)
                filename = f"pre-final_amp_story_{int(time.time())}.html"

                st.success("✅ Final AMP HTML generated successfully!")
                st.download_button(
                    label="📥 Download Final AMP HTML",
                    data=final_html,
                    file_name=filename,
                    mime="text/html"
                )

        except Exception as e:
            st.error(f"⚠️ Error: {str(e)}")

# -------------------------
# Tab 5 — Content submission, upload page to S3
# -------------------------
with tab5:
    st.header("Content Submission Form")

    if "last_title" not in st.session_state:
        st.session_state.last_title = ""
        st.session_state.meta_description = ""
        st.session_state.meta_keywords = ""

    story_title = st.text_input("Story Title")

    if story_title.strip() and story_title != st.session_state.last_title:
        with st.spinner("Generating meta description, keywords, and filter tags..."):
            messages = [{
                "role": "user",
                "content": f"""
                Generate the following for a web story titled '{story_title}':
                1. A short SEO-friendly meta description
                2. Meta keywords (comma separated)
                3. Relevant filter tags (comma separated, suitable for categorization and content filtering)
                """
            }]
            try:
                response = client.chat.completions.create(
                    model=AZURE_DEPLOYMENT,
                    messages=messages,
                    max_tokens=300,
                    temperature=0.5,
                )
                output = response.choices[0].message.content
                desc = re.search(r"[Dd]escription\s*[:\-]\s*(.+)", output)
                keys = re.search(r"[Kk]eywords\s*[:\-]\s*(.+)", output)
                tags = re.search(r"[Ff]ilter\s*[Tt]ags\s*[:\-]\s*(.+)", output)

                st.session_state.meta_description = desc.group(1).strip() if desc else ""
                st.session_state.meta_keywords = keys.group(1).strip() if keys else ""
                st.session_state.generated_filter_tags = tags.group(1).strip() if tags else ""
            except Exception as e:
                st.warning(f"Error: {e}")
            st.session_state.last_title = story_title

    meta_description = st.text_area("Meta Description", value=st.session_state.meta_description)
    meta_keywords = st.text_input("Meta Keywords (comma separated)", value=st.session_state.meta_keywords)
    content_type = st.selectbox("Select your contenttype", ["News", "Article"])
    language = st.selectbox("Select your Language", ["en-US", "hi"])
    image_url = st.text_input("Enter your Image URL")
    uploaded_prefinal = st.file_uploader("💾 Upload pre-final AMP HTML (required)", type=["html","htm"], key="prefinal_upload")

    categories = st.selectbox("Select your Categories", ["Art", "Travel", "Entertainment", "Literature", "Books", "Sports", "History", "Culture", "Wildlife", "Spiritual", "Food"])

    default_tags = [
        "Lata Mangeshkar","Indian Music Legends","Playback Singing","Bollywood Golden Era","Indian Cinema",
        "Musical Icons","Voice of India","Bharat Ratna","Indian Classical Music","Hindi Film Songs",
        "Legendary Singers","Cultural Heritage","Suvichaar Stories"
    ]
    tag_input = st.text_input(
        "Enter Filter Tags (comma separated):",
        value=st.session_state.get("generated_filter_tags", ", ".join(default_tags)),
        help="Example: Music, Culture, Lata Mangeshkar"
    )

    use_custom_cover = st.radio("Do you want to add a custom cover image URL?", ("No", "Yes"))
    cover_image_url = st.text_input("Enter your custom Cover Image URL") if use_custom_cover == "Yes" else image_url

    with st.form("content_form"):
        submit_button = st.form_submit_button("Submit")

    if submit_button:
        missing_fields = []
        if not story_title.strip():     missing_fields.append("Story Title")
        if not meta_description.strip():missing_fields.append("Meta Description")
        if not meta_keywords.strip():   missing_fields.append("Meta Keywords")
        if not content_type.strip():    missing_fields.append("Content Type")
        if not language.strip():        missing_fields.append("Language")
        if not image_url.strip():       missing_fields.append("Image URL")
        if not tag_input.strip():       missing_fields.append("Filter Tags")
        if not categories.strip():      missing_fields.append("Category")
        if not uploaded_prefinal:       missing_fields.append("Raw HTML File")

        if missing_fields:
            st.error("❌ Please fill all required fields:\n- " + "\n- ".join(missing_fields))
        else:
            st.markdown("### Submitted Data")
            st.write(f"**Story Title:** {story_title}")
            st.write(f"**Meta Description:** {meta_description}")
            st.write(f"**Meta Keywords:** {meta_keywords}")
            st.write(f"**Content Type:** {content_type}")
            st.write(f"**Language:** {language}")

        key_path = "media/default.png"
        uploaded_url = ""

        try:
            nano, slug_nano, canurl, canurl1 = generate_slug_and_urls(story_title)
            page_title = f"{story_title} | Suvichaar"
        except Exception as e:
            st.error(f"Error generating canonical URLs: {e}")
            nano = slug_nano = canurl = canurl1 = page_title = ""

        # Image upload/normalize
        if image_url:
            filename = os.path.basename(urlparse(image_url).path)
            ext = os.path.splitext(filename)[1].lower()
            if ext not in [".jpg", ".jpeg", ".png", ".gif"]:
                ext = ".jpg"

            if image_url.startswith("https://stories.suvichaar.org/"):
                uploaded_url = image_url
                key_path = "/".join(urlparse(image_url).path.split("/")[2:])
            else:
                try:
                    response = requests.get(image_url, timeout=10)
                    response.raise_for_status()
                    unique_filename = f"{uuid.uuid4().hex}{ext}"
                    s3_key = f"{S3_PREFIX}{unique_filename}"
                    s3_client.put_object(
                        Bucket=AWS_BUCKET,
                        Key=s3_key,
                        Body=response.content,
                        ContentType=response.headers.get("Content-Type", "image/jpeg"),
                    )
                    uploaded_url = f"{CDN_BASE}{s3_key}"
                    key_path = s3_key
                    st.success("Image uploaded successfully!")
                except Exception as e:
                    st.warning(f"Failed to fetch/upload image. Using fallback. Error: {e}")
                    uploaded_url = ""
        else:
            st.info("No Image URL provided. Using default.")

        try:
            html_template = uploaded_prefinal.read().decode("utf-8")

            user_mapping = {
                "Mayank": "https://www.instagram.com/iamkrmayank?igsh=eW82NW1qbjh4OXY2&utm_source=qr",
                "Onip":   "https://www.instagram.com/onip.mathur/profilecard/?igsh=MW5zMm5qMXhybGNmdA==",
                "Naman":  "https://njnaman.in/"
            }

            filter_tags = [tag.strip() for tag in tag_input.split(",") if tag.strip()]
            category_mapping = {
                "Art": 1, "Travel": 2, "Entertainment": 3, "Literature": 4, "Books": 5,
                "Sports": 6, "History": 7, "Culture": 8, "Wildlife": 9, "Spiritual": 10, "Food": 11
            }

            filternumber = category_mapping[categories]
            selected_user = random.choice(list(user_mapping.keys()))
            html_template = html_template.replace("{{user}}", selected_user)
            html_template = html_template.replace("{{userprofileurl}}", user_mapping[selected_user])
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

            # If using CDN image, also set resized presets
            if image_url.startswith("http://media.suvichaar.org") or image_url.startswith("https://media.suvichaar.org"):
                html_template = html_template.replace("{{image0}}", image_url)
                parsed_cdn_url = urlparse(image_url)
                cdn_key_path = parsed_cdn_url.path.lstrip("/")
                resize_presets = {
                    "potraitcoverurl": (640, 853),
                    "msthumbnailcoverurl": (300, 300),
                }
                for label, (width, height) in resize_presets.items():
                    template = {
                        "bucket": AWS_BUCKET,
                        "key": cdn_key_path,
                        "edits": { "resize": { "width": width, "height": height, "fit": "cover" } }
                    }
                    encoded = base64.urlsafe_b64encode(json.dumps(template).encode()).decode()
                    final_url = f"{CDN_PREFIX_MEDIA}{encoded}"
                    html_template = html_template.replace(f"{{{label}}}", final_url)

            # Cleanup accidental braces
            html_template = re.sub(r'href="\{(https://[^}]+)\}"', r'href="\1"', html_template)
            html_template = re.sub(r'src="\{(https://[^}]+)\}"', r'src="\1"', html_template)

            st.markdown("### Final Modified HTML")
            st.code(html_template, language="html")

            metadata_dict = {
                "story_title": story_title,
                "categories": filternumber,
                "filterTags": filter_tags,
                "story_uid": nano,
                "story_link": canurl,
                "storyhtmlurl": canurl1,
                "urlslug": slug_nano,
                "cover_image_link": cover_image_url,
                "publisher_id": 1,
                "story_logo_link": "https://media.suvichaar.org/filters:resize/96x96/media/brandasset/suvichaariconblack.png",
                "keywords": meta_keywords,
                "metadescription": meta_description,
                "lang": language
            }

            # Upload page
            s3_key = f"{slug_nano}.html"
            s3_client.put_object(
                Bucket="suvichaarstories",   # keep your publication bucket
                Key=s3_key,
                Body=html_template.encode("utf-8"),
                ContentType="text/html",
            )
            final_story_url = f"https://suvichaar.org/stories/{slug_nano}"
            st.success("✅ HTML uploaded successfully to S3!")
            st.markdown(f"🔗 **Live Story URL:** [Click to view your story]({final_story_url})")

            # Bundle download
            json_str = json.dumps(metadata_dict, indent=4)
            zip_buffer = io.BytesIO()
            with zipfile.ZipFile(zip_buffer, "w") as zip_file:
                zip_file.writestr(f"{slug_nano}.html", html_template)
                zip_file.writestr(f"{slug_nano}_metadata.json", json_str)
            zip_buffer.seek(0)
            st.download_button(
                label="📦 Download HTML + Metadata ZIP",
                data=zip_buffer,
                file_name=f"{story_title}.zip",
                mime="application/zip"
            )

        except Exception as e:
            st.error(f"Error processing HTML: {e}")

# -------------------------
# Tab 6 — Cover Image Request / thumbnail via Remotion
# -------------------------
with tab6:
    # Initialize S3 client once (again here in case of hot-reload)
    s3 = boto3.client(
        "s3",
        aws_access_key_id=AWS_ACCESS_KEY,
        aws_secret_access_key=AWS_SECRET_KEY,
        region_name=AWS_REGION,
    )

    st.title("Cover Image Request")

    uploaded = st.file_uploader("📥 Upload Suvichaar JSON", type=["json"])
    if not uploaded:
        st.info("Please upload a Suvichaar-style JSON to begin.")
        st.stop()

    # Parse & transform
    try:
        data = json.load(uploaded)
        transformed = {}
        for slide_key, info in data.items():
            idx = int(slide_key.replace("slide", "")) if slide_key.startswith("slide") else 0
            if "storytitle" in info:
                text = info["storytitle"]
            elif "hookline" in info:
                text = info["hookline"]
            else:
                text = next((v for k, v in info.items() if "paragraph" in k), "")

            # support either audio_url{n} or audio_url
            audio = info.get(f"audio_url{idx}", info.get("audio_url", ""))

            transformed[slide_key] = {
                f"s{idx}paragraph1": text,
                f"s{idx}audio1":    audio,
                f"s{idx}image1":    "https://media.suvichaar.org/upload/covers/default-cover.png",
                f"s{idx}paragraph2":"Suvichaar"
            }

        st.success("✅ Transformation Complete")
        st.json(transformed)

    except json.JSONDecodeError:
        st.error("❌ Uploaded file is not valid JSON.")
        st.stop()
    except Exception as e:
        st.error(f"❌ Error during transformation: {e}")
        st.stop()

    # Generate thumbnail
    if st.button("Generate Thumbnail"):
        with st.spinner("Generating…"):
            try:
                resp = requests.post(
                    "https://remotion.suvichaar.org/api/generate-news-thumbnail",
                    json=transformed,
                    timeout=30
                )
                resp.raise_for_status()
            except requests.RequestException as err:
                st.error(f"Thumbnail API error: {err}")
                st.stop()

        img_bytes = resp.content
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        key = f"{S3_PREFIX}cover_{ts}.png"

        # Upload without ACL
        try:
            s3.put_object(
                Bucket=AWS_BUCKET,
                Key=key,
                Body=img_bytes,
                ContentType=resp.headers.get("Content-Type", "image/png"),
            )
        except Exception as s3_err:
            st.error(f"S3 upload failed: {s3_err}")
            st.stop()

        cdn_url = f"{CDN_PREFIX_MEDIA}{key}"
        st.success("🖼️ Thumbnail generated and uploaded!")
        st.markdown(f"[View on CDN]({cdn_url})")
        st.image(cdn_url, use_column_width=True)

        # Offer JSON download
        st.download_button(
            label="⬇️ Download Transformed JSON",
            data=json.dumps(transformed, indent=2, ensure_ascii=False),
            file_name=f"CoverJSON_{ts}.json",
            mime="application/json"
        )
