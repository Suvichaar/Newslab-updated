import os, io, re, json, time, uuid, base64, zipfile, random, string, textwrap
from io import BytesIO
from pathlib import Path
from urllib.parse import urlparse
from datetime import datetime, timezone

import requests
import boto3
import nltk
import streamlit as st
from collections import OrderedDict
from dotenv import load_dotenv
from openai import AzureOpenAI

# Azure Speech SDK (for neural voices)
import azure.cognitiveservices.speech as speechsdk

# =========================
# Base Config & Utilities
# =========================
load_dotenv()

# NLTK once
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# ---------- Constants ----------
DEFAULT_COVER_URL = "https://media.suvichaar.org/upload/covers/default_news.png"
DEFAULT_SLIDE_IMAGE_URL = "https://media.suvichaar.org/upload/covers/default_news_slide.png"
DEFAULT_CTA_AUDIO = "https://cdn.suvichaar.org/media/tts_cta_default.mp3"

# Enforce total slides (Headline + middle slides + Hookline). CTA text is included on Hookline slide.
MIN_TOTAL_SLIDES = 8
MAX_TOTAL_SLIDES = 10

# ---- Azure OpenAI Client (for text/gen) ----
client = AzureOpenAI(
    azure_endpoint= st.secrets["azure_api"]["AZURE_OPENAI_ENDPOINT"],
    api_key= st.secrets["azure_api"]["AZURE_OPENAI_API_KEY"],
    api_version="2025-01-01-preview"
)

# ---- Azure Speech (Neural TTS) ----
AZURE_SPEECH_KEY   = st.secrets["azure"]["AZURE_API_KEY"]
AZURE_SPEECH_REGION = st.secrets["azure"].get("AZURE_REGION", "centralindia")  # ensure region matches your resource

# ---- AWS ----
AWS_ACCESS_KEY = st.secrets["aws"]["AWS_ACCESS_KEY"]
AWS_SECRET_KEY = st.secrets["aws"]["AWS_SECRET_KEY"]
AWS_REGION     = st.secrets["aws"]["AWS_REGION"]
AWS_BUCKET     = st.secrets["aws"]["AWS_BUCKET"]        # unified bucket usage
S3_PREFIX      = st.secrets["aws"].get("S3_PREFIX", "media/")
CDN_BASE       = st.secrets["aws"]["CDN_BASE"]
CDN_PREFIX_MEDIA = "https://media.suvichaar.org/"

s3_client = boto3.client(
    "s3",
    aws_access_key_id     = AWS_ACCESS_KEY,
    aws_secret_access_key = AWS_SECRET_KEY,
    region_name           = AWS_REGION,
)

# ---- Voice Options (Azure Neural) ----
voice_options = {
    "1": "en-US-BrandonMultilingualNeural",
    "2": "en-US-JennyNeural",
    "3": "hi-IN-SwaraNeural",
    "4": "hi-IN-MadhurNeural"
}

# -------- Slug and URL generator --------
def generate_slug_and_urls(title):
    if not title or not isinstance(title, str):
        raise ValueError("Invalid title")
    safe = ''.join(c for c in title.lower().replace(" ", "-").replace("_", "-")
                   if c in string.ascii_lowercase + string.digits + '-')
    slug = safe.strip('-') or "story"
    nano = ''.join(random.choices(string.ascii_letters + string.digits + '_-', k=10)) + '_G'
    slug_nano = f"{slug}_{nano}"  # -> slug_nano.html
    return nano, slug_nano, f"https://suvichaar.org/stories/{slug_nano}", f"https://stories.suvichaar.org/{slug_nano}.html"

# === Utility Functions ===
def extract_article(url):
    try:
        import newspaper
        from newspaper import Article
    except Exception:
        st.warning("newspaper3k not installed/available; using URL as title.")
        return url, "No summary available.", "No article content available."

    try:
        article = Article(url)
        article.download()
        article.parse()
        try:
            article.nlp()
        except:
            pass
        title = (article.title or "Untitled Article").strip()
        text = (article.text or "No article content available.").strip()
        summary = (article.summary or text[:300]).strip()
        return title, summary, text
    except Exception as e:
        st.error(f"❌ Failed to extract article from URL. Error: {str(e)}")
        return "Untitled Article", "No summary available.", "No article content available."

def get_sentiment(text):
    try:
        from textblob import TextBlob
    except Exception:
        return "neutral"
    if not text or not text.strip():
        return "neutral"
    clean_text = text.strip().replace("\n", " ")
    polarity = TextBlob(clean_text).sentiment.polarity
    if polarity > 0.2:
        return "positive"
    elif polarity < -0.2:
        return "negative"
    else:
        return "neutral"

def detect_category_and_subcategory(text, content_language="English"):
    if not text or len(text.strip()) < 50:
        return {"category": "Unknown", "subcategory": "General", "emotion": "Neutral"}

    if content_language == "Hindi":
        prompt = f"""
आप एक समाचार विश्लेषण विशेषज्ञ हैं।
इस समाचार लेख का विश्लेषण करें और नीचे तीन बातें बताएं:

1. category (श्रेणी)
2. subcategory (उपश्रेणी)
3. emotion (भावना)

लेख:
\"\"\"{text[:3000]}\"\"\"


जवाब केवल JSON में दें:
{{
  "category": "...",
  "subcategory": "...",
  "emotion": "..."
}}
"""
    else:
        prompt = f"""
You are an expert news analyst.
Analyze the following news article and return:

1. category
2. subcategory
3. emotion

Article:
\"\"\"{text[:3000]}\"\"\"


Return ONLY as JSON:
{{
  "category": "...",
  "subcategory": "...",
  "emotion": "..."
}}
"""
    try:
        response = client.chat.completions.create(
            model="gpt-5-chat",
            messages=[
                {"role": "system", "content": "Classify the news into category, subcategory, and emotion."},
                {"role": "user", "content": prompt.strip()}
            ],
            max_tokens=150
        )
        content = response.choices[0].message.content.strip()
        content = content.strip("json").strip("").strip()
        result = json.loads(content)
        if all(k in result for k in ["category", "subcategory", "emotion"]):
            return result
    except Exception as e:
        print("❌ Category detection failed:", e)

    return {"category": "Unknown", "subcategory": "General", "emotion": "Neutral"}

# -------- Slide content generator (no Polaris) --------
def title_script_generator(category, subcategory, emotion, article_text, content_language="English", middle_count=5):
    """
    Generates middle_count slides (for slides 3..N-1).
    Slide 1 (headline) and Slide 2 (connected context) are handled outside.
    No persona/Polaris voice is used.
    """
    system_prompt = f"""
You are a concise digital news editor.

Create exactly {middle_count} short slide snippets from the article below.
Language: {content_language}

Each slide must contain:
- A short title (max 8 words)
- A narration hint (what to say, not the actual narration)

Return JSON:
{{
  "slides": [
    {{ "title": "...", "prompt": "..." }},
    ...
  ]
}}
"""

    user_prompt = f"""
Category: {category}
Subcategory: {subcategory}
Emotion: {emotion}

Article:
\"\"\"{article_text[:3000]}\"\"\""""

    try:
        response = client.chat.completions.create(
            model="gpt-5-chat",
            messages=[
                {"role": "system", "content": system_prompt.strip()},
                {"role": "user", "content": user_prompt.strip()}
            ],
            temperature=0.4
        )
        content = response.choices[0].message.content.strip()
        content = content.strip("json").strip("").strip()
        payload = json.loads(content)
        slides_raw = payload.get("slides", [])[:middle_count]
    except Exception as e:
        print("❌ Slide generation failed:", e)
        slides_raw = []

    prepared = []
    for s in slides_raw:
        para = s.get("prompt") or s.get("title") or "Content unavailable"
        prepared.append(para.strip())
    while len(prepared) < middle_count:
        prepared.append("More context on this story.")
    return prepared

def modify_tab4_json(original_json):
    updated_json = OrderedDict()
    slide_number = 2  # Start from slide2 since slide1 & slide2 are removed
    for i in range(3, 100):  # Covers slide3 to slide99 (intentional)
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
                    audio_key: content.get("audio_url", ""),
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

# -------- Connected context for Slide 2 --------
def generate_connected_context(title, summary, content_language="English"):
    if content_language == "Hindi":
        base = (summary.split(".")[0] if summary else title).strip()
        return f"जुड़ी जानकारी: {base}."
    else:
        base = (summary.split(".")[0] if summary else title).strip()
        return f"Connected context: {base}."
    
# -------- Hookline (fixed structure for last slide) --------
def generate_fixed_hookline(hookline_candidate, content_language="English"):
    footer = "For Such Content Stay Connected with Suvichar Live\n\nRead | Share | Inspire"
    if not hookline_candidate:
        hookline_candidate = "Stay informed with the latest updates."
    if content_language == "Hindi":
        return f"{hookline_candidate}\n\n{footer}"
    else:
        return f"{hookline_candidate}\n\n{footer}"

def restructure_slide_output_for_middle(middle_slides):
    structured = {}
    for idx, text in enumerate(middle_slides, start=1):
        structured[f"s{idx}paragraph1"] = text.strip() if text else "Content unavailable"
    return structured

def generate_remotion_input(tts_output: dict, fixed_image_url: str, author_name: str = "Suvichaar"):
    remotion_data = OrderedDict()
    slide_index = 1

    # Slide 1: storytitle
    if "storytitle" in tts_output:
        remotion_data[f"slide{slide_index}"] = {
            f"s{slide_index}paragraph1": tts_output["storytitle"],
            f"s{slide_index}audio1": tts_output.get(f"slide{slide_index}", {}).get("audio_url", ""),
            f"s{slide_index}image1": fixed_image_url,
            f"s{slide_index}paragraph2": f"- {author_name}"
        }
        slide_index += 1

    # Middle slides
    for i in range(1, 50):
        key = f"s{i}paragraph1"
        if key in tts_output:
            slide_key = f"slide{slide_index}"
            remotion_data[slide_key] = {
                f"s{slide_index}paragraph1": tts_output[key],
                f"s{slide_index}audio1": tts_output.get(slide_key, {}).get("audio_url", ""),
                f"s{slide_index}image1": fixed_image_url,
                f"s{slide_index}paragraph2": f"- {author_name}"
            }
            slide_index += 1
        else:
            break

    # Hookline
    if "hookline" in tts_output:
        slide_key = f"slide{slide_index}"
        remotion_data[slide_key] = {
            f"s{slide_index}paragraph1": tts_output["hookline"],
            f"s{slide_index}audio1": tts_output.get(slide_key, {}).get("audio_url", ""),
            f"s{slide_index}image1": fixed_image_url,
            f"s{slide_index}paragraph2": f"- {author_name}"
        }
        slide_index += 1

    # Save to file
    timestamp = int(time.time())
    filename = f"remotion_input_{timestamp}.json"
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(remotion_data, f, indent=2, ensure_ascii=False)
    return filename

# ------------------ Azure Speech Neural TTS Helper ------------------
def azure_tts_generate(text: str, voice: str, retries: int = 2, backoff: float = 1.0) -> bytes:
    """
    Generate speech bytes using Azure Speech SDK neural voices.
    Works for Cognitive Services endpoint (not region-specific Speech resource).
    """
    speech_config = speechsdk.SpeechConfig(
        endpoint="https://suvichaarai008818057333687.cognitiveservices.azure.com/",
        subscription=AZURE_SPEECH_KEY
    )
    speech_config.speech_synthesis_voice_name = voice
    speech_config.set_speech_synthesis_output_format(
        speechsdk.SpeechSynthesisOutputFormat.Audio16Khz32KBitRateMonoMp3
    )

    # ✅ Output to memory — safe for Streamlit Cloud
    audio_config = None

    for attempt in range(retries + 1):
        synthesizer = speechsdk.SpeechSynthesizer(
            speech_config=speech_config,
            audio_config=audio_config
        )
        result = synthesizer.speak_text_async(text).get()

        if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
            return result.audio_data  # Return bytes directly

        if result.reason == speechsdk.ResultReason.Canceled and attempt < retries:
            time.sleep(backoff * (2 ** attempt))
            continue

        if result.reason == speechsdk.ResultReason.Canceled:
            details = result.cancellation_details
            raise RuntimeError(
                f"Azure TTS canceled: {details.reason}; error={getattr(details, 'error_details', None)}"
            )
        else:
            raise RuntimeError(f"Azure TTS failed with reason: {result.reason}")

    raise RuntimeError("Azure TTS failed after retries")


def synthesize_and_upload(paragraphs, voice):
    s3 = boto3.client(
        "s3",
        aws_access_key_id=AWS_ACCESS_KEY,
        aws_secret_access_key=AWS_SECRET_KEY,
        region_name=AWS_REGION,
    )
    result = OrderedDict()
    os.makedirs("temp", exist_ok=True)

    slide_index = 1

    # Slide 1: storytitle (headline)
    if "storytitle" in paragraphs:
        storytitle = paragraphs["storytitle"]
        audio_bytes = azure_tts_generate(storytitle, voice)
        filename = f"tts_{uuid.uuid4().hex}.mp3"
        local_path = os.path.join("temp", filename)
        with open(local_path, "wb") as f:
            f.write(audio_bytes)
        s3_key = f"{S3_PREFIX}{filename}"
        s3.upload_file(local_path, AWS_BUCKET, s3_key)
        cdn_url = f"{CDN_BASE}{s3_key}"
        result[f"slide{slide_index}"] = {"storytitle": storytitle, "audio_url": cdn_url, "voice": voice}
        os.remove(local_path)
        slide_index += 1

    # Slide 2..(N-1) : s1paragraph1.. (connected context is s1paragraph1)
    for i in range(1, 50):
        key = f"s{i}paragraph1"
        if key not in paragraphs:
            break
        text_val = paragraphs[key]
        st.write(f"🛠 Processing {key}")
        audio_bytes = azure_tts_generate(text_val, voice)
        filename = f"tts_{uuid.uuid4().hex}.mp3"
        local_path = os.path.join("temp", filename)
        with open(local_path, "wb") as f:
            f.write(audio_bytes)
        s3_key = f"{S3_PREFIX}{filename}"
        s3.upload_file(local_path, AWS_BUCKET, s3_key)
        cdn_url = f"{CDN_BASE}{s3_key}"
        result[f"slide{slide_index}"] = {key: text_val, "audio_url": cdn_url, "voice": voice}
        os.remove(local_path)
        slide_index += 1

    # Last slide: hookline + fixed footer
    if "hookline" in paragraphs:
        hookline_text = paragraphs["hookline"]
        audio_bytes = azure_tts_generate(hookline_text, voice)
        filename = f"tts_{uuid.uuid4().hex}.mp3"
        local_path = os.path.join("temp", filename)
        with open(local_path, "wb") as f:
            f.write(audio_bytes)
        s3_key = f"{S3_PREFIX}{filename}"
        s3.upload_file(local_path, AWS_BUCKET, s3_key)
        cdn_url = f"{CDN_BASE}{s3_key}"
        result[f"slide{slide_index}"] = {"hookline": hookline_text, "audio_url": cdn_url, "voice": voice}
        os.remove(local_path)

    return result

def transliterate_to_devanagari(json_data):
    updated = {}
    for k, v in json_data.items():
        if k.startswith("s") and "paragraph1" in k and isinstance(v, str) and v.strip():
            prompt = f"""Transliterate this Hindi sentence (written in Latin script) into Hindi Devanagari script. Return only the transliterated text:\n\n{v}"""
            try:
                response = client.chat.completions.create(
                    model="gpt-5-chat",
                    messages=[
                        {"role": "system", "content": "You are a Hindi transliteration expert."},
                        {"role": "user", "content": prompt.strip()}
                    ]
                )
                devanagari = response.choices[0].message.content.strip()
                updated[k] = devanagari
            except Exception:
                updated[k] = v
        else:
            updated[k] = v
    return updated

def generate_storytitle(title, summary, content_language="English"):
    if content_language == "Hindi":
        prompt = f"""
आप एक समाचार शीर्षक विशेषज्ञ हैं। नीचे दी गई अंग्रेज़ी समाचार शीर्षक और सारांश को पढ़कर, उसी का अर्थ बनाए रखते हुए एक नया आकर्षक *हिंदी शीर्षक* बनाइए।

अंग्रेज़ी शीर्षक: {title}
सारांश: {summary}

अनुरोध:
- केवल एक पंक्ति
- भाषा सरल और स्पष्ट हो
- उद्धरण ("") शामिल न करें

अब कृपया हिंदी शीर्षक दीजिए:
"""
        try:
            response = client.chat.completions.create(
                model="gpt-5-chat",
                messages=[
                    {"role": "system", "content": "You generate clear and catchy news headlines."},
                    {"role": "user", "content": prompt.strip()}
                ]
            )
            return response.choices[0].message.content.strip().strip('"')
        except Exception as e:
            print(f"❌ Storytitle generation failed: {e}")
            return title.strip()
    else:
        return title.strip()

# === Streamlit UI ===
st.title("🧠 Web Story Content Generator")

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["Step:1", "Step:2", "Step:3","Step:4","Step:5","Step:6"])

# 🧠 Streamlit UI – Tab 1
with tab1:
    st.title("🧠 Generalized Web Story Prompt Generator")

    url = st.text_input("Enter a news article URL")
    content_language = st.selectbox("Choose content language", ["English", "Hindi"])
    total_slides = st.number_input(
        "Total slides (including Headline and Hookline)",
        min_value=MIN_TOTAL_SLIDES,
        max_value=MAX_TOTAL_SLIDES,
        value=MIN_TOTAL_SLIDES,
        step=1
    )

    if st.button("🚀 Submit and Generate JSON"):
        if url:
            with st.spinner("Analyzing the article and generating prompts..."):
                try:
                    # Extract + Analyze
                    title, summary, full_text = extract_article(url)
                    sentiment = get_sentiment(summary or full_text)
                    result = detect_category_and_subcategory(full_text, content_language)
                    category = result["category"]
                    subcategory = result["subcategory"]
                    emotion = result["emotion"]

                    # Headline (slide 1)
                    storytitle = generate_storytitle(title, summary, content_language)

                    # Connected (slide 2)
                    connected_context = generate_connected_context(title, summary, content_language)

                    # Middle slides count = total - 3 (headline + connected + hookline)
                    middle_count = max(0, total_slides - 3)
                    middle_slides = title_script_generator(
                        category, subcategory, emotion, full_text, content_language, middle_count=middle_count
                    )

                    # Hookline seed
                    hook_seed = "यह कहानी आपको सोचने पर मजबूर कर देगी." if content_language == "Hindi" else "This story might surprise you."
                    final_hookline = generate_fixed_hookline(hook_seed, content_language)

                    # Flatten into story JSON
                    structured_output = OrderedDict()
                    structured_output["storytitle"] = storytitle

                    middle_struct = restructure_slide_output_for_middle([connected_context] + middle_slides)
                    for i in range(1, len(middle_struct) + 1):
                        structured_output[f"s{i}paragraph1"] = middle_struct[f"s{i}paragraph1"]

                    structured_output["hookline"] = final_hookline

                    # Hindi transliteration (only for paragraphs)
                    if content_language == "Hindi":
                        structured_output = transliterate_to_devanagari(structured_output)

                    # Save + Download JSON
                    timestamp = int(time.time())
                    filename = f"structured_slides_{timestamp}.json"
                    with open(filename, "w", encoding="utf-8") as f:
                        json.dump(structured_output, f, indent=2, ensure_ascii=False)

                    with open(filename, "r", encoding="utf-8") as f:
                        st.success("✅ Prompt generation complete!! Click below to download:")
                        st.download_button(
                            label=f"⬇ Download JSON ({timestamp})",
                            data=f.read(),
                            file_name=filename,
                            mime="application/json"
                        )

                    st.info(f"Total slides targeted: {total_slides}  ➜ Headline (1) + Middle ({total_slides-3}) + Hookline (1) + Connected (1)")

                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
        else:
            st.warning("Please enter a valid URL.")

with tab2:
    st.title("🎙 Text-to-Speech to S3 (Azure Neural Voices)")
    uploaded_file = st.file_uploader("Upload structured slide JSON", type=["json"])
    voice_label = st.selectbox("Choose Voice", list(voice_options.values()))

    if uploaded_file and voice_label:
        paragraphs = json.load(uploaded_file)
        st.success(f"✅ Loaded {len(paragraphs)} keys")

        if st.button("🚀 Generate TTS + Upload to S3"):
            with st.spinner("Please wait..."):
                output = synthesize_and_upload(paragraphs, voice_label)
                st.success("✅ Done uploading to S3!")
                timestamp = int(time.time())
                output_filename = f"tts_output_{timestamp}.json"
        
                # Save TTS output
                with open(output_filename, "w", encoding="utf-8") as f:
                    json.dump(output, f, indent=2, ensure_ascii=False)
        
                # Remotion generation (neutral image)
                fixed_image_url = DEFAULT_SLIDE_IMAGE_URL
                remotion_filename = generate_remotion_input(output, fixed_image_url, author_name="Suvichaar")
        
                # Download TTS JSON
                with open(output_filename, "r", encoding="utf-8") as f:
                    st.download_button(
                        label="⬇ Download Output JSON",
                        data=f.read(),
                        file_name=output_filename,
                        mime="application/json"
                    )

with tab3:
    st.title("🧩 Saving modified file")
    uploaded_file = st.file_uploader("📤 Upload Full Slide JSON (with slide1 to slideN)", type=["json"])

    if uploaded_file:
        json_data = json.load(uploaded_file)
        st.success("✅ JSON Loaded")

        try:
            with open("test.html", "r", encoding="utf-8") as f:
                html_template = f.read()
        except FileNotFoundError:
            st.error("❌ Could not find templates/test.html. Please make sure it exists.")
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
                    label="⬇ Download ZIP with HTML + JSON",
                    data=buffer,
                    file_name=zip_filename,
                    mime="application/zip"
                )

with tab4:
    st.title("🎞 AMP Web Story Generator with Full Animation and Audio")
    
    def generate_slide(paragraph: str, audio_url: str):
        return f"""
        <amp-story-page auto-advance-after="page-audio" class="i-amphtml-layout-container" i-amphtml-layout="container">
            <amp-story-grid-layer template="fill">
                <amp-img layout="fill" src="{DEFAULT_SLIDE_IMAGE_URL}" alt="slide" disable-inline-width="true"></amp-img>
            </amp-story-grid-layer>

            <amp-story-grid-layer template="fill">
                <amp-video id="page-audio" autoplay layout="fixed" width="1" height="1">
                    <source type="audio/mpeg" src="{audio_url}">
                </amp-video>
            </amp-story-grid-layer>

            <amp-story-grid-layer template="vertical" aspect-ratio="412:618" style="--aspect-ratio:412/618;">
                <div class="page-fullbleed-area">
                    <div class="page-safe-area">
                        <h3 class="story-text" style="padding:16px; font-size:22px; line-height:1.35; font-weight:600;">
                            {paragraph}
                        </h3>
                    </div>
                </div>
            </amp-story-grid-layer>
        </amp-story-page>
        """

    uploaded_html_file = st.file_uploader("📄 Upload AMP Template HTML (with <!--INSERT_SLIDES_HERE-->)", type=["html"], key="html_upload_tab3")
    uploaded_json_file = st.file_uploader("📦 Upload Output JSON", type=["json"], key="json_upload_tab3")

    if uploaded_html_file and uploaded_json_file:
        try:
            template_html = uploaded_html_file.read().decode("utf-8")
            output_data = json.load(uploaded_json_file)

            if "<!--INSERT_SLIDES_HERE-->" not in template_html:
                st.error("❌ Placeholder <!--INSERT_SLIDES_HERE--> not found in uploaded HTML.")
            else:
                all_slides = ""
                ordered_keys = sorted(
                    [k for k in output_data.keys() if k.startswith("slide")],
                    key=lambda x: int(x.replace("slide", "")) if x.startswith("slide") else 9999
                )

                if not ordered_keys:
                    # Fallback to reconstruct from generic structure
                    constructed = OrderedDict()
                    idx = 1
                    if "storytitle" in output_data:
                        constructed["slide1"] = {
                            "s1paragraph1": output_data["storytitle"],
                            "audio_url1": output_data.get("slide1", {}).get("audio_url", "")
                        }
                        idx += 1
                    p_i = 1
                    while f"s{p_i}paragraph1" in output_data:
                        constructed[f"slide{idx}"] = {
                            f"s{idx}paragraph1": output_data[f"s{p_i}paragraph1"],
                            f"audio_url{idx}": output_data.get(f"slide{idx}", {}).get("audio_url", "")
                        }
                        idx += 1
                        p_i += 1
                    if "hookline" in output_data:
                        constructed[f"slide{idx}"] = {
                            f"s{idx}paragraph1": output_data["hookline"],
                            f"audio_url{idx}": output_data.get(f"slide{idx}", {}).get("audio_url", "")
                        }
                    output_data = constructed
                    ordered_keys = sorted(output_data.keys(), key=lambda x: int(x.replace("slide", "")))

                for key in ordered_keys:
                    slide_num = key.replace("slide", "")
                    data = output_data[key]
                    para_key = f"s{slide_num}paragraph1"
                    audio_key = f"audio_url{slide_num}"

                    if para_key in data and audio_key in data:
                        raw = str(data[para_key]).replace("’", "'").replace('"', '&quot;')
                        paragraph = textwrap.shorten(raw, width=180, placeholder="...")
                        audio_url = data[audio_key] or ""
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
            st.error(f"⚠ Error: {str(e)}")

with tab5:
    st.header("Content Submission Form")

    if "last_title" not in st.session_state:
        st.session_state.last_title = ""
        st.session_state.meta_description = ""
        st.session_state.meta_keywords = ""

    story_title = st.text_input("Story Title")
    
    if story_title.strip() and story_title != st.session_state.last_title:
        with st.spinner("Generating meta description, keywords, and filter tags..."):
            messages = [
                {
                    "role": "user",
                    "content": f"""
                    Generate the following for a web story titled '{story_title}':
                    1. A short SEO-friendly meta description
                    2. Meta keywords (comma separated)
                    3. Relevant filter tags (comma separated, suitable for categorization and content filtering)"""
                }
            ]
            try:
                response = client.chat.completions.create(
                    model="gpt-5-chat",
                    messages=messages,
                    max_tokens=300,
                    temperature=0.5,
                )
                output = response.choices[0].message.content
    
                # Extract metadata using regex
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
    uploaded_prefinal = st.file_uploader("💾 Upload pre-final AMP HTML (optional)", type=["html","htm"], key="prefinal_upload")
    
    if uploaded_prefinal is None:
        st.error("Please upload a pre-final AMP HTML file before submitting.")

    categories = st.selectbox("Select your Categories", ["Art", "Travel", "Entertainment", "Literature", "Books", "Sports", "History", "Culture", "Wildlife", "Spiritual", "Food"])

    default_tags = [
        "Indian News",
        "Current Affairs",
        "Public Interest",
        "Suvichaar Stories"
    ]
    tag_input = st.text_input(
        "Enter Filter Tags (comma separated):",
        value=st.session_state.get("generated_filter_tags", ", ".join(default_tags)),
        help="Example: News, India, Updates"
    )

    use_custom_cover = st.radio("Do you want to add a custom cover image URL?", ("No", "Yes"))
    if use_custom_cover == "Yes":
        cover_image_url = st.text_input("Enter your custom Cover Image URL")
    else:
        cover_image_url = image_url or DEFAULT_COVER_URL

    with st.form("content_form"):
        submit_button = st.form_submit_button("Submit")

if 'submit_button' in locals() and submit_button:
    # Validation before processing
    missing_fields = []
    if not story_title.strip(): missing_fields.append("Story Title")
    if not meta_description.strip(): missing_fields.append("Meta Description")
    if not meta_keywords.strip(): missing_fields.append("Meta Keywords")
    if not content_type.strip(): missing_fields.append("Content Type")
    if not language.strip(): missing_fields.append("Language")
    if not image_url.strip(): st.info("No Image URL provided. Using default.")
    if not tag_input.strip(): missing_fields.append("Filter Tags")
    if not categories.strip(): missing_fields.append("Category")
    if not uploaded_prefinal: missing_fields.append("Raw HTML File")

    if missing_fields:
        st.error(f"❌ Please fill all required fields before submitting:\n- " + "\n- ".join(missing_fields))
    else:
        st.markdown("### Submitted Data")
        st.write(f"*Story Title:* {story_title}")
        st.write(f"*Meta Description:* {meta_description}")
        st.write(f"*Meta Keywords:* {meta_keywords}")
        st.write(f"*Content Type:* {content_type}")
        st.write(f"*Language:* {language}")

    key_path = "media/default.png"
    uploaded_url = ""

    try:
        nano, slug_nano, canurl, canurl1 = generate_slug_and_urls(story_title)
        page_title = f"{story_title} | Suvichaar"
    except Exception as e:
        st.error(f"Error generating canonical URLs: {e}")
        nano = slug_nano = canurl = canurl1 = page_title = ""

    # Image URL handling
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
        uploaded_url = DEFAULT_COVER_URL

    try:
        # use the uploaded HTML as the working template
        html_template = uploaded_prefinal.read().decode("utf-8")
    
        user_mapping = {
            "Mayank": "https://www.instagram.com/iamkrmayank?igsh=eW82NW1qbjh4OXY2&utm_source=qr",
            "Onip": "https://www.instagram.com/onip.mathur/profilecard/?igsh=MW5zMm5qMXhybGNmdA==",
            "Naman": "https://njnaman.in/"
        }

        filter_tags = [tag.strip() for tag in tag_input.split(",") if tag.strip()]
        category_mapping = {
            "Art": 1, "Travel": 2, "Entertainment": 3, "Literature": 4, "Books": 5,
            "Sports": 6, "History": 7, "Culture": 8, "Wildlife": 9, "Spiritual": 10, "Food": 11
        }

        filternumber = category_mapping.get(categories, 1)
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

        # Replace image placeholders
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
                    "edits": {"resize": {"width": width, "height": height, "fit": "cover"}}
                }
                encoded = base64.urlsafe_b64encode(json.dumps(template).encode()).decode()
                final_url = f"{CDN_PREFIX_MEDIA}{encoded}"
                html_template = html_template.replace(f"{{{label}}}", final_url)
        else:
            html_template = html_template.replace("{{image0}}", uploaded_url or DEFAULT_COVER_URL)
            for label in ["potraitcoverurl", "msthumbnailcoverurl"]:
                html_template = html_template.replace(f"{{{label}}}", uploaded_url or DEFAULT_COVER_URL)

        # Cleanup incorrect {url} wrapping
        html_template = re.sub(r'href="\{(https://[^}]+)\}"', r'href="\1"', html_template)
        html_template = re.sub(r'src="\{(https://[^}]+)\}"', r'src="\1"', html_template)

        st.markdown("### Final Modified HTML")
        st.code(html_template, language="html")

        # ----------- Generate and Provide Metadata JSON -------------
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

        # Upload HTML using unified AWS_BUCKET
        s3_client.put_object(
            Bucket=AWS_BUCKET,
            Key=f"{slug_nano}.html",
            Body=html_template.encode("utf-8"),
            ContentType="text/html",
        )

        final_story_url = f"https://suvichaar.org/stories/{slug_nano}"  # canurl
        st.success("✅ HTML uploaded successfully to S3!")
        st.markdown(f"🔗 *Live Story URL:* [Click to view your story]({final_story_url})")
        
        json_str = json.dumps(metadata_dict, indent=4)

        # Save data to session_state
        zip_buffer = BytesIO()
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

with tab6:
    # ── AWS CONFIG (same clients) ────────────────────────────────────
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
            if idx == 0:
                continue
            if "storytitle" in info:
                text = info["storytitle"]
            elif "hookline" in info:
                text = info["hookline"]
            else:
                text = next((v for k, v in info.items() if "paragraph" in k), "")
            audio = info.get("audio_url", "")
    
            transformed[slide_key] = {
                f"s{idx}paragraph1": text,
                f"s{idx}audio1":    audio,
                f"s{idx}image1":    DEFAULT_SLIDE_IMAGE_URL,
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
    
        # Upload
        try:
            s3_client.put_object(
                Bucket=AWS_BUCKET,
                Key=key,
                Body=img_bytes,
                ContentType=resp.headers.get("Content-Type", "image/png"),
            )
        except Exception as s3_err:
            st.error(f"S3 upload failed: {s3_err}")
            st.stop()
    
        cdn_url = f"{CDN_PREFIX_MEDIA}{key}"
        st.success("🖼 Thumbnail generated and uploaded!")
        st.markdown(f"[View on CDN]({cdn_url})")
        st.image(cdn_url, use_column_width=True)
    
        # Offer JSON download
        st.download_button(
            label="⬇ Download Transformed JSON",
            data=json.dumps(transformed, indent=2, ensure_ascii=False),
            file_name=f"CoverJSON_{ts}.json",
            mime="application/json"
        )
