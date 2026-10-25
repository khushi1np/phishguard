import re
import io
import datetime
import pandas as pd
import streamlit as st
from urllib.parse import urlparse
from sklearn.ensemble import RandomForestClassifier

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PhishGuard",
    page_icon="🛡️",
    layout="centered",
)

# ── Custom dark-theme CSS ──────────────────────────────────────────────────────
st.markdown(
    """
    <style>
        .stApp { background-color: #0f1117; color: #e6edf3; }
        label { color: #8b949e !important; font-size: 0.9rem; }
        .stTextInput > div > div > input {
            background-color: #161b22;
            border: 1px solid #30363d;
            border-radius: 8px;
            color: #e6edf3;
            padding: 10px 14px;
        }
        .stButton > button {
            background-color: #238636;
            color: #ffffff;
            border: none;
            border-radius: 8px;
            padding: 10px 28px;
            font-size: 1rem;
            font-weight: 600;
            width: 100%;
            margin-top: 8px;
        }
        .stButton > button:hover { background-color: #2ea043; }

        /* Verdict banners */
        .result-safe {
            background-color: #0d1117;
            border: 1px solid #238636;
            border-left: 5px solid #3fb950;
            border-radius: 8px;
            padding: 16px 20px;
            margin-top: 20px;
            color: #3fb950;
            font-size: 1.1rem;
            font-weight: 600;
        }
        .result-suspicious {
            background-color: #0d1117;
            border: 1px solid #e3b341;
            border-left: 5px solid #e3b341;
            border-radius: 8px;
            padding: 16px 20px;
            margin-top: 20px;
            color: #e3b341;
            font-size: 1.1rem;
            font-weight: 600;
        }
        .result-dangerous {
            background-color: #0d1117;
            border: 1px solid #f85149;
            border-left: 5px solid #f85149;
            border-radius: 8px;
            padding: 16px 20px;
            margin-top: 20px;
            color: #f85149;
            font-size: 1.1rem;
            font-weight: 600;
        }

        /* ML prediction box */
        .ml-box {
            background-color: #161b22;
            border: 1px solid #30363d;
            border-radius: 8px;
            padding: 12px 18px;
            margin-top: 14px;
            font-size: 0.92rem;
            color: #e6edf3;
        }
        .ml-box span.ml-label { font-weight: 600; }
        .ml-safe    { color: #3fb950; }
        .ml-phishing { color: #f85149; }

        /* Risk score bar */
        .score-wrap { margin-top: 20px; margin-bottom: 4px; }
        .score-label {
            display: flex;
            justify-content: space-between;
            color: #8b949e;
            font-size: 0.85rem;
            margin-bottom: 6px;
        }
        .score-track {
            background-color: #21262d;
            border-radius: 6px;
            height: 14px;
            width: 100%;
            overflow: hidden;
        }
        .score-fill { height: 14px; border-radius: 6px; }
        .score-fill-safe       { background-color: #3fb950; }
        .score-fill-suspicious { background-color: #e3b341; }
        .score-fill-dangerous  { background-color: #f85149; }

        /* Per-check table */
        .check-table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 18px;
            font-size: 0.9rem;
        }
        .check-table th {
            text-align: left;
            color: #8b949e;
            border-bottom: 1px solid #30363d;
            padding: 6px 10px;
        }
        .check-table td {
            padding: 8px 10px;
            border-bottom: 1px solid #21262d;
            color: #e6edf3;
            vertical-align: top;
        }
        .badge-pass {
            background-color: #1a3a24;
            color: #3fb950;
            border-radius: 4px;
            padding: 2px 10px;
            font-weight: 600;
            white-space: nowrap;
        }
        .badge-fail {
            background-color: #3a1a1a;
            color: #f85149;
            border-radius: 4px;
            padding: 2px 10px;
            font-weight: 600;
            white-space: nowrap;
        }
        .reason-text { color: #8b949e; font-size: 0.85rem; }

        /* History table */
        .hist-table {
            width: 100%;
            border-collapse: collapse;
            font-size: 0.88rem;
            margin-top: 10px;
        }
        .hist-table th {
            text-align: left;
            color: #8b949e;
            border-bottom: 1px solid #30363d;
            padding: 6px 10px;
        }
        .hist-table td {
            padding: 7px 10px;
            border-bottom: 1px solid #21262d;
            color: #c9d1d9;
            word-break: break-all;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── ML: sample dataset & model ────────────────────────────────────────────────
# 40 labelled example URLs (0 = safe, 1 = phishing)
SAMPLE_URLS = [
    # Safe (label 0)
    ("https://www.google.com", 0),
    ("https://github.com/openai/gpt", 0),
    ("https://stackoverflow.com/questions/12345", 0),
    ("https://docs.python.org/3/library/re.html", 0),
    ("https://en.wikipedia.org/wiki/Phishing", 0),
    ("https://www.amazon.com/dp/B09XYZ", 0),
    ("https://www.bbc.co.uk/news", 0),
    ("https://streamlit.io/docs", 0),
    ("https://developer.mozilla.org/en-US/docs/Web", 0),
    ("https://news.ycombinator.com", 0),
    ("https://www.reddit.com/r/python", 0),
    ("https://pypi.org/project/scikit-learn/", 0),
    ("https://www.microsoft.com/en-us/windows", 0),
    ("https://support.apple.com/en-us/HT201222", 0),
    ("https://cloud.google.com/storage/docs", 0),
    ("https://www.linkedin.com/in/johndoe", 0),
    ("https://medium.com/@author/article-slug", 0),
    ("https://www.nytimes.com/2024/01/01/tech", 0),
    ("https://hub.docker.com/_/python", 0),
    ("https://www.youtube.com/watch?v=dQw4w9WgXcQ", 0),
    # Phishing (label 1)
    ("http://192.168.1.1/login/verify", 1),
    ("http://paypal.com.secure-login.xyz/update", 1),
    ("https://verify.login.bank-secure.com/update/account", 1),
    ("http://www.amazon.com@evil.com/login", 1),
    ("http://secure.login.verify.bank.phish.net/update", 1),
    ("http://172.16.0.1/secure/bank/login", 1),
    ("https://login.verify.update.secure.fakebank.com/account", 1),
    ("http://ebay.com.login-verify.ru/secure", 1),
    ("http://10.0.0.1/bank/login/verify/update", 1),
    ("https://apple.com.id-verify.com/secure-update/login", 1),
    ("http://paypal-secure.login.verify.update.fakesite.biz/account", 1),
    ("http://192.0.2.50/login/bank/verify", 1),
    ("https://netflix.login.verify.secure.update.phishme.com/", 1),
    ("http://bankofamerica.com.login.secure.xyz/verify/update", 1),
    ("http://wellsfargo.secure.login.verify.1337hax.com/update", 1),
    ("http://microsoft.com@malicious.ru/login", 1),
    ("https://login.verify.secure.bank.update.shadydomain.com/path", 1),
    ("http://203.0.113.5/secure/login?bank=update&verify=1", 1),
    ("http://instagram.com.login.secure.verify.phish.io/update", 1),
    ("http://amazon.secure.login.verify.update.badsite.co/account", 1),
]

SUSPICIOUS_WORDS = ["login", "verify", "bank", "secure", "update"]

def extract_features(url: str) -> list:
    """Turn a URL into a list of 6 numbers the model can read."""
    hostname = urlparse(url).hostname or ""
    is_ip    = int(bool(re.match(r"^\d{1,3}(\.\d{1,3}){3}$", hostname)))
    n_suspicious = sum(1 for w in SUSPICIOUS_WORDS if w in url.lower())
    return [
        len(url),                          # 1. URL length
        url.count("."),                    # 2. number of dots
        int("@" in url),                   # 3. has @ symbol (1/0)
        is_ip,                             # 4. hostname is IP (1/0)
        int(url.startswith("https://")),   # 5. uses HTTPS (1/0)
        n_suspicious,                      # 6. count of suspicious words
    ]

@st.cache_resource
def train_model():
    """Train a RandomForestClassifier on the sample dataset. Cached so it only runs once."""
    X = [extract_features(url) for url, _ in SAMPLE_URLS]
    y = [label               for _,   label in SAMPLE_URLS]
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model

# ── Rule-based checks ─────────────────────────────────────────────────────────

def check_at_symbol(url: str) -> dict:
    passed = "@" not in url
    return {
        "name": "No @ symbol", "points": 20, "passed": passed,
        "reason": "Safe — no @ symbol found." if passed
                  else "Contains '@' — attackers hide the real destination after this symbol.",
    }

def check_url_length(url: str) -> dict:
    passed = len(url) <= 75
    return {
        "name": "URL length ≤ 75", "points": 10, "passed": passed,
        "reason": f"Safe — URL is {len(url)} characters." if passed
                  else f"URL is {len(url)} characters — unusually long URLs are a common phishing trick.",
    }

def check_ip_address(url: str) -> dict:
    hostname = urlparse(url).hostname or ""
    is_ip    = bool(re.match(r"^\d{1,3}(\.\d{1,3}){3}$", hostname))
    passed   = not is_ip
    return {
        "name": "Domain (not IP)", "points": 25, "passed": passed,
        "reason": "Safe — uses a domain name, not a raw IP address." if passed
                  else f"Uses IP address '{hostname}' instead of a domain — a red flag for phishing.",
    }

def check_subdomains(url: str) -> dict:
    hostname = urlparse(url).hostname or ""
    count    = len(hostname.split("."))
    passed   = count <= 4
    return {
        "name": "Subdomain count ≤ 3", "points": 15, "passed": passed,
        "reason": f"Safe — {count - 1} subdomain level(s) detected." if passed
                  else f"Has {count - 1} subdomain levels — phishing sites often stack subdomains to look legit.",
    }

def check_https(url: str) -> dict:
    passed = url.startswith("https://")
    return {
        "name": "Uses HTTPS", "points": 15, "passed": passed,
        "reason": "Safe — connection is encrypted (HTTPS)." if passed
                  else "Does not use HTTPS — your data could be intercepted on this site.",
    }

def check_suspicious_words(url: str) -> dict:
    found  = [w for w in SUSPICIOUS_WORDS if w in url.lower()]
    passed = len(found) == 0
    return {
        "name": "No suspicious words", "points": 15, "passed": passed,
        "reason": "Safe — no suspicious keywords found." if passed
                  else f"Contains suspicious word(s): {', '.join(found)} — commonly used in phishing URLs.",
    }

def run_all_checks(url: str) -> list:
    return [
        check_https(url),
        check_at_symbol(url),
        check_url_length(url),
        check_ip_address(url),
        check_subdomains(url),
        check_suspicious_words(url),
    ]

def calculate_score(results: list) -> int:
    return sum(r["points"] for r in results if not r["passed"])

def get_verdict(score: int) -> tuple:
    if score <= 30:   return "Safe",       "result-safe",       "✅"
    elif score <= 60: return "Suspicious", "result-suspicious", "⚠️"
    else:             return "Dangerous",  "result-dangerous",  "🚨"

# ── Session state: history list ───────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []   # list of dicts

# ── UI: header ────────────────────────────────────────────────────────────────
st.title("🛡️ PhishGuard")
st.markdown(
    "<p style='color:#8b949e; margin-top:-10px;'>"
    "Paste any URL below to instantly check whether it looks <strong>safe</strong> "
    "or <strong>suspicious</strong>. No data is sent anywhere — everything runs locally."
    "</p>",
    unsafe_allow_html=True,
)
st.markdown("---")

url_input = st.text_input("URL to check", placeholder="https://example.com/page")

if st.button("Check URL"):
    url = url_input.strip()

    if not url:
        st.warning("Please enter a URL first.")
    else:
        # ── Rule-based analysis ──
        results              = run_all_checks(url)
        score                = calculate_score(results)
        label, css_class, icon = get_verdict(score)

        bar_colour = {
            "result-safe":       "score-fill-safe",
            "result-suspicious": "score-fill-suspicious",
            "result-dangerous":  "score-fill-dangerous",
        }[css_class]

        # ── ML prediction ──
        model       = train_model()
        features    = extract_features(url)
        prediction  = model.predict([features])[0]           # 0 = safe, 1 = phishing
        probability = model.predict_proba([features])[0]     # [prob_safe, prob_phishing]
        ml_label    = "Phishing" if prediction == 1 else "Safe"
        ml_conf     = round(probability[prediction] * 100, 1)
        ml_css      = "ml-phishing" if prediction == 1 else "ml-safe"

        # ── Risk score bar ──
        st.markdown(
            f"<div class='score-wrap'>"
            f"  <div class='score-label'>"
            f"    <span>Risk Score</span>"
            f"    <span><strong>{score}</strong> / 100</span>"
            f"  </div>"
            f"  <div class='score-track'>"
            f"    <div class='score-fill {bar_colour}' style='width:{score}%'></div>"
            f"  </div>"
            f"</div>",
            unsafe_allow_html=True,
        )

        # ── Rule-based verdict banner ──
        st.markdown(
            f"<div class='{css_class}'>{icon} {label}</div>",
            unsafe_allow_html=True,
        )

        # ── ML verdict box ──
        st.markdown(
            f"<div class='ml-box'>"
            f"  🤖 <strong>ML Prediction:</strong> "
            f"  <span class='ml-label {ml_css}'>{ml_label}</span>"
            f"  <span style='color:#8b949e'> — {ml_conf}% confidence</span>"
            f"</div>",
            unsafe_allow_html=True,
        )

        # ── Per-check breakdown table ──
        rows_html = ""
        for r in results:
            if r["passed"]:
                badge = "<span class='badge-pass'>PASS</span>"
                pts   = "<span style='color:#8b949e'>+0 pts</span>"
            else:
                badge = "<span class='badge-fail'>FAIL</span>"
                pts   = f"<span style='color:#f85149'>+{r['points']} pts</span>"
            rows_html += (
                f"<tr>"
                f"<td>{badge}</td>"
                f"<td><strong>{r['name']}</strong><br>"
                f"<span class='reason-text'>{r['reason']}</span></td>"
                f"<td style='text-align:right;white-space:nowrap'>{pts}</td>"
                f"</tr>"
            )

        st.markdown(
            f"<table class='check-table'>"
            f"<thead><tr><th>Result</th><th>Check</th>"
            f"<th style='text-align:right'>Points</th></tr></thead>"
            f"<tbody>{rows_html}</tbody>"
            f"</table>",
            unsafe_allow_html=True,
        )

        # ── Save to history ──
        st.session_state.history.append({
            "Time":        datetime.datetime.now().strftime("%H:%M:%S"),
            "URL":         url,
            "Score":       score,
            "Verdict":     label,
            "ML Result":   ml_label,
            "ML Conf (%)": ml_conf,
        })

# ── History section ───────────────────────────────────────────────────────────
if st.session_state.history:
    st.markdown("---")
    st.subheader("📋 Check History")

    col1, col2 = st.columns([1, 1])

    with col1:
        if st.button("🗑️ Clear History"):
            st.session_state.history = []
            st.rerun()

    with col2:
        # Build CSV in memory and offer it as a download
        df  = pd.DataFrame(st.session_state.history)
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="⬇️ Download CSV",
            data=csv,
            file_name="phishguard_history.csv",
            mime="text/csv",
        )

    # Render history as an HTML table
    rows_html = ""
    for entry in reversed(st.session_state.history):   # newest first
        rows_html += (
            f"<tr>"
            f"<td>{entry['Time']}</td>"
            f"<td>{entry['URL']}</td>"
            f"<td>{entry['Score']}</td>"
            f"<td>{entry['Verdict']}</td>"
            f"<td>{entry['ML Result']} ({entry['ML Conf (%)']}%)</td>"
            f"</tr>"
        )

    st.markdown(
        f"<table class='hist-table'>"
        f"<thead><tr>"
        f"<th>Time</th><th>URL</th><th>Score</th><th>Verdict</th><th>ML</th>"
        f"</tr></thead>"
        f"<tbody>{rows_html}</tbody>"
        f"</table>",
        unsafe_allow_html=True,
    )
