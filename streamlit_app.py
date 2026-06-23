import os
import re
import json
import subprocess
import sys
import smtplib
import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="The Indian Flowers USA",
    page_icon="🌸",
    layout="centered"
)

st.title("🌸 The Indian Flowers USA")
st.caption("Chat with Priya — your personal flower assistant")

# ─────────────────────────────────────────────
# 1. INSTALL PLAYWRIGHT BROWSERS (once)
# ─────────────────────────────────────────────
@st.cache_resource
def install_playwright():
    try:
        subprocess.run(
            [sys.executable, "-m", "playwright", "install", "chromium"],
            capture_output=True,
            check=True
        )
    except Exception as e:
        print(f"[Playwright install error]: {e}")

install_playwright()

# ─────────────────────────────────────────────
# 2. LLM SETUP — cached so it loads only once
# ─────────────────────────────────────────────
@st.cache_resource
def get_llms():
    try:
        groq_api_key = st.secrets["GROQ_API_KEY"]
    except Exception:
        try:
            groq_api_key = st.secrets["groq_api_key"]
        except Exception:
            groq_api_key = os.environ.get("GROQ_API_KEY", "")
    chat_llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0.2,
        max_tokens=400,
        groq_api_key=groq_api_key,
    )
    extract_llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0.0,
        max_tokens=200,
        groq_api_key=groq_api_key,
    )
    return chat_llm, extract_llm

chat_llm, extract_llm = get_llms()

# ─────────────────────────────────────────────
# 3. SCRAPER HELPER — fetch one category page
# ─────────────────────────────────────────────
def scrape_category(page, url: str) -> list[dict]:
    """
    Visit a WooCommerce category page and extract (name, price, unit) for each product.
    Returns a list of dicts: {name, price, unit}
    """
    results = []
    try:
        page.goto(url, timeout=30000)
        page.wait_for_load_state("networkidle", timeout=20000)
        html = page.inner_text("body")

        # Each product appears as lines like:
        #   "Mullai String 5ft $13.00"  or  "Jasmine String $13.00 $11.00" (sale)
        # We parse: product name  +  final (current / lowest) price
        # Pattern: capture everything before the last $XX.XX on the line
        for line in html.splitlines():
            line = line.strip()
            # Find all dollar amounts on this line
            prices = re.findall(r'\$(\d+(?:\.\d{2})?)', line)
            if not prices:
                continue
            # Skip lines that are clearly navigation/footer noise
            if len(line) > 120 or line.lower().startswith(("home", "skip", "menu", "search", "copyright")):
                continue
            # The last price is the current (sale) price; first is original
            current_price = float(prices[-1])
            if current_price <= 0:
                continue
            # Remove all "$XX.XX" fragments and the word "Sale!" to get the name
            name_raw = re.sub(r'Sale!', '', line, flags=re.IGNORECASE)
            name_raw = re.sub(r'\$\d+(?:\.\d{2})?[^\s]*', '', name_raw)
            name_raw = re.sub(r'(Original price was|Current price is|Add to cart|Read more)', '', name_raw, flags=re.IGNORECASE)
            name_raw = re.sub(r'\s+', ' ', name_raw).strip().strip('.')

            if len(name_raw) < 3 or len(name_raw) > 80:
                continue

            # Derive unit from name (e.g. "5ft", "100g", "piece", "pair", "pack")
            unit = "piece"
            name_low = name_raw.lower()
            if "5ft" in name_low or "5 ft" in name_low:
                unit = "box (5 ft)"
            elif "100g" in name_low or "100 g" in name_low:
                unit = "100g"
            elif "per foot" in name_low or "/ft" in name_low:
                unit = "per foot"
            elif "pair" in name_low:
                unit = "pair"
            elif "pack" in name_low:
                unit = "pack"
            elif "set" in name_low:
                unit = "set"

            # Normalise the key: lowercase, strip size suffixes for matching
            key = re.sub(r'\s*(5ft|5 ft|100g|per foot)\s*', ' ', name_raw, flags=re.IGNORECASE)
            key = key.strip().lower()

            results.append({"name": key, "price": current_price, "unit": unit, "display": name_raw})

    except Exception as e:
        print(f"[scrape_category error] {url}: {e}")

    return results


# ─────────────────────────────────────────────
# 4. KNOWLEDGE BASE + PRODUCT CATALOG
#    Both fetched live from indian-flowers.com
# ─────────────────────────────────────────────

# The flower/garland category URLs to scrape for products & prices
FLOWER_CATEGORY_URLS = [
    "https://indian-flowers.com/product-category/strings/",
    "https://indian-flowers.com/product-category/pooja-flowers/",
    "https://indian-flowers.com/product-category/green-leafs/",
    "https://indian-flowers.com/product-category/indian-wedding-garlands/",
    "https://indian-flowers.com/product-category/temple-and-pooja-garlands/",
    "https://indian-flowers.com/product-category/house-warming-garlands/",
    "https://indian-flowers.com/product-category/veni/",
    "https://indian-flowers.com/product-category/nanthiyavattam-garlands/",
    "https://indian-flowers.com/product-category/tube-rose-garlands/",
    "https://indian-flowers.com/product-category/coconut/",
]

FALLBACK_PRODUCTS = {
    "jasmine string":        {"price": 13.00,  "unit": "box (5 ft)"},
    "jathimalli string":     {"price": 15.00,  "unit": "box (5 ft)"},
    "mullai string":         {"price": 13.00,  "unit": "box (5 ft)"},
    "kathambam string":      {"price": 15.00,  "unit": "box (5 ft)"},
    "kanakambaram string":   {"price": 15.00,  "unit": "box (5 ft)"},
    "neem flowers":          {"price": 10.00,  "unit": "100g"},
    "jasmine flowers":       {"price": 10.00,  "unit": "100g"},
    "mullai loose":          {"price": 10.00,  "unit": "100g"},
    "kanakambaram":          {"price": 10.00,  "unit": "100g"},
    "lilly loose":           {"price": 8.00,   "unit": "100g"},
    "marigold loose":        {"price": 8.00,   "unit": "100g"},
    "yellow rose":           {"price": 6.00,   "unit": "100g"},
    "red rose":              {"price": 6.00,   "unit": "100g"},
    "arali pink":            {"price": 6.00,   "unit": "100g"},
    "arali red":             {"price": 6.00,   "unit": "100g"},
    "lotus flowers":         {"price": 2.00,   "unit": "piece"},
    "turmeric":              {"price": 8.00,   "unit": "each"},
    "avarampoo":             {"price": 10.00,  "unit": "100g"},
    "tulasi leaves":         {"price": 10.00,  "unit": "100g"},
    "neem leaf":             {"price": 10.00,  "unit": "100g"},
    "vilvam leaves":         {"price": 10.00,  "unit": "100g"},
    "erukkam leaves":        {"price": 10.00,  "unit": "100g"},
    "marikolunthu":          {"price": 10.00,  "unit": "100g"},
    "maruvam":               {"price": 10.00,  "unit": "100g"},
    "betel leaf":            {"price": 6.00,   "unit": "pack of 25"},
    "mango leaf":            {"price": 2.00,   "unit": "pack of 10"},
    "mango leaf thoranam":   {"price": 5.00,   "unit": "4 ft"},
    "banana leaf":           {"price": 10.00,  "unit": "pack"},
    "coconut thoranam":      {"price": 5.00,   "unit": "5 pieces"},
    "ala mokku":             {"price": 10.00,  "unit": "pack"},
    "marigold garland":      {"price": 20.00,  "unit": "per foot"},
    "rose petal garland":    {"price": 25.00,  "unit": "per foot (total both sides)"},
    "carnation garland":     {"price": 25.00,  "unit": "per foot (total both sides)"},
    "lilly garland":         {"price": 25.00,  "unit": "per foot (total both sides)"},
    "lily garland":          {"price": 25.00,  "unit": "per foot (total both sides)"},
    "exchange garland":      {"price": 80.00,  "unit": "pair"},
    "button rose garland":   {"price": 26.00,  "unit": "per foot (each side)"},
    "wedding garland":       {"price": 25.00,  "unit": "per foot (total both sides)"},
    "north indian garland":  {"price": 100.00, "unit": "pair"},
    "door garland":          {"price": 130.00, "unit": "10 ft set"},
    "house warming garland": {"price": 130.00, "unit": "10 ft set"},
    "veni":                  {"price": 15.00,  "unit": "piece"},
    "jadai":                 {"price": 15.00,  "unit": "piece"},
    "veni jadai":            {"price": 80.00,  "unit": "set"},
    "gajra":                 {"price": 80.00,  "unit": "set"},
    "temple garland":        {"price": 26.00,  "unit": "per foot"},
    "pooja garland":         {"price": 26.00,  "unit": "per foot"},
    "vadamalli garland":     {"price": 26.00,  "unit": "per foot"},
    "bouquet":               {"price": 8.00,   "unit": "piece"},
}


@st.cache_data(ttl=3600)
def load_knowledge_and_products() -> tuple[str, dict]:
    """
    Single Playwright session that:
      1. Scrapes the homepage for general knowledge text
      2. Visits each flower category page to extract live product prices
    Returns (website_info_str, products_dict)
    """
    website_info = ""
    products = {}

    try:
        from playwright.sync_api import sync_playwright
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()

            # ── Step 1: homepage knowledge text ──
            try:
                page.goto("https://indian-flowers.com/", timeout=30000)
                page.wait_for_load_state("networkidle", timeout=20000)
                content = page.inner_text("body")
                if content and len(content.strip()) > 100:
                    website_info = content[:6000]
            except Exception as e:
                print(f"[Homepage fetch error]: {e}")

            # ── Step 2: scrape each category for live prices ──
            seen_keys = set()
            for cat_url in FLOWER_CATEGORY_URLS:
                items = scrape_category(page, cat_url)
                for item in items:
                    key = item["name"]
                    if key not in seen_keys and len(key) >= 3:
                        products[key] = {"price": item["price"], "unit": item["unit"]}
                        seen_keys.add(key)

            browser.close()

    except Exception as e:
        print(f"[Playwright session error]: {e}")

    # Fallback if scraping returned nothing useful
    if len(products) < 5:
        print("[Products] Scraping returned too few products — using fallback.")
        products = FALLBACK_PRODUCTS

    if not website_info:
        website_info = (
            "The Indian Flowers USA (Malar Traders) delivers fresh Indian flowers "
            "and garlands nationwide across the USA. "
            "Products include jasmine strings, mullai strings, rose petal garlands, "
            "carnation garlands, wedding garlands, temple garlands, pooja garlands, "
            "marigold garlands, veni, jadai, gajra, and loose flowers. "
            "Shipping: Local $30, California $55, Nationwide $70. "
            "Payment via Zelle to Malar Traders only."
        )

    print(f"[Products] Loaded {len(products)} products from live website.")
    return website_info, products


# Load once, cached for 1 hour
WEBSITE_INFO, PRODUCTS = load_knowledge_and_products()

# Sidebar: show source
st.sidebar.success(f"✅ Live prices loaded from indian-flowers.com ({len(PRODUCTS)} products)")

# Sidebar: daily summary email
st.sidebar.divider()
st.sidebar.subheader("📧 Daily Order Summary")
todays_orders = st.session_state.get("all_confirmed_orders", [])
st.sidebar.metric("Chat Orders Today", len(todays_orders))
total_today = sum(o.get("grand_total", 0) for o in todays_orders)
if todays_orders:
    st.sidebar.metric("Total Revenue Today", f"${total_today:.2f}")

already_sent = st.session_state.get("email_sent_date") == get_today_str()
if already_sent:
    st.sidebar.success("✅ Summary email sent today")

if st.sidebar.button("📤 Send Summary Email Now", use_container_width=True):
    with st.sidebar:
        with st.spinner("Sending email..."):
            success = send_daily_email(todays_orders)
        if success:
            st.session_state["email_sent_date"] = get_today_str()
            st.success("✅ Email sent successfully!")
        else:
            st.error("❌ Failed — check Gmail credentials in Streamlit secrets")

PRODUCT_CATALOG_TEXT = "\n".join(
    f"  - {name.title()}: ${v['price']:.2f} per {v['unit']}"
    for name, v in PRODUCTS.items()
)

ORDER_KEYWORDS = list(PRODUCTS.keys()) + [
    "garland", "string", "flower", "veni", "jasmine", "rose", "carnation",
    "lily", "lilly", "mullai", "tulasi", "pooja", "temple", "wedding", "exchange",
    "marigold", "lotus", "gajra", "jadai", "bouquet", "door", "housewarming",
]

SKIP_WORDS = {
    "ok", "yes", "no", "confirm", "proceed", "hello", "hi", "hey", "thanks", "thank",
    "please", "sure", "great", "good", "need", "want", "order", "get", "i", "me", "my",
    "the", "and", "for", "a", "an", "is", "it", "in", "to", "of", "can", "do", "what",
    "box", "boxes", "piece", "pieces", "pair", "pairs", "bunch", "bunches", "set", "sets",
}

# ─────────────────────────────────────────────
# 5. SHIPPING CALCULATOR
# ─────────────────────────────────────────────
LOCAL_ZIPS = {
    "92335", "92336", "92337", "92316", "92324", "92376", "92377",
    "91710", "91761", "91762", "91763", "91764", "91766", "91767", "91768",
}
CA_PREFIXES = {
    "900", "901", "902", "903", "904", "905", "906", "907", "908",
    "910", "911", "912", "913", "914", "915", "916", "917", "918", "919",
    "920", "921", "922", "923", "924", "925", "926", "927", "928",
    "930", "931", "932", "933", "934", "935", "936", "937", "938", "939",
    "940", "941", "942", "943", "944", "945", "946", "947", "948", "949",
    "950", "951", "952", "953", "954", "955", "956", "957", "958", "959", "960", "961",
}

def calculate_shipping(zip_code: str, weight_lbs: float = 2.0) -> dict:
    z = zip_code.strip()
    if z in LOCAL_ZIPS:
        return {"fee": 30.00, "method": "Local Delivery (within 50 miles)",
                "note": "Delivered fresh same/next day."}
    if len(z) >= 3 and z[:3] in CA_PREFIXES:
        return {"fee": 55.00, "method": "California State Shipping",
                "note": "Delivered in 1-2 business days."}
    if weight_lbs > 15:
        return {"fee": 0.00, "method": "Southwest Cargo (Bulk Order)",
                "note": "Rate quoted separately."}
    return {"fee": 70.00, "method": "Nationwide USA Shipping",
            "note": "Delivered in 2-3 business days via overnight cold-pack."}

def parse_order_items(text: str) -> list:
    found, seen = [], set()
    low = text.lower()
    for name, info in PRODUCTS.items():
        if name in low and name not in seen:
            match = re.search(
                r'(\d+(?:\.\d+)?)\s*(?:feet|ft|foot|box(?:es)?|piece[s]?|pair[s]?|bunch(?:es)?|set[s]?)?\s*(?:of\s*)?' + re.escape(name),
                low
            )
            if not match:
                match = re.search(r'(\d+(?:\.\d+)?)\s*' + re.escape(name), low)
            qty = float(match.group(1)) if match else 1.0
            found.append({
                "name":       name,
                "qty":        qty,
                "unit_price": info["price"],
                "unit":       info["unit"],
            })
            seen.add(name)
    return found

def build_order_summary(items: list, zip_code: str) -> dict:
    subtotal    = sum(i["qty"] * i["unit_price"] for i in items)
    weight      = len(items) * 1.5
    shipping    = calculate_shipping(zip_code, weight)
    grand_total = subtotal + shipping["fee"]
    return {"items": items, "subtotal": subtotal,
            "shipping": shipping, "grand_total": grand_total}

def format_order_summary(summary: dict) -> str:
    lines = ["📦 ORDER BREAKDOWN (use EXACT these numbers):"]
    for item in summary["items"]:
        lines.append(
            f"  • {item['qty']} x {item['name'].title()} "
            f"@ ${item['unit_price']:.2f}/{item['unit']} = ${item['qty'] * item['unit_price']:.2f}"
        )
    lines.append(f"  Subtotal   : ${summary['subtotal']:.2f}")
    lines.append(f"  Shipping   : ${summary['shipping']['fee']:.2f} ({summary['shipping']['method']})")
    lines.append(f"  GRAND TOTAL: ${summary['grand_total']:.2f}")
    lines.append(f"  Delivery note: {summary['shipping']['note']}")
    return "\n".join(lines)

# ─────────────────────────────────────────────
# 6. LEAD FIELDS
# ─────────────────────────────────────────────
LEAD_FIELDS = ["name", "phone", "email", "zip_code"]
LEAD_QUESTIONS = {
    "name":     "May I have your full name please? 😊",
    "phone":    "Thank you! Could you share your phone number?",
    "email":    "Got it! What's your email address for order updates?",
    "zip_code": "Almost there! What's your delivery ZIP code so I can calculate shipping?",
}

def next_missing_field(memory: dict):
    for f in LEAD_FIELDS:
        if not memory.get(f):
            return f
    return None

# ─────────────────────────────────────────────
# 7. EXTRACTORS
# ─────────────────────────────────────────────
def regex_extract_lead(text: str, memory: dict) -> dict:
    updated = dict(memory)
    low = text.lower().strip()
    if not updated.get("email"):
        m = re.search(r'[\w.\-+]+@[\w\-]+\.[a-z]{2,}', text, re.IGNORECASE)
        if m:
            updated["email"] = m.group(0).lower()
    if not updated.get("phone"):
        digits = re.sub(r'\D', '', text)
        if len(digits) == 10:
            updated["phone"] = digits
        elif len(digits) == 11 and digits[0] == '1':
            updated["phone"] = digits[1:]
    if not updated.get("zip_code"):
        m = re.search(r'\b(\d{5})\b', text)
        if m:
            updated["zip_code"] = m.group(1)
    if not updated.get("name"):
        m = re.search(
            r'(?:my name is|i am|i\'m|this is|call me)\s+([A-Za-z ]{2,30})',
            text, re.IGNORECASE
        )
        if m:
            candidate = m.group(1).strip().rstrip(".,!?")
            if candidate.lower() not in SKIP_WORDS:
                updated["name"] = candidate.title()
        else:
            words = low.split()
            if (len(words) <= 3
                    and not re.search(r'\d', text)
                    and not any(w in SKIP_WORDS for w in words)
                    and not any(kw in low for kw in ORDER_KEYWORDS)):
                updated["name"] = text.strip().title()
    return updated

def extract_order_intent(text: str, memory: dict) -> dict:
    updated = dict(memory)
    low = text.lower()
    if any(kw in low for kw in ORDER_KEYWORDS):
        existing = updated.get("raw_order_text") or ""
        updated["raw_order_text"] = (existing + " " + text).strip()
    if not updated.get("occasion"):
        for occ in ["wedding", "pooja", "temple", "birthday", "engagement", "festival", "puja", "housewarming"]:
            if occ in low:
                updated["occasion"] = occ
                break
    return updated

def llm_extract_lead_sync(text: str, memory: dict) -> dict:
    updated = dict(memory)
    try:
        prompt = ChatPromptTemplate.from_template(
            """Extract customer info from this message. Reply ONLY with valid JSON, no explanation.
Message: "{message}"
Return exactly this JSON (null if not found):
{{"name": null, "phone": null, "email": null, "zip_code": null, "occasion": null}}
Rules:
- name: real person name only, never: ok/yes/no/confirm/hello/hi/thanks
- phone: 10-digit US number as plain digits
- email: valid email only
- zip_code: 5-digit US zip only"""
        )
        chain = prompt | extract_llm | StrOutputParser()
        raw   = chain.invoke({"message": text})
        raw   = re.sub(r"^```(?:json)?", "", raw.strip())
        raw   = re.sub(r"```$", "", raw).strip()
        m     = re.search(r'\{.*?\}', raw, re.DOTALL)
        if m:
            extracted = json.loads(m.group(0))
            for field in ["name", "phone", "email", "zip_code", "occasion"]:
                val = extracted.get(field)
                if val and str(val).strip().lower() not in ("null", "none", ""):
                    if not updated.get(field):
                        updated[field] = str(val).strip()
    except Exception as e:
        print(f"[LLM extract error]: {e}")
    return updated

def smart_extract_lead(text: str, memory: dict) -> dict:
    updated = regex_extract_lead(text, memory)
    missing = [f for f in LEAD_FIELDS if not updated.get(f)]
    if missing:
        updated = llm_extract_lead_sync(text, updated)
    return updated

# ─────────────────────────────────────────────
# 8. SYSTEM PROMPT & CHAIN
# ─────────────────────────────────────────────
SYSTEM_PROMPT = f"""You are Priya, a warm sales assistant for The Indian Flowers USA (Malar Traders).

YOUR ROLE:
- Help customers find the right flowers/garlands for their occasion
- Ask about their needs, occasion, and preferences warmly
- Present prices EXACTLY as given in the ORDER BREAKDOWN block — NEVER recalculate
- Keep replies SHORT — 2 to 4 sentences max
- Use the customer's name warmly
- Ask ONE question at a time only
- NEVER mention product codes

LIVE WEBSITE CONTENT (updated every hour):
{WEBSITE_INFO[:4000]}

PRODUCT CATALOG WITH PRICING (live from indian-flowers.com, updated every hour):
{PRODUCT_CATALOG_TEXT}

PRICING NOTES:
- Rose petal / carnation / lilly / wedding garlands: $25 per foot (total length both sides combined)
  Example: 4 ft total (2 ft each side) = $100. 5 ft total (2.5 ft each side) = $125.
- Button rose garland: $26 per foot PER SIDE
- Temple / pooja / vadamalli garland: $26 per foot
- Marigold garland: $20 per foot

SHIPPING RATES:
  - Local delivery (Fontana/Ontario CA, within 50 miles): $30.00
  - California state: $55.00
  - Nationwide USA: $70.00
  - Bulk over 15 lbs: Southwest Cargo (quoted separately)
  - NO local pickup — all orders shipped

PAYMENT: Zelle to "Malar Traders" only.

CRITICAL RULES:
- NEVER assume an order — always ask what the customer wants
- NEVER show a price unless ORDER BREAKDOWN is in context
- NEVER recalculate prices from the ORDER BREAKDOWN
- Keep replies SHORT and warm
"""

chat_prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}"),
])

conversation_chain = chat_prompt | chat_llm | StrOutputParser()

# ─────────────────────────────────────────────
# 9. ORDER SUMMARY HELPERS
# ─────────────────────────────────────────────
def order_confirmation_text(memory: dict) -> str:
    summary = build_order_summary(memory["order_items"], memory["zip_code"])
    memory["confirmed_summary"] = summary
    name  = memory.get("name", "there")
    lines = [f"Here's your order summary, **{name}**! 🌺\n"]
    for item in summary["items"]:
        lines.append(
            f"• {item['qty']} x {item['name'].title()} ({item['unit']}) "
            f"— **${item['qty'] * item['unit_price']:.2f}**"
        )
    lines.append(f"\n**Subtotal:** ${summary['subtotal']:.2f}")
    lines.append(f"**Shipping ({summary['shipping']['method']}):** ${summary['shipping']['fee']:.2f}")
    lines.append(f"**────────────────────────────**")
    lines.append(f"🧾 **Grand Total: ${summary['grand_total']:.2f}**")
    lines.append(f"\n_{summary['shipping']['note']}_")
    lines.append(f"\nType **confirm** to place your order, or **change** to modify. 😊")
    return "\n".join(lines)

def final_confirmation_text(memory: dict) -> str:
    summary = memory.get("confirmed_summary") or build_order_summary(
        memory["order_items"], memory["zip_code"]
    )
    name  = memory.get("name", "there")
    lines = [f"🎉 **Thank you, {name}! Your order is confirmed.**\n"]
    for item in summary["items"]:
        lines.append(
            f"• {item['qty']} x {item['name'].title()} "
            f"— **${item['qty'] * item['unit_price']:.2f}**"
        )
    lines.append(f"\n**Shipping ({summary['shipping']['method']}):** ${summary['shipping']['fee']:.2f}")
    lines.append(f"**Grand Total: ${summary['grand_total']:.2f}**")
    lines.append(f"\n💳 Please send **${summary['grand_total']:.2f}** via **Zelle to Malar Traders**.")
    lines.append("\n✅ Our team will send your receipt and delivery update shortly. Wishing you a beautiful celebration! 🌺")
    return "\n".join(lines)

def order_recap_text(memory: dict) -> str:
    summary = memory.get("confirmed_summary")
    if not summary:
        return "I don't have a confirmed order on file yet."
    name  = memory.get("name", "there")
    lines = [f"Here's your confirmed order recap, **{name}**! 🌺\n"]
    for item in summary["items"]:
        lines.append(
            f"• {item['qty']} x {item['name'].title()} ({item['unit']}) "
            f"— **${item['qty'] * item['unit_price']:.2f}**"
        )
    lines.append(f"\n**Subtotal:** ${summary['subtotal']:.2f}")
    lines.append(f"**Shipping ({summary['shipping']['method']}):** ${summary['shipping']['fee']:.2f}")
    lines.append(f"**────────────────────────────**")
    lines.append(f"🧾 **Grand Total: ${summary['grand_total']:.2f}**")
    lines.append(f"\n💳 Please send **${summary['grand_total']:.2f}** via **Zelle to Malar Traders**.")
    return "\n".join(lines)

# ─────────────────────────────────────────────
# 10. DAILY EMAIL SUMMARY
# ─────────────────────────────────────────────

def get_today_str() -> str:
    return datetime.datetime.now().strftime("%B %d, %Y")

def get_current_hour_et() -> int:
    """Return current hour in US Eastern time (UTC-4 or UTC-5)."""
    utc_now = datetime.datetime.utcnow()
    # EST = UTC-5, EDT = UTC-4 (approximate: use UTC-4 for simplicity)
    et_now = utc_now - datetime.timedelta(hours=4)
    return et_now.hour

def build_email_html(chat_orders: list) -> str:
    """Build a nicely formatted HTML email with all confirmed chat orders for today."""
    today = get_today_str()
    total_revenue = sum(
        o.get("grand_total", 0) for o in chat_orders
    )

    rows = ""
    if chat_orders:
        for i, order in enumerate(chat_orders, 1):
            summary = order.get("confirmed_summary", {})
            items_html = ""
            for item in summary.get("items", []):
                items_html += (
                    f"<tr>"
                    f"<td style='padding:4px 8px;'>{item['name'].title()}</td>"
                    f"<td style='padding:4px 8px;text-align:center;'>{item['qty']}</td>"
                    f"<td style='padding:4px 8px;'>{item['unit']}</td>"
                    f"<td style='padding:4px 8px;text-align:right;'>${item['qty'] * item['unit_price']:.2f}</td>"
                    f"</tr>"
                )
            shipping = summary.get("shipping", {})
            rows += f"""
            <div style="background:#fff8f0;border:1px solid #f0c08a;border-radius:8px;padding:16px;margin-bottom:16px;">
                <h3 style="margin:0 0 8px;color:#b45309;">Order #{i} — {order.get('name','N/A')}</h3>
                <p style="margin:2px 0;font-size:13px;color:#555;">
                    📞 {order.get('phone','—')} &nbsp;|&nbsp;
                    ✉️ {order.get('email','—')} &nbsp;|&nbsp;
                    📍 ZIP: {order.get('zip_code','—')}
                    {f"&nbsp;|&nbsp; 🎉 Occasion: {order.get('occasion','').title()}" if order.get('occasion') else ''}
                </p>
                <table style="width:100%;border-collapse:collapse;margin-top:10px;font-size:13px;">
                    <thead>
                        <tr style="background:#fde68a;">
                            <th style="padding:4px 8px;text-align:left;">Product</th>
                            <th style="padding:4px 8px;">Qty</th>
                            <th style="padding:4px 8px;text-align:left;">Unit</th>
                            <th style="padding:4px 8px;text-align:right;">Amount</th>
                        </tr>
                    </thead>
                    <tbody>{items_html}</tbody>
                </table>
                <p style="margin:8px 0 2px;font-size:13px;">
                    <strong>Subtotal:</strong> ${summary.get('subtotal', 0):.2f} &nbsp;|&nbsp;
                    <strong>Shipping ({shipping.get('method','')}):</strong> ${shipping.get('fee', 0):.2f}
                </p>
                <p style="margin:4px 0;font-size:15px;font-weight:bold;color:#16a34a;">
                    Grand Total: ${order.get('grand_total', 0):.2f}
                </p>
            </div>
            """
    else:
        rows = "<p style='color:#888;'>No confirmed orders via chat today.</p>"

    html = f"""
    <html><body style="font-family:Arial,sans-serif;background:#fafafa;padding:24px;">
        <div style="max-width:620px;margin:auto;background:white;border-radius:12px;
                    box-shadow:0 2px 8px rgba(0,0,0,0.08);overflow:hidden;">
            <div style="background:linear-gradient(135deg,#f97316,#ec4899);padding:24px;text-align:center;">
                <h1 style="color:white;margin:0;font-size:22px;">🌸 The Indian Flowers USA</h1>
                <p style="color:#ffe4e6;margin:4px 0 0;">Daily Order Summary — {today}</p>
            </div>
            <div style="padding:24px;">
                <h2 style="color:#92400e;border-bottom:2px solid #fde68a;padding-bottom:8px;">
                    💬 Chat Orders ({len(chat_orders)} confirmed)
                </h2>
                {rows}
                <div style="background:#f0fdf4;border:1px solid #86efac;border-radius:8px;
                            padding:16px;margin-top:16px;text-align:center;">
                    <p style="margin:0;font-size:18px;font-weight:bold;color:#15803d;">
                        Total Chat Revenue Today: ${total_revenue:.2f}
                    </p>
                </div>
                <hr style="margin:24px 0;border:none;border-top:1px solid #e5e7eb;">
                <p style="font-size:12px;color:#9ca3af;text-align:center;">
                    Auto-generated by Priya AI Agent · {today}<br>
                    💳 Payment: Zelle to Malar Traders
                </p>
            </div>
        </div>
    </body></html>
    """
    return html


def send_daily_email(chat_orders: list) -> bool:
    """Send the daily summary email via Gmail SMTP. Returns True on success."""
    try:
        gmail        = st.secrets.get("gmail", {})
        sender       = gmail.get("user", "")
        app_password = gmail.get("password", "")
        recipient    = gmail.get("owner", "")

        if not all([sender, app_password, recipient]):
            print("[Email] Missing Gmail credentials in secrets.")
            return False

        msg = MIMEMultipart("alternative")
        msg["Subject"] = f"🌸 Indian Flowers — Daily Orders Summary {get_today_str()}"
        msg["From"]    = f"Priya AI Agent <{sender}>"
        msg["To"]      = recipient

        html_body = build_email_html(chat_orders)
        msg.attach(MIMEText(html_body, "html"))

        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(sender, app_password)
            server.sendmail(sender, recipient, msg.as_string())

        print(f"[Email] Daily summary sent to {recipient}")
        return True

    except Exception as e:
        print(f"[Email error]: {e}")
        return False


def collect_todays_orders() -> list:
    """Pull all confirmed orders stored in today's session."""
    orders = []
    for entry in st.session_state.get("all_confirmed_orders", []):
        orders.append(entry)
    return orders


def maybe_auto_send_email():
    """Auto-send daily summary at 9 PM ET if not already sent today."""
    today = get_today_str()
    already_sent = st.session_state.get("email_sent_date") == today
    if already_sent:
        return
    hour = get_current_hour_et()
    if hour >= 21:  # 9 PM ET
        orders = collect_todays_orders()
        success = send_daily_email(orders)
        if success:
            st.session_state["email_sent_date"] = today


# ─────────────────────────────────────────────
# 10. SESSION STATE INIT
# ─────────────────────────────────────────────
def init_session():
    if "memory" not in st.session_state:
        st.session_state.memory = {
            "name": None, "phone": None, "email": None, "zip_code": None,
            "occasion": None, "raw_order_text": None, "order_items": [],
            "stage": "lead_capture", "order_confirmed": False,
            "confirmed_summary": None,
        }
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": (
                    "🌸 Welcome to **The Indian Flowers USA**!\n\n"
                    "I'm Priya, your personal flower assistant. "
                    "We deliver fresh jasmine, rose, carnation garlands and more "
                    "to your doorstep anywhere in the USA.\n\n"
                    "May I have your **full name** to get started? 😊"
                )
            }
        ]
    if "lc_history" not in st.session_state:
        st.session_state.lc_history = []
    if "all_confirmed_orders" not in st.session_state:
        st.session_state.all_confirmed_orders = []
    if "email_sent_date" not in st.session_state:
        st.session_state.email_sent_date = None

init_session()

# ── Auto-send check (runs on every page load/interaction) ──
maybe_auto_send_email()

# ─────────────────────────────────────────────
# 11. DISPLAY CHAT HISTORY
# ─────────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"], avatar="🌸" if msg["role"] == "assistant" else "👤"):
        st.markdown(msg["content"])

# ─────────────────────────────────────────────
# 12. HANDLE USER INPUT
# ─────────────────────────────────────────────
CONFIRM_TRIGGERS = {
    "confirm", "yes confirm", "proceed", "confirm order", "place order",
    "proceed with payment", "i confirm", "yes proceed", "go ahead",
    "finalize", "that's correct", "looks good", "looks correct", "yes",
}

if user_input := st.chat_input("Type your message here..."):
    memory  = st.session_state.memory
    lowered = user_input.lower().strip()
    stage   = memory.get("stage", "lead_capture")

    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user", avatar="👤"):
        st.markdown(user_input)

    reply = None

    # ── STAGE 1: LEAD CAPTURE ──────────────────────────────────────────
    if stage == "lead_capture":
        memory = smart_extract_lead(user_input, memory)
        missing = next_missing_field(memory)
        if missing:
            just_provided = memory.get(missing)
            if not just_provided:
                if missing == "name" and any(kw in lowered for kw in ORDER_KEYWORDS):
                    reply = f"🌸 We'd love to help with your order! First, {LEAD_QUESTIONS['name']}"
                else:
                    reply = LEAD_QUESTIONS[missing]
            else:
                next_field = next_missing_field(memory)
                if next_field:
                    reply = f"Thank you! {LEAD_QUESTIONS[next_field]}"
                else:
                    memory["stage"] = "need_discovery"
                    name = memory.get("name", "there")
                    reply = (
                        f"Wonderful, {name}! 🌺 Thank you for your details.\n\n"
                        "What can I help you with today? Are you looking for wedding garlands, "
                        "temple flowers, jasmine strings, or something else? And what's the occasion? 😊"
                    )
        else:
            memory["stage"] = "need_discovery"
            name = memory.get("name", "there")
            reply = (
                f"Wonderful, {name}! 🌺 Thank you for your details.\n\n"
                "What can I help you with today? Are you looking for wedding garlands, "
                "temple flowers, jasmine strings, or something else? And what's the occasion? 😊"
            )

    # ── STAGE 2: NEED DISCOVERY ────────────────────────────────────────
    elif stage == "need_discovery":
        memory = extract_order_intent(user_input, memory)
        if not memory.get("zip_code"):
            m = re.search(r'\b(\d{5})\b', user_input)
            if m:
                memory["zip_code"] = m.group(1)
        if memory.get("raw_order_text") or memory.get("occasion"):
            memory["stage"] = "order_building"

    # ── STAGE 3: ORDER BUILDING ────────────────────────────────────────
    if stage in ("order_building", "need_discovery") and reply is None:
        if stage == "order_building":
            memory = extract_order_intent(user_input, memory)
            if memory.get("raw_order_text"):
                items = parse_order_items(memory["raw_order_text"])
                if items:
                    memory["order_items"] = items

        if any(t in lowered for t in CONFIRM_TRIGGERS) and stage == "order_building":
            if memory.get("order_items") and memory.get("zip_code"):
                memory["stage"] = "order_confirm"
                reply = order_confirmation_text(memory)
            elif not memory.get("zip_code"):
                reply = "I need your delivery ZIP code to calculate shipping. What's your ZIP? 📦"
            else:
                reply = "I don't have your order details yet. What would you like to order? 🌸"

    # ── STAGE 4: ORDER CONFIRM ─────────────────────────────────────────
    elif stage == "order_confirm" and reply is None:
        if any(t in lowered for t in CONFIRM_TRIGGERS):
            memory["stage"]           = "order_done"
            memory["order_confirmed"] = True
            reply = final_confirmation_text(memory)
            # ── Save to today's order log ──
            summary = memory.get("confirmed_summary", {})
            st.session_state.all_confirmed_orders.append({
                "name":              memory.get("name"),
                "phone":             memory.get("phone"),
                "email":             memory.get("email"),
                "zip_code":          memory.get("zip_code"),
                "occasion":          memory.get("occasion"),
                "confirmed_summary": summary,
                "grand_total":       summary.get("grand_total", 0),
                "confirmed_at":      datetime.datetime.now().strftime("%I:%M %p"),
            })
        elif any(t in lowered for t in ["no", "change", "wrong", "different", "edit", "modify"]):
            memory["stage"]          = "order_building"
            memory["order_items"]    = []
            memory["raw_order_text"] = None
            reply = "No problem! Let's update your order. What would you like to change? 😊"

    # ── STAGE 5: ORDER DONE ────────────────────────────────────────────
    elif stage == "order_done" and reply is None:
        price_words = {"price", "pricing", "total", "cost", "how much", "amount", "summary", "breakdown"}
        if any(w in lowered for w in price_words):
            reply = order_recap_text(memory)
        else:
            reply = (
                f"Your order is confirmed, {memory.get('name', 'there')}! 🌺 "
                "Please send payment via Zelle to **Malar Traders**. "
                "Is there anything else I can help you with?"
            )

    # ── LLM RESPONSE ───────────────────────────────────────────────────
    if reply is None:
        context_parts = [
            f"=== CUSTOMER INFO ===",
            f"Name    : {memory.get('name')}",
            f"Phone   : {memory.get('phone')}",
            f"Email   : {memory.get('email')}",
            f"ZIP     : {memory.get('zip_code')}",
            f"Stage   : {memory.get('stage')}",
        ]
        if memory.get("occasion"):
            context_parts.append(f"Occasion: {memory['occasion']}")
        if memory.get("raw_order_text"):
            context_parts.append(f"Order so far: \"{memory['raw_order_text']}\"")
        if memory.get("order_items") and memory.get("zip_code"):
            summary = build_order_summary(memory["order_items"], memory["zip_code"])
            context_parts.append("\n" + format_order_summary(summary))
            context_parts.append(
                "\nINSTRUCTION: Present the ORDER BREAKDOWN above and ask customer to type 'confirm'."
            )
        else:
            context_parts.append(
                "\nINSTRUCTION: Ask warmly what flowers/garlands they need and for what occasion. "
                "Get specifics: product type, quantity, size. "
                "Once you understand the order, ask them to say 'confirm'."
            )

        augmented_input = "\n".join(context_parts) + f"\n\n---\nCustomer: {user_input}"

        with st.chat_message("assistant", avatar="🌸"):
            with st.spinner("Priya is thinking..."):
                try:
                    response = conversation_chain.invoke({
                        "history": st.session_state.lc_history,
                        "input":   augmented_input,
                    })
                    reply = response
                except Exception as e:
                    reply = f"⚠️ {str(e)}"
                    print(f"[Chain error]: {e}")
            st.markdown(reply)

        st.session_state.lc_history.append(HumanMessage(content=user_input))
        st.session_state.lc_history.append(AIMessage(content=reply))
        if len(st.session_state.lc_history) > 20:
            st.session_state.lc_history = st.session_state.lc_history[-20:]

    else:
        with st.chat_message("assistant", avatar="🌸"):
            st.markdown(reply)

    st.session_state.messages.append({"role": "assistant", "content": reply})
    st.session_state.memory = memory
