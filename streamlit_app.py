import os
import re
import json
import subprocess
import sys
import smtplib
import datetime
import requests                          # ← NEW: for WooCommerce REST API
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
# 2. LLM SETUP
# ─────────────────────────────────────────────
@st.cache_resource
def get_llms():
    try:
        groq_api_key = st.secrets["groq"]["api_key"]
    except Exception:
        try:
            groq_api_key = st.secrets["GROQ_API_KEY"]
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
# 3. WOOCOMMERCE CREDENTIALS
# ─────────────────────────────────────────────
def get_wc_creds():
    try:
        wc = st.secrets["woocommerce"]
        return wc["url"].rstrip("/"), wc["consumer_key"], wc["consumer_secret"]
    except Exception:
        return (
            os.environ.get("WC_URL", ""),
            os.environ.get("WC_CONSUMER_KEY", ""),
            os.environ.get("WC_CONSUMER_SECRET", ""),
        )

WC_URL, WC_KEY, WC_SECRET = get_wc_creds()

# Basic Auth header — needed to access private + all products
import base64
def get_wc_auth_header():
    credentials = f"{WC_KEY}:{WC_SECRET}"
    encoded = base64.b64encode(credentials.encode()).decode()
    return {"Authorization": f"Basic {encoded}"}

# ─────────────────────────────────────────────
# 4. WOOCOMMERCE — LIVE PRODUCTS
#    Fetches all published products with price,
#    stock status, categories from the REST API.
#    Cached 1 hour so it doesn't hammer the API.
# ─────────────────────────────────────────────
@st.cache_data(ttl=1)
def fetch_wc_products() -> dict:
    """
    Returns dict: { "product name lowercase": {"price": float, "unit": str, "stock": str} }
    Falls back to FALLBACK_PRODUCTS if API is unreachable.
    """
    if not WC_URL or not WC_KEY:
        print("[WC Products] No credentials — using fallback.")
        return {}

    products  = {}
    seen_keys = set()   # avoid city-duplicate products
    page      = 1

    while page <= 30:   # safety cap — 30 pages × 100 = 3000 products max
        try:
            resp = requests.get(
                f"{WC_URL}/wp-json/wc/v3/products",
                headers=get_wc_auth_header(),        # ← Basic Auth (reads private too)
                params={
                    "per_page": 100,
                    "page":     page,
                    "status":   "publish,private",   # ← includes private 577
                },
                timeout=20,
            )
            if resp.status_code != 200:
                print(f"[WC Products] API error {resp.status_code}: {resp.text[:200]}")
                break
            batch = resp.json()
            if not batch:
                break

            for p in batch:
                raw_name = p.get("name", "").strip()
                if not raw_name:
                    continue

                # ── Extract product code like LG21, RP35, WG10 ──
                code_match   = re.search(r':?\s*([A-Z]{1,3}\d{2,4})\s*$', raw_name, re.IGNORECASE)
                product_code = code_match.group(1).upper() if code_match else None

                # ── Deduplicate city SEO variants ──
                # e.g. "Jasmine String Fort Wayne", "Jasmine String Charlotte" → same product
                if product_code:
                    base_name = raw_name   # keep full name with code e.g. "Lilly Wedding Garland: LG21"
                else:
                    base_name = re.sub(
                        r'\s+(in\s+)?('
                        r'fort wayne|charlotte|irvine|cerritos|atlanta|chicago|'
                        r'los angeles|milpitas|huntington beach|north las vegas|'
                        r'burbank|covington|destin|arlington|usa'
                        r')\s*$',
                        '', raw_name, flags=re.IGNORECASE
                    ).strip()

                name = base_name.lower()

                # Skip duplicates
                if name in seen_keys:
                    continue

                price_str = p.get("price") or p.get("regular_price") or "0"
                try:
                    price = float(price_str)
                except ValueError:
                    price = 0.0
                if price <= 0:
                    continue

                # ── Derive unit from name ──
                unit     = "piece"
                name_low = name
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

                products[name] = {
                    "price": price,
                    "unit":  unit,
                    "stock": p.get("stock_status", "instock"),
                    "code":  product_code,   # ← store LG21, RP35 etc.
                }
                seen_keys.add(name)

            page += 1
            if len(batch) < 100:
                break

        except Exception as e:
            print(f"[WC Products] Request error: {e}")
            break

    print(f"[WC Products] Loaded {len(products)} unique products from API.")
    return products


# ─────────────────────────────────────────────
# 5. WOOCOMMERCE — FETCH TODAY'S WEBSITE ORDERS
#    Pulls orders placed directly on the website
#    (not via chat) for inclusion in daily email.
# ─────────────────────────────────────────────
def fetch_wc_orders_today() -> list:
    """
    Returns list of WooCommerce order dicts placed today.
    Each dict has: id, customer name, email, phone, total,
    items, shipping address, status.
    """
    if not WC_URL or not WC_KEY:
        return []

    today_start = datetime.datetime.now().replace(
        hour=0, minute=0, second=0, microsecond=0
    ).isoformat()

    results = []
    try:
        resp = requests.get(
            f"{WC_URL}/wp-json/wc/v3/orders",
            params={
                "consumer_key":    WC_KEY,
                "consumer_secret": WC_SECRET,
                "after":           today_start,
                "per_page":        100,
                "status":          "processing,completed,pending",
            },
            timeout=15,
        )
        if resp.status_code == 200:
            for order in resp.json():
                billing  = order.get("billing", {})
                line_items = [
                    {
                        "name":       item.get("name", ""),
                        "qty":        item.get("quantity", 1),
                        "unit_price": float(item.get("price", 0)),
                        "total":      float(item.get("total", 0)),
                    }
                    for item in order.get("line_items", [])
                ]
                results.append({
                    "order_id":   order.get("id"),
                    "status":     order.get("status"),
                    "name":       f"{billing.get('first_name','')} {billing.get('last_name','')}".strip(),
                    "email":      billing.get("email", ""),
                    "phone":      billing.get("phone", ""),
                    "zip_code":   billing.get("postcode", ""),
                    "city":       billing.get("city", ""),
                    "state":      billing.get("state", ""),
                    "items":      line_items,
                    "total":      float(order.get("total", 0)),
                    "created_at": order.get("date_created", ""),
                })
        else:
            print(f"[WC Orders] API error {resp.status_code}")
    except Exception as e:
        print(f"[WC Orders] Request error: {e}")

    print(f"[WC Orders] Found {len(results)} website orders today.")
    return results


# ─────────────────────────────────────────────
# 6. WOOCOMMERCE — FETCH CATEGORIES
#    Used to enrich the AI knowledge base with
#    live category names and descriptions.
# ─────────────────────────────────────────────
@st.cache_data(ttl=3600)
def fetch_wc_categories() -> str:
    """Returns a text summary of all product categories for the AI system prompt."""
    if not WC_URL or not WC_KEY:
        return ""
    try:
        resp = requests.get(
            f"{WC_URL}/wp-json/wc/v3/products/categories",
            params={
                "consumer_key":    WC_KEY,
                "consumer_secret": WC_SECRET,
                "per_page":        100,
                "hide_empty":      True,
            },
            timeout=15,
        )
        if resp.status_code == 200:
            cats = resp.json()
            lines = ["PRODUCT CATEGORIES (live from website):"]
            for c in cats:
                desc = re.sub(r'<[^>]+>', '', c.get("description", "")).strip()
                line = f"  - {c['name']} ({c.get('count', 0)} products)"
                if desc:
                    line += f": {desc[:100]}"
                lines.append(line)
            return "\n".join(lines)
    except Exception as e:
        print(f"[WC Categories] Error: {e}")
    return ""


# ─────────────────────────────────────────────
# 7. SCRAPER HELPER — fetch /shop + homepage
#    Playwright is KEPT for rich page knowledge:
#    banners, announcements, /shop descriptions.
#    It is NOT used for pricing anymore (API does that).
# ─────────────────────────────────────────────
def scrape_shop_knowledge(page) -> str:
    """
    Visits /shop and homepage to extract general knowledge text:
    announcements, delivery info, banners, policies.
    NOT used for pricing — WooCommerce API handles that.
    """
    knowledge = ""
    urls_to_scrape = [
        "https://indian-flowers.com/",
        "https://indian-flowers.com/shop/",
    ]
    for url in urls_to_scrape:
        try:
            page.goto(url, timeout=30000)
            page.wait_for_load_state("networkidle", timeout=20000)
            content = page.inner_text("body")
            if content and len(content.strip()) > 100:
                knowledge += content[:3000] + "\n\n"
        except Exception as e:
            print(f"[Playwright scrape error] {url}: {e}")
    return knowledge.strip()


# ─────────────────────────────────────────────
# 8. FALLBACK PRODUCTS (if API is down)
# ─────────────────────────────────────────────
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


# ─────────────────────────────────────────────
# 9. LOAD KNOWLEDGE BASE + PRODUCTS
#    Playwright  → /shop + homepage knowledge text
#    WooCommerce API → live prices + categories
# ─────────────────────────────────────────────
@st.cache_data(ttl=3600)
def load_knowledge_and_products() -> tuple[str, dict, str]:
    """
    Returns (website_info_str, products_dict, categories_str)
    """
    website_info = ""
    products     = {}

    # ── Step 1: WooCommerce API for live prices ──
    products = fetch_wc_products()
    if len(products) < 5:
        print("[Products] API returned too few — using fallback.")
        products = FALLBACK_PRODUCTS

    # ── Step 2: WooCommerce API for categories ──
    categories_text = fetch_wc_categories()

    # ── Step 3: Playwright for /shop + homepage rich text ──
    try:
        from playwright.sync_api import sync_playwright
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            pw_page = browser.new_page()
            website_info = scrape_shop_knowledge(pw_page)
            browser.close()
    except Exception as e:
        print(f"[Playwright session error]: {e}")

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

    print(f"[Knowledge] Loaded. Products: {len(products)}, Categories scraped: {bool(categories_text)}")
    return website_info, products, categories_text


# Load once, cached for 1 hour
WEBSITE_INFO, PRODUCTS, CATEGORIES_TEXT = load_knowledge_and_products()

# ── Product code lookup map: {"LG21": "lilly wedding garland: lg21"} ──
PRODUCT_CODE_MAP = {
    v["code"]: name
    for name, v in PRODUCTS.items()
    if v.get("code")
}

PRODUCT_CATALOG_TEXT = "\n".join(
    f"  - {name.title()}"
    + (f" [Code: {v['code']}]" if v.get("code") else "")
    + f": ${v['price']:.2f} per {v['unit']}"
    + (" [OUT OF STOCK]" if v.get("stock") == "outofstock" else "")
    for name, v in PRODUCTS.items()
)

ORDER_KEYWORDS = list(PRODUCTS.keys()) + list(PRODUCT_CODE_MAP.keys()) + [
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
# 10. SHIPPING CALCULATOR
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

    # ── Step 1: match product codes like LG21, RP35, WG10 ──
    code_hits = re.findall(r'\b([A-Z]{1,3}\d{2,4})\b', text, re.IGNORECASE)
    for code in code_hits:
        mapped_name = PRODUCT_CODE_MAP.get(code.upper())
        if mapped_name and mapped_name not in seen:
            info      = PRODUCTS[mapped_name]
            qty_match = re.search(r'(\d+)\s*(?:x\s*)?' + re.escape(code), text, re.IGNORECASE)
            qty       = float(qty_match.group(1)) if qty_match else 1.0
            found.append({
                "name":       mapped_name,
                "qty":        qty,
                "unit_price": info["price"],
                "unit":       info["unit"],
            })
            seen.add(mapped_name)

    # ── Step 2: match by full product name as before ──
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
# 11. LEAD FIELDS
# ─────────────────────────────────────────────
LEAD_FIELDS    = ["name", "phone", "email", "zip_code"]
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
# 12. EXTRACTORS
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
# 13. ENQUIRY TRACKER
#     Records every customer question/topic so
#     the daily email can summarise what people
#     asked about, even if they didn't order.
# ─────────────────────────────────────────────
def track_enquiry(user_text: str, memory: dict):
    """
    ONE enquiry record per customer per session.
    Creates the record on the first meaningful message,
    then updates topics/message as the conversation continues.
    Skips filler / confirmation / lead-capture-only messages entirely.
    """
    if "enquiries" not in st.session_state:
        st.session_state.enquiries = []

    low = user_text.lower().strip()

    # ── Skip filler / confirmation words ──
    IGNORE_PHRASES = {
        "ok", "okay", "yes", "no", "confirm", "proceed", "sure", "great",
        "thanks", "thank you", "thank", "hi", "hello", "hey", "fine",
        "got it", "sounds good", "looks good", "go ahead", "yes confirm",
        "i confirm", "yes proceed", "finalize", "that's correct",
        "looks correct", "place order", "confirm order",
    }
    if low in IGNORE_PHRASES:
        return

    # ── Skip pure lead-capture answers (phone / email / zip) ──
    if re.fullmatch(r'[\d\s\-\(\)\+]{7,15}', user_text.strip()):
        return
    if re.fullmatch(r'[\w.\-+]+@[\w\-]+\.[a-z]{2,}', user_text.strip()):
        return
    if re.fullmatch(r'\d{5}', user_text.strip()):
        return

    # ── Skip short messages with no product/occasion keyword ──
    words = low.split()
    has_product_keyword  = any(kw in low for kw in ORDER_KEYWORDS if len(kw) > 3)
    has_occasion_keyword = any(kw in low for kw in [
        "wedding", "pooja", "temple", "birthday", "engagement",
        "housewarming", "festival", "puja"
    ])
    if len(words) < 4 and not has_product_keyword and not has_occasion_keyword:
        return

    # ── Collect topics from this message ──
    new_topics = []
    for kw in ["wedding", "pooja", "temple", "birthday", "engagement",
               "housewarming", "festival", "puja"]:
        if kw in low:
            new_topics.append(kw)
    for kw in ORDER_KEYWORDS[:30]:
        if kw in low and len(kw) > 4:
            new_topics.append(kw)

    customer_key = memory.get("name", "Unknown")

    # ── Find existing enquiry for this customer and UPDATE it (no duplicates) ──
    for existing in st.session_state.enquiries:
        if existing["customer"] == customer_key:
            existing["topics"]  = list(set(existing["topics"] + new_topics)) or ["general inquiry"]
            existing["message"] = user_text[:200]
            existing["time"]    = datetime.datetime.now().strftime("%I:%M %p")
            return

    # ── No existing record — create one (first meaningful message only) ──
    st.session_state.enquiries.append({
        "time":     datetime.datetime.now().strftime("%I:%M %p"),
        "customer": customer_key,
        "message":  user_text[:200],
        "topics":   list(set(new_topics)) or ["general inquiry"],
    })


def build_enquiry_summary_text() -> str:
    """Builds a readable plain-text summary of all enquiries for the email."""
    enquiries = st.session_state.get("enquiries", [])
    if not enquiries:
        return "No enquiries recorded today."

    # Group by customer
    by_customer = {}
    for e in enquiries:
        c = e["customer"]
        if c not in by_customer:
            by_customer[c] = []
        by_customer[c].append(e)

    lines = []
    for customer, msgs in by_customer.items():
        lines.append(f"\n👤 {customer}:")
        for m in msgs:
            topics_str = ", ".join(m["topics"])
            lines.append(f"   [{m['time']}] Topics: {topics_str}")
            lines.append(f"   Message: {m['message']}")
    return "\n".join(lines)


# ─────────────────────────────────────────────
# 14. SYSTEM PROMPT & CHAIN
# ─────────────────────────────────────────────
SYSTEM_PROMPT = f"""You are Priya, a warm sales assistant for The Indian Flowers USA (Malar Traders).

YOUR ROLE:
- Help customers find the right flowers/garlands they need
- Ask about what products they want and how many — keep it simple and direct
- Present prices EXACTLY as given in the ORDER BREAKDOWN block — NEVER recalculate
- Keep replies SHORT — 2 to 4 sentences max
- Use the customer's name warmly
- Ask ONE question at a time only
- NEVER mention product codes
- NEVER ask about the occasion — customers just want to order flowers
- If a product shows [OUT OF STOCK], apologize and suggest a similar available product

LIVE WEBSITE CONTENT (updated every hour):
{WEBSITE_INFO[:3000]}

{CATEGORIES_TEXT}

PRODUCT CATALOG WITH PRICING (live from WooCommerce API, updated every hour):
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
- NEVER ask about occasion or purpose of the order
- NEVER show a price unless ORDER BREAKDOWN is in context
- NEVER recalculate prices from the ORDER BREAKDOWN
- NEVER recalculate or adjust shipping — the system handles shipping automatically
- NEVER offer local pickup — all orders are shipped, no exceptions
- If a customer mentions local pickup, politely inform them all orders are shipped only
- Keep replies SHORT and warm
"""

chat_prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}"),
])

conversation_chain = chat_prompt | chat_llm | StrOutputParser()

# ─────────────────────────────────────────────
# 15. ORDER SUMMARY HELPERS
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
    # ALWAYS use the locked summary shown to the customer — NEVER recalculate
    summary = memory.get("confirmed_summary")
    if not summary:
        summary = build_order_summary(memory["order_items"], memory["zip_code"])
        memory["confirmed_summary"] = summary   # lock it immediately
    name  = memory.get("name", "there")
    lines = [f"🎉 **Thank you, {name}! Your order is confirmed.**\n"]
    for item in summary["items"]:
        lines.append(
            f"• {item['qty']} x {item['name'].title()} "
            f"— **${item['qty'] * item['unit_price']:.2f}**"
        )
    lines.append(f"\n**Subtotal:** ${summary['subtotal']:.2f}")
    lines.append(f"**Shipping ({summary['shipping']['method']}):** ${summary['shipping']['fee']:.2f}")
    lines.append(f"**────────────────────────────**")
    lines.append(f"🧾 **Grand Total: ${summary['grand_total']:.2f}**")
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
# 16. DAILY EMAIL — WITH 3 SECTIONS
#     Section 1: Website orders (WooCommerce API)
#     Section 2: Chat orders (AI agent confirmed)
#     Section 3: Chat enquiries summary
# ─────────────────────────────────────────────
def get_today_str() -> str:
    return datetime.datetime.now().strftime("%B %d, %Y")

def get_current_hour_et() -> int:
    utc_now = datetime.datetime.utcnow()
    et_now  = utc_now - datetime.timedelta(hours=4)
    return et_now.hour


def build_website_orders_html(website_orders: list) -> str:
    """HTML block for Section 1 — orders placed directly on the website."""
    if not website_orders:
        return "<p style='color:#888;'>No website orders found today via WooCommerce API.</p>"

    rows = ""
    for order in website_orders:
        items_html = "".join(
            f"<tr>"
            f"<td style='padding:4px 8px;'>{item['name']}</td>"
            f"<td style='padding:4px 8px;text-align:center;'>{item['qty']}</td>"
            f"<td style='padding:4px 8px;text-align:right;'>${item['total']:.2f}</td>"
            f"</tr>"
            for item in order.get("items", [])
        )
        rows += f"""
        <div style="background:#f0f9ff;border:1px solid #93c5fd;border-radius:8px;padding:14px;margin-bottom:12px;">
            <h3 style="margin:0 0 6px;color:#1e40af;">
                🛒 Order #{order['order_id']} — {order['name']}
                <span style="font-size:11px;font-weight:normal;color:#6b7280;margin-left:8px;">
                    [{order['status'].upper()}]
                </span>
            </h3>
            <p style="margin:2px 0;font-size:13px;color:#555;">
                📞 {order.get('phone','—')} &nbsp;|&nbsp;
                ✉️ {order.get('email','—')} &nbsp;|&nbsp;
                📍 {order.get('city','')}, {order.get('state','')} {order.get('zip_code','')}
            </p>
            <table style="width:100%;border-collapse:collapse;margin-top:8px;font-size:13px;">
                <thead>
                    <tr style="background:#bfdbfe;">
                        <th style="padding:4px 8px;text-align:left;">Product</th>
                        <th style="padding:4px 8px;">Qty</th>
                        <th style="padding:4px 8px;text-align:right;">Amount</th>
                    </tr>
                </thead>
                <tbody>{items_html}</tbody>
            </table>
            <p style="margin:8px 0 0;font-size:15px;font-weight:bold;color:#1d4ed8;">
                Order Total: ${order['total']:.2f}
            </p>
        </div>
        """
    return rows


def build_chat_orders_html(chat_orders: list) -> str:
    """HTML block for Section 2 — orders confirmed through Priya AI chat."""
    if not chat_orders:
        return "<p style='color:#888;'>No confirmed orders via chat today.</p>"

    rows = ""
    for i, order in enumerate(chat_orders, 1):
        summary    = order.get("confirmed_summary", {})
        items_html = "".join(
            f"<tr>"
            f"<td style='padding:4px 8px;'>{item['name'].title()}</td>"
            f"<td style='padding:4px 8px;text-align:center;'>{item['qty']}</td>"
            f"<td style='padding:4px 8px;'>{item['unit']}</td>"
            f"<td style='padding:4px 8px;text-align:right;'>${item['qty'] * item['unit_price']:.2f}</td>"
            f"</tr>"
            for item in summary.get("items", [])
        )
        shipping = summary.get("shipping", {})
        rows += f"""
        <div style="background:#fff8f0;border:1px solid #f0c08a;border-radius:8px;padding:14px;margin-bottom:12px;">
            <h3 style="margin:0 0 6px;color:#b45309;">
                💬 Chat Order #{i} — {order.get('name','N/A')}
                <span style="font-size:11px;font-weight:normal;color:#6b7280;margin-left:8px;">
                    [{order.get('confirmed_at','')}]
                </span>
            </h3>
            <p style="margin:2px 0;font-size:13px;color:#555;">
                📞 {order.get('phone','—')} &nbsp;|&nbsp;
                ✉️ {order.get('email','—')} &nbsp;|&nbsp;
                📍 ZIP: {order.get('zip_code','—')}
                {f"&nbsp;|&nbsp; 🎉 {order.get('occasion','').title()}" if order.get('occasion') else ''}
            </p>
            <table style="width:100%;border-collapse:collapse;margin-top:8px;font-size:13px;">
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
                <strong>Subtotal:</strong> ${summary.get('subtotal',0):.2f} &nbsp;|&nbsp;
                <strong>Shipping ({shipping.get('method','')}):</strong> ${shipping.get('fee',0):.2f}
            </p>
            <p style="margin:4px 0;font-size:15px;font-weight:bold;color:#16a34a;">
                Grand Total: ${order.get('grand_total',0):.2f}
            </p>
        </div>
        """
    return rows


def build_enquiries_html(enquiries: list) -> str:
    """HTML block for Section 3 — chat enquiries summary."""
    if not enquiries:
        return "<p style='color:#888;'>No enquiries recorded today.</p>"

    # Group by customer
    by_customer = {}
    for e in enquiries:
        c = e["customer"]
        if c not in by_customer:
            by_customer[c] = []
        by_customer[c].append(e)

    rows = ""
    for customer, msgs in by_customer.items():
        all_topics = []
        for m in msgs:
            all_topics.extend(m.get("topics", []))
        unique_topics = list(set(all_topics))

        msg_rows = "".join(
            f"<tr>"
            f"<td style='padding:3px 8px;color:#6b7280;font-size:12px;'>{m['time']}</td>"
            f"<td style='padding:3px 8px;font-size:13px;'>{m['message']}</td>"
            f"</tr>"
            for m in msgs[:5]   # show max 5 messages per customer
        )
        rows += f"""
        <div style="background:#f9fafb;border:1px solid #d1d5db;border-radius:8px;padding:12px;margin-bottom:10px;">
            <h4 style="margin:0 0 4px;color:#374151;">👤 {customer}</h4>
            <p style="margin:0 0 6px;font-size:12px;color:#6b7280;">
                Interested in: <strong>{', '.join(unique_topics) or 'general inquiry'}</strong>
                &nbsp;|&nbsp; {len(msgs)} message(s)
            </p>
            <table style="width:100%;border-collapse:collapse;font-size:12px;">
                {msg_rows}
            </table>
        </div>
        """
    return rows


def build_email_html(chat_orders: list, website_orders: list, enquiries: list) -> str:
    """
    Full HTML email with 3 sections:
      1. Website Orders (WooCommerce API)
      2. Chat Orders (AI agent confirmed)
      3. Chat Enquiries Summary
    """
    today         = get_today_str()
    chat_revenue  = sum(o.get("grand_total", 0) for o in chat_orders)
    web_revenue   = sum(o.get("total", 0) for o in website_orders)
    total_revenue = chat_revenue + web_revenue

    html = f"""
    <html><body style="font-family:Arial,sans-serif;background:#fafafa;padding:24px;">
    <div style="max-width:640px;margin:auto;background:white;border-radius:12px;
                box-shadow:0 2px 8px rgba(0,0,0,0.08);overflow:hidden;">

        <!-- HEADER -->
        <div style="background:linear-gradient(135deg,#f97316,#ec4899);padding:24px;text-align:center;">
            <h1 style="color:white;margin:0;font-size:22px;">🌸 The Indian Flowers USA</h1>
            <p style="color:#ffe4e6;margin:4px 0 0;">Daily Summary Report — {today}</p>
        </div>

        <div style="padding:24px;">

            <!-- REVENUE SUMMARY BAR -->
            <div style="display:flex;gap:12px;margin-bottom:24px;">
                <div style="flex:1;background:#f0fdf4;border:1px solid #86efac;border-radius:8px;
                            padding:12px;text-align:center;">
                    <p style="margin:0;font-size:12px;color:#6b7280;">Website Orders</p>
                    <p style="margin:4px 0 0;font-size:18px;font-weight:bold;color:#15803d;">
                        ${web_revenue:.2f}
                    </p>
                    <p style="margin:0;font-size:11px;color:#9ca3af;">{len(website_orders)} order(s)</p>
                </div>
                <div style="flex:1;background:#fff7ed;border:1px solid #fdba74;border-radius:8px;
                            padding:12px;text-align:center;">
                    <p style="margin:0;font-size:12px;color:#6b7280;">Chat Orders</p>
                    <p style="margin:4px 0 0;font-size:18px;font-weight:bold;color:#c2410c;">
                        ${chat_revenue:.2f}
                    </p>
                    <p style="margin:0;font-size:11px;color:#9ca3af;">{len(chat_orders)} order(s)</p>
                </div>
                <div style="flex:1;background:#eff6ff;border:1px solid #93c5fd;border-radius:8px;
                            padding:12px;text-align:center;">
                    <p style="margin:0;font-size:12px;color:#6b7280;">Total Revenue</p>
                    <p style="margin:4px 0 0;font-size:18px;font-weight:bold;color:#1d4ed8;">
                        ${total_revenue:.2f}
                    </p>
                    <p style="margin:0;font-size:11px;color:#9ca3af;">{len(enquiries)} enquiries</p>
                </div>
            </div>

            <!-- SECTION 1: WEBSITE ORDERS -->
            <h2 style="color:#1e40af;border-bottom:2px solid #bfdbfe;padding-bottom:8px;">
                🛒 Section 1 — Website Orders ({len(website_orders)})
            </h2>
            {build_website_orders_html(website_orders)}

            <!-- SECTION 2: CHAT ORDERS -->
            <h2 style="color:#92400e;border-bottom:2px solid #fde68a;padding-bottom:8px;margin-top:28px;">
                💬 Section 2 — Chat Orders ({len(chat_orders)})
            </h2>
            {build_chat_orders_html(chat_orders)}

            <!-- SECTION 3: ENQUIRIES SUMMARY -->
            <h2 style="color:#374151;border-bottom:2px solid #d1d5db;padding-bottom:8px;margin-top:28px;">
                🔍 Section 3 — Chat Enquiries Summary ({len(enquiries)})
            </h2>
            <p style="font-size:13px;color:#6b7280;margin-top:0;">
                Customers who enquired but may not have ordered yet — potential follow-ups.
            </p>
            {build_enquiries_html(enquiries)}

            <!-- FOOTER -->
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


def send_daily_email(chat_orders: list, website_orders: list = None, enquiries: list = None) -> bool:
    """Send the daily summary email. Returns True on success."""
    if website_orders is None:
        website_orders = fetch_wc_orders_today()
    if enquiries is None:
        enquiries = st.session_state.get("enquiries", [])

    try:
        gmail        = st.secrets.get("gmail", {})
        sender       = gmail.get("user", "")
        app_password = gmail.get("password", "")
        recipient    = gmail.get("owner", "")

        if not all([sender, app_password, recipient]):
            print("[Email] Missing Gmail credentials in secrets.")
            return False

        msg            = MIMEMultipart("alternative")
        msg["Subject"] = f"🌸 Indian Flowers — Daily Summary {get_today_str()}"
        msg["From"]    = f"Priya AI Agent <{sender}>"
        msg["To"]      = recipient

        html_body = build_email_html(chat_orders, website_orders, enquiries)
        msg.attach(MIMEText(html_body, "html"))

        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(sender, app_password)
            server.sendmail(sender, recipient, msg.as_string())

        print(f"[Email] Daily summary sent to {recipient}")
        return True

    except Exception as e:
        print(f"[Email error]: {e}")
        return False


def maybe_auto_send_email():
    """Auto-send daily summary at 9 PM ET if not already sent today."""
    today        = get_today_str()
    already_sent = st.session_state.get("email_sent_date") == today
    if already_sent:
        return
    if get_current_hour_et() >= 21:
        chat_orders    = st.session_state.get("all_confirmed_orders", [])
        website_orders = fetch_wc_orders_today()
        enquiries      = st.session_state.get("enquiries", [])
        success        = send_daily_email(chat_orders, website_orders, enquiries)
        if success:
            st.session_state["email_sent_date"] = today


# ─────────────────────────────────────────────
# 17. SESSION STATE INIT
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
    if "enquiries" not in st.session_state:
        st.session_state.enquiries = []
    if "email_sent_date" not in st.session_state:
        st.session_state.email_sent_date = None

init_session()
maybe_auto_send_email()

# ── Sidebar ──
wc_status = "✅ WooCommerce API Connected" if WC_KEY else "⚠️ Using fallback prices"
st.sidebar.success(wc_status)

st.sidebar.divider()
st.sidebar.subheader("📧 Daily Order Summary")
todays_chat_orders = st.session_state.get("all_confirmed_orders", [])
todays_enquiries   = st.session_state.get("enquiries", [])
st.sidebar.metric("Chat Orders Today",  len(todays_chat_orders))
st.sidebar.metric("Enquiries Today",    len(todays_enquiries))
total_today = sum(o.get("grand_total", 0) for o in todays_chat_orders)
if todays_chat_orders:
    st.sidebar.metric("Chat Revenue Today", f"${total_today:.2f}")

if st.session_state.get("email_sent_date") == get_today_str():
    st.sidebar.success("✅ Summary email sent today")

# ─────────────────────────────────────────────
# 18. DISPLAY CHAT HISTORY
# ─────────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"], avatar="🌸" if msg["role"] == "assistant" else "👤"):
        st.markdown(msg["content"])

# ─────────────────────────────────────────────
# 19. HANDLE USER INPUT
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

    # ── Track every message as an enquiry ──
    track_enquiry(user_input, memory)

    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user", avatar="👤"):
        st.markdown(user_input)

    reply = None

    # ── STAGE 1: LEAD CAPTURE ──────────────────────────────────────────
    if stage == "lead_capture":
        memory  = smart_extract_lead(user_input, memory)
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
                        "temple flowers, jasmine strings, or something else? 😊"
                    )
        else:
            memory["stage"] = "need_discovery"
            name = memory.get("name", "there")
            reply = (
                f"Wonderful, {name}! 🌺 Thank you for your details.\n\n"
                "What can I help you with today? Are you looking for wedding garlands, "
                "temple flowers, jasmine strings, or something else? 😊"
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
            summary = memory.get("confirmed_summary", {})

            # ── Append to confirmed orders list ──
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

            # ── Trigger email immediately on order confirmation ──
            try:
                website_orders = fetch_wc_orders_today()
                send_daily_email(
                    st.session_state.all_confirmed_orders,
                    website_orders,
                    st.session_state.get("enquiries", [])
                )
                st.session_state["email_sent_date"] = get_today_str()
                print(f"[Auto Email] Sent on order confirmation for {memory.get('name')}")
            except Exception as e:
                print(f"[Auto email error]: {e}")

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
                "\nINSTRUCTION: Ask warmly what flowers/garlands they need. "
                "Get specifics: product type, quantity, size. "
                "Do NOT ask about occasion or purpose. "
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
