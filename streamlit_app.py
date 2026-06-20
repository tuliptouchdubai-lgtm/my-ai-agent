import re
import streamlit as st

st.set_page_config(
    page_title="The Indian Flowers USA Tester",
    layout="wide",
)

PRODUCTS = {
    "jasmine string":            {"price": 13.00,  "unit": "box (5 ft)"},
    "jathimalli string":         {"price": 15.00,  "unit": "box (5 ft)"},
    "mullai string":             {"price": 15.00,  "unit": "box (5 ft)"},
    "kathambam string":          {"price": 15.00,  "unit": "box (5 ft)"},
    "kanakambaram string":       {"price": 15.00,  "unit": "box (5 ft)"},
    "neem flowers":              {"price": 10.00,  "unit": "100g"},
    "jasmine flowers":           {"price": 10.00,  "unit": "100g"},
    "mullai loose":              {"price": 10.00,  "unit": "100g"},
    "kanakambaram":              {"price": 10.00,  "unit": "100g"},
    "lilly loose":               {"price": 8.00,   "unit": "100g"},
    "marigold loose":            {"price": 8.00,   "unit": "100g"},
    "yellow rose":               {"price": 6.00,   "unit": "100g"},
    "red rose":                  {"price": 6.00,   "unit": "100g"},
    "arali pink":                {"price": 6.00,   "unit": "100g"},
    "arali red":                 {"price": 6.00,   "unit": "100g"},
    "lotus flowers":             {"price": 2.00,   "unit": "piece"},
    "turmeric":                  {"price": 8.00,   "unit": "each"},
    "avarampoo":                 {"price": 10.00,  "unit": "100g"},
    "tulasi leaves":             {"price": 10.00,  "unit": "100g"},
    "neem leaf":                 {"price": 10.00,  "unit": "100g"},
    "vilvam leaves":             {"price": 10.00,  "unit": "100g"},
    "erukkam leaves":            {"price": 10.00,  "unit": "100g"},
    "marikolunthu":              {"price": 10.00,  "unit": "100g"},
    "maruvam":                   {"price": 10.00,  "unit": "100g"},
    "betel leaf":                {"price": 6.00,   "unit": "pack of 25"},
    "mango leaf":                {"price": 2.00,   "unit": "pack of 10"},
    "mango leaf thoranam":       {"price": 5.00,   "unit": "4 ft"},
    "banana leaf":               {"price": 10.00,  "unit": "pack"},
    "coconut thoranam":          {"price": 5.00,   "unit": "5 pieces"},
    "ala mokku":                 {"price": 10.00,  "unit": "pack"},
    "marigold garland":          {"price": 20.00,  "unit": "per foot"},
    "rose petal garland":        {"price": 100.00, "unit": "pair (4 ft total)"},
    "carnation garland":         {"price": 100.00, "unit": "pair (4 ft total)"},
    "lilly garland":             {"price": 100.00, "unit": "pair (4 ft total)"},
    "lily garland":              {"price": 100.00, "unit": "pair (4 ft total)"},
    "exchange garland":          {"price": 80.00,  "unit": "pair"},
    "button rose garland":       {"price": 26.00,  "unit": "per foot (each side)"},
    "wedding garland":           {"price": 100.00, "unit": "pair (4 ft total)"},
    "north indian garland":      {"price": 100.00, "unit": "pair"},
    "door garland":              {"price": 130.00, "unit": "10 ft set"},
    "house warming garland":     {"price": 130.00, "unit": "10 ft set"},
    "veni":                      {"price": 15.00,  "unit": "piece"},
    "jadai":                     {"price": 15.00,  "unit": "piece"},
    "veni jadai":                {"price": 80.00,  "unit": "set"},
    "gajra":                     {"price": 80.00,  "unit": "set"},
    "temple garland":            {"price": 26.00,  "unit": "per foot"},
    "pooja garland":             {"price": 26.00,  "unit": "per foot"},
    "vadamalli garland":         {"price": 26.00,  "unit": "per foot"},
    "bouquet":                   {"price": 8.00,   "unit": "piece"},
}

LOCAL_ZIPS = {
    "92335","92336","92337","92316","92324","92376","92377",
    "91710","91761","91762","91763","91764","91766","91767","91768",
}

CA_PREFIXES = {
    "900","901","902","903","904","905","906","907","908",
    "910","911","912","913","914","915","916","917","918","919",
    "920","921","922","923","924","925","926","927","928",
    "930","931","932","933","934","935","936","937","938","939",
    "940","941","942","943","944","945","946","947","948","949",
    "950","951","952","953","954","955","956","957","958","959","960","961",
}

ORDER_KEYWORDS = list(PRODUCTS.keys())


def calculate_shipping(zip_code: str, weight_lbs: float = 2.0) -> dict:
    z = zip_code.strip()
    if z in LOCAL_ZIPS:
        return {
            "fee": 30.00,
            "method": "Local Delivery (within 50 miles)",
            "note": "Delivered fresh same/next day.",
        }
    if len(z) >= 3 and z[:3] in CA_PREFIXES:
        return {
            "fee": 55.00,
            "method": "California State Shipping",
            "note": "Delivered in 1-2 business days.",
        }
    if weight_lbs > 15:
        return {
            "fee": 0.00,
            "method": "Southwest Cargo (Bulk Order)",
            "note": "Southwest Cargo preferred for freshness. Rate quoted separately.",
        }
    return {
        "fee": 70.00,
        "method": "Nationwide USA Shipping",
        "note": "Delivered in 2-3 business days via overnight cold-pack.",
    }


def parse_order_items(text: str) -> list:
    found = []
    seen = set()
    low = text.lower()
    for name, info in PRODUCTS.items():
        if name in low and name not in seen:
            match = re.search(
                r"(\d+(?:\.\d+)?)\s*(?:feet|ft|foot|box(?:es)?|piece[s]?|pair[s]?|bunch(?:es)?|set[s]?|pack(?:s)?|each)?\s*(?:of\s*)?"
                + re.escape(name),
                low,
            )
            if not match:
                match = re.search(r"(\d+(?:\.\d+)?)\s*" + re.escape(name), low)
            qty = float(match.group(1)) if match else 1.0
            found.append({
                "name": name,
                "qty": qty,
                "unit_price": info["price"],
                "unit": info["unit"],
            })
            seen.add(name)
    return found


def build_order_summary(items: list, zip_code: str) -> dict:
    subtotal = sum(i["qty"] * i["unit_price"] for i in items)
    weight = len(items) * 1.5
    shipping = calculate_shipping(zip_code, weight)
    grand_total = subtotal + shipping["fee"]
    return {
        "items": items,
        "subtotal": subtotal,
        "shipping": shipping,
        "grand_total": grand_total,
    }


def format_order_summary(summary: dict) -> str:
    lines = ["ORDER SUMMARY"]
    for item in summary["items"]:
        lines.append(
            f"• {item['qty']} x {item['name'].title()} @ ${item['unit_price']:.2f}/{item['unit']} = ${item['qty'] * item['unit_price']:.2f}"
        )
    lines.append(f"Subtotal: ${summary['subtotal']:.2f}")
    lines.append(
        f"Shipping: ${summary['shipping']['fee']:.2f} ({summary['shipping']['method']})"
    )
    lines.append(f"Grand Total: ${summary['grand_total']:.2f}")
    lines.append(f"Note: {summary['shipping']['note']}")
    return "\n".join(lines)


def validate_zip(zip_code: str) -> bool:
    return bool(re.fullmatch(r"\d{5}", zip_code.strip()))


def render_product_catalog():
    lines = [f"{name.title()} — ${info['price']:.2f} / {info['unit']}" for name, info in PRODUCTS.items()]
    return "\n".join(lines)


st.title("The Indian Flowers USA — Streamlit Test App")
st.markdown(
    "Use this tool to validate order parsing, shipping logic, and pricing calculations for the customer flow."
)

with st.sidebar:
    st.header("Customer Details")
    customer_name = st.text_input("Name", "Ananya")
    phone = st.text_input("Phone", "9091234567")
    email = st.text_input("Email", "ananya@example.com")
    zip_code = st.text_input("ZIP Code", "92336")
    st.markdown("---")
    st.markdown("**Shipping rules**")
    st.write("Local delivery: $30 (selected local zips)")
    st.write("California state: $55")
    st.write("Nationwide USA: $70")
    st.write("Bulk over 15 lbs: quoted separately")

st.subheader("Order Input")
order_text = st.text_area(
    "Customer message / order details",
    "I need 2 jasmine strings and 1 rose petal garland for a wedding",
    height=180,
)

col1, col2 = st.columns([2, 1])
with col1:
    st.subheader("Parsed Order")
    items = parse_order_items(order_text)
    if items:
        for item in items:
            st.write(
                f"- {item['qty']} x {item['name'].title()} ({item['unit']}) — ${item['qty'] * item['unit_price']:.2f}"
            )
    else:
        st.warning("No recognized product items were parsed from the order text.")

with col2:
    st.subheader("Debug Info")
    st.write(f"Name: {customer_name}")
    st.write(f"Phone: {phone}")
    st.write(f"Email: {email}")
    st.write(f"ZIP: {zip_code}")
    if not validate_zip(zip_code):
        st.error("ZIP code must be exactly 5 digits.")

if st.button("Calculate Summary"):
    if not customer_name.strip():
        st.error("Customer name is required.")
    elif not phone.strip():
        st.error("Phone number is required.")
    elif not email.strip():
        st.error("Email is required.")
    elif not validate_zip(zip_code):
        st.error("ZIP code must be exactly 5 digits.")
    elif not items:
        st.error("No valid order items parsed. Please update the order message.")
    else:
        summary = build_order_summary(items, zip_code)
        st.success("Order summary generated successfully.")
        st.code(format_order_summary(summary))
        st.markdown(
            "#### Payment Instructions\n"
            f"Please collect **${summary['grand_total']:.2f}** via Zelle to **Malar Traders**."
        )

st.markdown("---")
st.subheader("Product Catalog")
st.code(render_product_catalog())

st.markdown("---")
st.subheader("Notes for testers")
st.write(
    "- Paste a customer message and verify that the correct products and quantities are parsed.\n"
    "- Check the shipping fee for local CA zips, CA state zips, and other USA zips.\n"
    "- This app is intended for testing the order capture and pricing logic independent of the LLM chat flow."
)
