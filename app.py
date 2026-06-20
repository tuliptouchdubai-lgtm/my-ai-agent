import os
import re
import json
import chainlit as cl

from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser

# ─────────────────────────────────────────────
# 1. OLLAMA LLM SETUP
# ─────────────────────────────────────────────
OLLAMA_MODEL = "llama3"

chat_llm = ChatOllama(
    model=OLLAMA_MODEL,
    temperature=0.2,
    num_predict=400,
    base_url="http://localhost:11434",
)

extract_llm = ChatOllama(
    model=OLLAMA_MODEL,
    temperature=0.0,
    num_predict=200,
    base_url="http://localhost:11434",
)

# ─────────────────────────────────────────────
# 2. KNOWLEDGE BASE
# ─────────────────────────────────────────────
BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
WEBSITE_PATH = os.path.join(BASE_DIR, "website_info.txt")

def load_knowledge() -> str:
    if os.path.exists(WEBSITE_PATH):
        with open(WEBSITE_PATH, "r", encoding="utf-8") as f:
            content = f.read().strip()
        print(f"[INFO] Loaded website_info.txt ({len(content)} chars)")
        return content
    print("[WARN] website_info.txt not found — using default description")
    return (
        "The Indian Flowers USA (Malar Traders) delivers fresh Indian flowers "
        "and garlands nationwide across the USA."
    )

WEBSITE_INFO = load_knowledge()

# ─────────────────────────────────────────────
# 3. PRODUCT CATALOG — built from website_info
#    Covers all real products with real prices
# ─────────────────────────────────────────────
PRODUCTS = {
    # Strings — website shows $15 per box (5 ft) for all strings
    "jasmine string":            {"price": 13.00,  "unit": "box (5 ft)"},   # website: $13
    "jathimalli string":         {"price": 15.00,  "unit": "box (5 ft)"},
    "mullai string":             {"price": 15.00,  "unit": "box (5 ft)"},
    "kathambam string":          {"price": 15.00,  "unit": "box (5 ft)"},
    "kanakambaram string":       {"price": 15.00,  "unit": "box (5 ft)"},

    # Loose flowers / leaves (100g or each) — verified from website_info.txt
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

    # Leaves
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

    # Marigold garlands (per foot)
    "marigold garland":          {"price": 20.00,  "unit": "per foot"},

    # Wedding / Exchange garlands — standard price per pair (2 ft each side = 4 ft total)
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

    # Veni / Jadai
    "veni":                      {"price": 15.00,  "unit": "piece"},
    "jadai":                     {"price": 15.00,  "unit": "piece"},
    "veni jadai":                {"price": 80.00,  "unit": "set"},
    "gajra":                     {"price": 80.00,  "unit": "set"},

    # Temple / Pooja garlands
    "temple garland":            {"price": 26.00,  "unit": "per foot"},
    "pooja garland":             {"price": 26.00,  "unit": "per foot"},
    "vadamalli garland":         {"price": 26.00,  "unit": "per foot"},

    # Bouquets
    "bouquet":                   {"price": 8.00,   "unit": "piece"},
}

PRODUCT_CATALOG_TEXT = "\n".join(
    f"  - {name.title()}: ${v['price']:.2f} per {v['unit']}"
    for name, v in PRODUCTS.items()
)

# ─────────────────────────────────────────────
# 4. SHIPPING CALCULATOR
# ─────────────────────────────────────────────
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
                "note": "Southwest Cargo preferred for freshness. Rate quoted separately."}
    return {"fee": 70.00, "method": "Nationwide USA Shipping",
            "note": "Delivered in 2-3 business days via overnight cold-pack."}

def parse_order_items(text: str) -> list:
    """Extract product + quantity from confirmed order text."""
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
    lines = ["📦 ORDER BREAKDOWN (calculated by system — present EXACTLY these numbers):"]
    for item in summary["items"]:
        lines.append(
            f"  • {item['qty']} x {item['name'].title()} "
            f"@ ${item['unit_price']:.2f}/{item['unit']} = ${item['qty'] * item['unit_price']:.2f}"
        )
    lines.append(f"  Subtotal  : ${summary['subtotal']:.2f}")
    lines.append(
        f"  Shipping  : ${summary['shipping']['fee']:.2f} "
        f"({summary['shipping']['method']})"
    )
    lines.append(f"  ─────────────────────────────────")
    lines.append(f"  GRAND TOTAL: ${summary['grand_total']:.2f}")
    lines.append(f"  Delivery note: {summary['shipping']['note']}")
    lines.append("INSTRUCTION: Present these EXACT figures. Do NOT recalculate or guess prices.")
    return "\n".join(lines)

# ─────────────────────────────────────────────
# 5. CONVERSATION STAGES
#    Stage controls what Mithu is doing
# ─────────────────────────────────────────────
# Stages:
#   "lead_capture"    → collecting name/phone/email/zip (in order)
#   "need_discovery"  → all lead fields collected, now ask WHAT they need
#   "order_building"  → customer told us what they want, clarifying details
#   "order_confirm"   → show price + shipping, ask for confirmation
#   "order_done"      → confirmed, show payment instructions

LEAD_FIELDS = ["name", "phone", "email", "zip_code"]
LEAD_QUESTIONS = {
    "name":     "May I have your full name please? 😊",
    "phone":    "Thank you! Could you share your phone number?",
    "email":    "Got it! What's your email address for order updates?",
    "zip_code": "Almost there! What's your delivery ZIP code so I can calculate shipping?",
}

def next_missing_field(memory: dict) -> str | None:
    for f in LEAD_FIELDS:
        if not memory.get(f):
            return f
    return None

# ─────────────────────────────────────────────
# 6. REGEX ENTITY EXTRACTOR
#    IMPORTANT: Only extracts order info AFTER lead_capture stage
# ─────────────────────────────────────────────
SKIP_WORDS = {
    "ok","yes","no","confirm","proceed","hello","hi","hey","thanks","thank",
    "please","sure","great","good","need","want","order","get","i","me","my",
    "the","and","for","a","an","is","it","in","to","of","can","do","what",
    "box","boxes","piece","pieces","pair","pairs","bunch","bunches","set","sets",
}

ORDER_KEYWORDS = list(PRODUCTS.keys()) + [
    "garland","string","flower","veni","jasmine","rose","carnation",
    "lily","lilly","mullai","tulasi","pooja","temple","wedding","exchange",
    "marigold","lotus","gajra","jadai","bouquet","door","housewarming",
]

def regex_extract_lead(text: str, memory: dict) -> dict:
    """
    Only extracts lead fields (name/phone/email/zip).
    Does NOT capture order intent — that happens after lead_capture stage.
    """
    updated = dict(memory)
    low     = text.lower().strip()

    # Email
    if not updated.get("email"):
        m = re.search(r'[\w.\-+]+@[\w\-]+\.[a-z]{2,}', text, re.IGNORECASE)
        if m:
            updated["email"] = m.group(0).lower()

    # Phone — 10-digit US
    if not updated.get("phone"):
        digits = re.sub(r'\D', '', text)
        if len(digits) == 10:
            updated["phone"] = digits
        elif len(digits) == 11 and digits[0] == '1':
            updated["phone"] = digits[1:]

    # ZIP — exactly 5 digits
    if not updated.get("zip_code"):
        m = re.search(r'\b(\d{5})\b', text)
        if m:
            updated["zip_code"] = m.group(1)

    # Name (only if not yet captured)
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
            is_short      = len(words) <= 3
            no_digits     = not re.search(r'\d', text)
            no_skip       = not any(w in SKIP_WORDS for w in words)
            no_order_word = not any(kw in low for kw in ORDER_KEYWORDS)
            if is_short and no_digits and no_skip and no_order_word:
                updated["name"] = text.strip().title()

    return updated

def extract_order_intent(text: str, memory: dict) -> dict:
    """
    Only called during need_discovery / order_building stages.
    Captures what the customer wants to order.
    """
    updated = dict(memory)
    low     = text.lower()

    if any(kw in low for kw in ORDER_KEYWORDS):
        existing = updated.get("raw_order_text") or ""
        updated["raw_order_text"] = (existing + " " + text).strip()

    # Occasion
    if not updated.get("occasion"):
        for occ in ["wedding","pooja","temple","birthday","engagement","festival","puja","housewarming"]:
            if occ in low:
                updated["occasion"] = occ
                break

    return updated

# ─────────────────────────────────────────────
# 7. LLM EXTRACTION CHAIN (fallback)
# ─────────────────────────────────────────────
EXTRACT_PROMPT = ChatPromptTemplate.from_template(
    """Extract customer info from this single message. Reply ONLY with valid JSON — no explanation, no markdown fences.

Message: "{message}"

Return exactly this JSON (null for anything not found):
{{
  "name": null,
  "phone": null,
  "email": null,
  "zip_code": null,
  "occasion": null
}}

Strict rules:
- name: real person name only. NEVER capture: ok/yes/no/confirm/hello/hi/thanks/proceed/sure/great/please/order/want/need
- phone: 10-digit US number as plain digit string only
- email: valid email format only
- zip_code: 5-digit US zip only
- occasion: wedding/pooja/temple/birthday/housewarming/other — or null"""
)

extract_chain = EXTRACT_PROMPT | extract_llm | StrOutputParser()

async def llm_extract_lead(text: str, memory: dict) -> dict:
    updated = dict(memory)
    try:
        raw = await extract_chain.ainvoke({"message": text})
        raw = re.sub(r"^```(?:json)?", "", raw.strip())
        raw = re.sub(r"```$", "", raw).strip()
        m   = re.search(r'\{.*?\}', raw, re.DOTALL)
        if m:
            extracted = json.loads(m.group(0))
            for field in ["name","phone","email","zip_code","occasion"]:
                val = extracted.get(field)
                if val and str(val).strip().lower() not in ("null","none",""):
                    if not updated.get(field):
                        updated[field] = str(val).strip()
    except Exception as e:
        print(f"[LLM extract error]: {e}")
    return updated

async def smart_extract_lead(text: str, memory: dict) -> dict:
    """Regex first, LLM fallback only if fields still missing."""
    updated = regex_extract_lead(text, memory)
    missing = [f for f in LEAD_FIELDS if not updated.get(f)]
    if missing:
        updated = await llm_extract_lead(text, updated)
    return updated

# ─────────────────────────────────────────────
# 8. SYSTEM PROMPT
# ─────────────────────────────────────────────
SYSTEM_PROMPT = f"""You are Mithu, a warm and helpful sales assistant for The Indian Flowers USA (Malar Traders).

YOUR ROLE:
- Help customers find the right flowers/garlands for their occasion
- Ask about their needs, occasion, and preferences warmly
- Present prices EXACTLY as given in the ORDER BREAKDOWN block — NEVER recalculate or guess
- Keep replies SHORT — 2 to 4 sentences max
- Use the customer's name warmly
- Ask ONE question at a time only
- NEVER mention product codes (like RP83, LG01, etc.)

ABOUT THE BUSINESS:
{WEBSITE_INFO[:3000]}

PRODUCT CATALOG (for reference only — use ORDER BREAKDOWN for final pricing):
{PRODUCT_CATALOG_TEXT}

SHIPPING RATES:
  - Local delivery (Fontana/Ontario CA area, within 50 miles): $30.00
  - California state: $55.00
  - Nationwide USA: $70.00
  - Bulk over 15 lbs: Southwest Cargo (quoted separately)
  - NO local pickup — all orders shipped

PAYMENT: Zelle to "Malar Traders" only. Never ask for card details.

CONVERSATION FLOW (follow this strictly):
1. Lead fields already collected by system — do NOT re-ask them
2. When stage is "need_discovery": warmly greet customer by name, ask what they're looking for and what occasion it's for
3. When stage is "order_building": help clarify their order (style, quantity, size preferences). Suggest popular options based on occasion.
4. When stage is "order_confirm": show the ORDER BREAKDOWN exactly, ask customer to type "confirm" to proceed
5. When stage is "order_done": thank customer, remind about Zelle payment to Malar Traders

CRITICAL RULES:
- NEVER assume or make up an order — always ask the customer what they want
- NEVER show a price summary unless ORDER BREAKDOWN is provided in the context
- NEVER recalculate prices — only use ORDER BREAKDOWN figures
- NEVER suggest pickup
- Keep replies SHORT and warm
"""

# ─────────────────────────────────────────────
# 9. LANGCHAIN CONVERSATION CHAIN
# ─────────────────────────────────────────────
chat_prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}"),
])

conversation_chain = chat_prompt | chat_llm | StrOutputParser()

# ─────────────────────────────────────────────
# 10. CHAINLIT HANDLERS
# ─────────────────────────────────────────────
@cl.on_chat_start
async def start():
    cl.user_session.set("history", [])
    cl.user_session.set("memory", {
        "name":            None,
        "phone":           None,
        "email":           None,
        "zip_code":        None,
        "occasion":        None,
        "raw_order_text":  None,
        "order_items":     [],
        "stage":           "lead_capture",   # track conversation stage
        "order_confirmed": False,
    })
    await cl.Message(
        content=(
            "🌸 Welcome to **The Indian Flowers USA**!\n\n"
            "I'm Mithu, your personal flower assistant. "
            "We deliver fresh jasmine, rose, carnation garlands and more "
            "to your doorstep anywhere in the USA.\n\n"
            "May I have your **full name** to get started? 😊"
        )
    ).send()


@cl.on_message
async def main(message: cl.Message):
    history   = cl.user_session.get("history")
    memory    = cl.user_session.get("memory")
    user_text = message.content.strip()
    lowered   = user_text.lower()
    stage     = memory.get("stage", "lead_capture")

    # ── STAGE 1: LEAD CAPTURE ─────────────────────────────────────────────
    if stage == "lead_capture":
        # Extract lead fields ONLY (no order capture here)
        memory = await smart_extract_lead(user_text, memory)
        cl.user_session.set("memory", memory)

        missing_field = next_missing_field(memory)

        if missing_field:
            # Check if this message provided the field we just asked for
            just_provided = memory.get(missing_field)
            if not just_provided:
                # Still missing — ask again
                # Special case: if they mentioned order intent before giving name
                if missing_field == "name" and any(kw in lowered for kw in ORDER_KEYWORDS):
                    await cl.Message(
                        content=(
                            "🌸 We'd love to help with your order! "
                            f"First, {LEAD_QUESTIONS['name']}"
                        )
                    ).send()
                else:
                    await cl.Message(content=LEAD_QUESTIONS[missing_field]).send()
            else:
                # Just provided this field — ask for the next one
                next_field = next_missing_field(memory)
                if next_field:
                    await cl.Message(
                        content=f"Thank you! {LEAD_QUESTIONS[next_field]}"
                    ).send()
                else:
                    # All lead fields now complete → move to need discovery
                    memory["stage"] = "need_discovery"
                    cl.user_session.set("memory", memory)
                    name = memory.get("name", "there")
                    await cl.Message(
                        content=(
                            f"Wonderful, {name}! 🌺 Thank you for your details.\n\n"
                            "What can I help you with today? "
                            "Are you looking for wedding garlands, temple flowers, jasmine strings, or something else? "
                            "And what's the special occasion? 😊"
                        )
                    ).send()
            return
        else:
            # All fields were already complete — transition
            memory["stage"] = "need_discovery"
            cl.user_session.set("memory", memory)
            name = memory.get("name", "there")
            await cl.Message(
                content=(
                    f"Wonderful, {name}! 🌺 Thank you for your details.\n\n"
                    "What can I help you with today? "
                    "Are you looking for wedding garlands, temple flowers, jasmine strings, or something else? "
                    "And what's the special occasion? 😊"
                )
            ).send()
            return

    # ── STAGE 2: NEED DISCOVERY ───────────────────────────────────────────
    if stage == "need_discovery":
        # Now we capture order intent
        memory = extract_order_intent(user_text, memory)

        # Also check if zip was missing (edge case)
        if not memory.get("zip_code"):
            m = re.search(r'\b(\d{5})\b', user_text)
            if m:
                memory["zip_code"] = m.group(1)

        cl.user_session.set("memory", memory)

        # If they mentioned what they want, move to order_building
        if memory.get("raw_order_text") or memory.get("occasion"):
            memory["stage"] = "order_building"
            cl.user_session.set("memory", memory)

        # Fall through to LLM to respond naturally and ask clarifying questions
        # about what they need, style preferences, quantities, etc.

    # ── STAGE 3: ORDER BUILDING ───────────────────────────────────────────
    if stage == "order_building":
        # Keep accumulating order text
        memory = extract_order_intent(user_text, memory)

        # Parse items from accumulated order text
        if memory.get("raw_order_text"):
            items = parse_order_items(memory["raw_order_text"])
            if items:
                memory["order_items"] = items

        cl.user_session.set("memory", memory)

        # Check for confirm triggers
        CONFIRM_TRIGGERS = {
            "confirm","yes confirm","proceed","confirm order","place order",
            "proceed with payment","i confirm","yes proceed","confirm to proceed",
            "ok confirm","yes i confirm","let's proceed","lets proceed","yes place",
            "go ahead","finalize","that's correct","looks good","looks correct",
        }
        if any(t in lowered for t in CONFIRM_TRIGGERS):
            # Move to confirm stage
            if memory.get("order_items") and memory.get("zip_code"):
                memory["stage"] = "order_confirm"
                await _send_order_confirmation(memory)   # stores confirmed_summary in memory
                cl.user_session.set("memory", memory)   # save AFTER so confirmed_summary is persisted
                return
            elif not memory.get("zip_code"):
                await cl.Message(
                    content="I need your delivery ZIP code to calculate shipping. What's your ZIP? 📦"
                ).send()
                return
            elif not memory.get("order_items"):
                await cl.Message(
                    content="I don't have your order details yet. What would you like to order? 🌸"
                ).send()
                return

    # ── STAGE 4: ORDER CONFIRM ────────────────────────────────────────────
    if stage == "order_confirm":
        CONFIRM_TRIGGERS = {
            "confirm","yes","yes confirm","proceed","ok","sure","go ahead",
            "place order","that's correct","looks good","i confirm",
        }
        if any(t in lowered for t in CONFIRM_TRIGGERS):
            memory["stage"] = "order_done"
            memory["order_confirmed"] = True
            await _send_final_confirmation(memory)
            cl.user_session.set("memory", memory)
            return
        elif any(t in lowered for t in ["no","change","wrong","different","edit","modify"]):
            memory["stage"] = "order_building"
            memory["order_items"] = []
            memory["raw_order_text"] = None
            cl.user_session.set("memory", memory)
            await cl.Message(
                content="No problem! Let's update your order. What would you like to change? 😊"
            ).send()
            return

    # ── STAGE 5: ORDER DONE ───────────────────────────────────────────────
    if stage == "order_done":
        # If customer asks about price/summary again, re-show the stored summary
        price_query_words = {"price","pricing","total","cost","how much","amount","summary","breakdown","order"}
        if any(w in lowered for w in price_query_words):
            summary = memory.get("confirmed_summary")
            if summary:
                name = memory.get("name", "there")
                lines = [f"Here's your confirmed order recap, **{name}**! 🌺\n"]
                for item in summary["items"]:
                    lines.append(
                        f"  • {item['qty']} x {item['name'].title()} "
                        f"({item['unit']}) — **${item['qty'] * item['unit_price']:.2f}**"
                    )
                lines.append(f"\n  **Subtotal:** ${summary['subtotal']:.2f}")
                lines.append(
                    f"  **Shipping ({summary['shipping']['method']}):** "
                    f"${summary['shipping']['fee']:.2f}"
                )
                lines.append(f"  **────────────────────────────**")
                lines.append(f"  🧾 **Grand Total: ${summary['grand_total']:.2f}**")
                lines.append(
                    f"\n💳 Please send **${summary['grand_total']:.2f}** via **Zelle to Malar Traders**."
                )
                await cl.Message(content="\n".join(lines)).send()
                return
        await cl.Message(
            content=(
                f"Your order is confirmed, {memory.get('name', 'there')}! 🌺 "
                "Please send payment via Zelle to **Malar Traders** and our team will be in touch shortly. "
                "Is there anything else I can help you with?"
            )
        ).send()
        return

    # ── LLM RESPONSE — for need_discovery and order_building stages ────────
    context_parts = ["=== CUSTOMER INFO ==="]
    context_parts.append(f"Name    : {memory.get('name')}")
    context_parts.append(f"Phone   : {memory.get('phone')}")
    context_parts.append(f"Email   : {memory.get('email')}")
    context_parts.append(f"ZIP     : {memory.get('zip_code')}")
    context_parts.append(f"Stage   : {memory.get('stage')}")
    if memory.get("occasion"):
        context_parts.append(f"Occasion: {memory['occasion']}")
    if memory.get("raw_order_text"):
        context_parts.append(f"Order so far: \"{memory['raw_order_text']}\"")

    # If we have items AND zip, pre-calculate and inject
    if memory.get("order_items") and memory.get("zip_code"):
        summary = build_order_summary(memory["order_items"], memory["zip_code"])
        context_parts.append("\n" + format_order_summary(summary))
        context_parts.append(
            "\nINSTRUCTION: If the customer seems ready to order, present the ORDER BREAKDOWN "
            "above and ask them to type 'confirm' to proceed."
        )
    elif memory.get("stage") in ("need_discovery", "order_building"):
        context_parts.append(
            "\nINSTRUCTION: Ask the customer warmly about what flowers/garlands they need "
            "and for what occasion. Do NOT assume any order. Get specifics: product type, "
            "quantity, size if applicable. Once you understand their order, summarize it "
            "and ask them to say 'confirm' so you can show the final price."
        )

    context_block   = "\n".join(context_parts)
    augmented_input = f"{context_block}\n\n---\nCustomer: {user_text}"

    msg           = cl.Message(content="")
    await msg.send()
    full_response = ""

    try:
        lc_history = []
        for turn in history:
            if turn["role"] == "user":
                lc_history.append(HumanMessage(content=turn["content"]))
            else:
                lc_history.append(AIMessage(content=turn["content"]))

        async for chunk in conversation_chain.astream({
            "history": lc_history,
            "input":   augmented_input,
        }):
            await msg.stream_token(chunk)
            full_response += chunk

    except Exception as e:
        err = str(e).lower()
        if "connect" in err or "refused" in err:
            full_response = (
                "⚠️ Cannot reach Ollama. "
                "Please run `ollama serve` in a terminal and try again."
            )
        else:
            full_response = "I had a hiccup — please try again! 🙏"
            print(f"[Chain error]: {e}")
        msg.content = full_response
        await msg.update()
        return

    history.append({"role": "user",      "content": user_text})
    history.append({"role": "assistant", "content": full_response})
    if len(history) > 20:
        history = history[-20:]

    cl.user_session.set("history", history)
    cl.user_session.set("memory",  memory)


async def _send_order_confirmation(memory: dict):
    """
    Calculate and show the order summary, then ask for confirmation.
    Stores the calculated summary in memory so it is NEVER recalculated later.
    """
    summary = build_order_summary(memory["order_items"], memory["zip_code"])
    # ★ KEY FIX: store summary in memory so _send_final_confirmation uses
    #   these exact same numbers — no risk of recalculation with wrong prices
    memory["confirmed_summary"] = summary
    name = memory.get("name", "there")

    lines = [f"Here's your order summary, **{name}**! 🌺\n"]
    for item in summary["items"]:
        lines.append(
            f"  • {item['qty']} x {item['name'].title()} "
            f"({item['unit']}) — **${item['qty'] * item['unit_price']:.2f}**"
        )
    lines.append(f"\n  **Subtotal:** ${summary['subtotal']:.2f}")
    lines.append(
        f"  **Shipping ({summary['shipping']['method']}):** "
        f"${summary['shipping']['fee']:.2f}"
    )
    lines.append(f"  **────────────────────────────**")
    lines.append(f"  🧾 **Grand Total: ${summary['grand_total']:.2f}**")
    lines.append(f"\n  _{summary['shipping']['note']}_")
    lines.append(
        f"\nType **confirm** to place your order, or **change** if you'd like to modify anything. 😊"
    )
    await cl.Message(content="\n".join(lines)).send()


async def _send_final_confirmation(memory: dict):
    """
    Send the final confirmed order with payment instructions.
    ALWAYS uses the pre-calculated summary stored in memory — never recalculates.
    """
    # ★ KEY FIX: use stored summary, not a fresh build_order_summary() call
    summary = memory.get("confirmed_summary") or build_order_summary(
        memory["order_items"], memory["zip_code"]
    )
    name = memory.get("name", "there")

    lines = [f"🎉 **Thank you, {name}! Your order is confirmed.**\n"]
    for item in summary["items"]:
        lines.append(
            f"  • {item['qty']} x {item['name'].title()} "
            f"— **${item['qty'] * item['unit_price']:.2f}**"
        )
    lines.append(
        f"\n  **Shipping ({summary['shipping']['method']}):** "
        f"${summary['shipping']['fee']:.2f}"
    )
    lines.append(f"  **Grand Total: ${summary['grand_total']:.2f}**")
    lines.append(
        f"\n💳 Please send **${summary['grand_total']:.2f}** via **Zelle to Malar Traders**."
    )
    lines.append(
        "\n✅ Our team will send your receipt and delivery update shortly. "
        "Wishing you a beautiful celebration! 🌺"
    )
    await cl.Message(content="\n".join(lines)).send()