import os
import re
import json
import subprocess
import sys
import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
import smtplib
from email.mime.text import MIMEText
from apscheduler.schedulers.background import BackgroundScheduler
import pandas as pd
from datetime import datetime

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

# (… your existing Playwright, LLM setup, scraping, product catalog, shipping calculator, lead extractors, system prompt, etc. remain unchanged …)

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
    if "orders" not in st.session_state:
        st.session_state.orders = []

init_session()

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

    # Example: when user confirms order
    if lowered in CONFIRM_TRIGGERS and memory.get("order_items"):
        memory["order_confirmed"] = True
        # Build final summary
        summary_text = final_confirmation_text(memory)
        st.session_state.messages.append({"role": "assistant", "content": summary_text})
        with st.chat_message("assistant", avatar="🌸"):
            st.markdown(summary_text)
        # Add to daily summary list
        st.session_state.orders.append({
            "Name": memory.get("name"),
            "Phone": memory.get("phone"),
            "Email": memory.get("email"),
            "Zip": memory.get("zip_code"),
            "Items": memory["confirmed_summary"]["items"],
            "Subtotal": memory["confirmed_summary"]["subtotal"],
            "Shipping": memory["confirmed_summary"]["shipping"]["fee"],
            "GrandTotal": memory["confirmed_summary"]["grand_total"],
        })

    # (… your other chat handling stages remain unchanged …)

# ─────────────────────────────────────────────
# 13. DAILY SUMMARY EMAIL (Gmail SMTP)
# ─────────────────────────────────────────────
def send_daily_summary():
    if not st.session_state.orders:
        st.warning("No confirmed orders yet.")
        return

    lines = []
    for o in st.session_state.orders:
        lines.append(f"Customer: {o['Name']} ({o['Phone']}, {o['Email']}, ZIP {o['Zip']})")
        for item in o["Items"]:
            lines.append(
                f"  • {item['qty']} x {item['name'].title()} @ ${item['unit_price']:.2f}/{item['unit']} = ${item['qty'] * item['unit_price']:.2f}"
            )
        lines.append(f"  Subtotal: ${o['Subtotal']:.2f}")
        lines.append(f"  Shipping: ${o['Shipping']:.2f}")
        lines.append(f"  Grand Total: ${o['GrandTotal']:.2f}")
        lines.append("────────────────────────────")

    summary_text = "\n".join(lines)

    msg = MIMEText(summary_text)
    msg['Subject'] = f"Daily Order Summary - {datetime.now().strftime('%Y-%m-%d')}"
    msg['From'] = st.secrets["gmail"]["user"]
    msg['To'] = st.secrets["gmail"]["owner"]

    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
        server.login(st.secrets["gmail"]["user"], st.secrets["gmail"]["password"])
        server.send_message(msg)

    st.success("✅ Daily summary email sent!")

# Manual button trigger
if st.button("Send Daily Summary Now"):
    send_daily_summary()

# Scheduler setup (auto daily send at 23:59)
def schedule_daily_email():
    send_daily_summary()

if "scheduler" not in st.session_state:
    scheduler = BackgroundScheduler()
    scheduler.add_job(schedule_daily_email, 'cron', hour=23, minute=59)
    scheduler.start()
    st.session_state.scheduler = scheduler
