import chainlit as cl
from langchain_community.llms import Ollama
from langchain.memory import ConversationBufferMemory # pyright: ignore[reportMissingImports]
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate # type: ignore
from langchain.chains import LLMChain # pyright: ignore[reportMissingImports]
from langchain_core.messages import SystemMessage
import json

# =========================================================================
# ⚙️ CONSTANTS & PRODUCT DATA
# =========================================================================
LOGISTICS_RULES = {
    "PICKUP_LOCATIONS": [
        "Houston", "Tampa", "Seattle", "Sacramento", "Pittsburgh", 
        "Reminderville (Ohio)", "Livermore", "Greenwood Village (CO)", 
        "Devon (PA 19333)", "Minnesota (MN 55077)", "Fontana (CA 92336)"
    ]
}

PRODUCT_DATA = """
- Jasmine String (5ft): Freshly strung loose jasmine strands, bulk or single.
- Mullai String: Fragrant premium local traditional loose flowers.
- Rose Petal Garland (RP32): Premium hand-crafted wedding garland.
- Lilly Garland (LG23): Elegant celebratory structural floral garland.
- Other Items: Kakada, Lotus, Banana Leaf, Betel Leaf.
"""

# =========================================================================
# 🌸 SYSTEM INSTRUCTIONS
# =========================================================================
SYSTEM_INSTRUCTION = f"""
Role: Mira, the Lead AI Concierge for "theindianflowers".
Persona: You are a professional, helpful, warm, and approachable concierge. 

Intro Message: You MUST start the conversation with exactly this phrase: "Hi! Welcome to theindianflowers. 🌸 I'm here to help with your festive decor and orders. To get started, may I have your name, phone number, and zip code please?"

Rule: Do NOT provide prices or shipping quotes until you have: Name, Phone, Email, and 5-digit Zip Code.
- If the user asks for a price before providing details, say: "I'd love to help! May I have your name and zip code first to check delivery for your area?"
- Collect details one by one. Do not ask for everything at once in your follow-up messages.

Initial Interaction:
- For the VERY FIRST message, you must guide them to provide Name, Phone, and Zip Code. 
- Once you have some details, ask for the missing ones (e.g., "Email").
- ONLY after all details (Name, Phone, Email, Zip Code) are collected, discuss specific orders, product parameters, or shipping rates.

Guidelines:
1. Keep verbal responses very short (1-2 sentences max).
2. Be reactive. Ask for missing details one by one.
3. Be professional, warm, and helpful.

Logistics:
- Pickup Locations: {json.dumps(LOGISTICS_RULES['PICKUP_LOCATIONS'])}

Products Available:
{PRODUCT_DATA}

Temperature: Keep responses professional but warm. Temperature: 0.7
"""

# =========================================================================
# 🔧 LANGCHAIN SETUP
# =========================================================================

@cl.on_chat_start
async def start():
    """Initialize the chatbot with LangChain memory and chain."""
    
    # Initialize Ollama LLM
    llm = Ollama(
        model="llama3",
        base_url="http://localhost:11434",
        temperature=0.7,
        num_predict=512,
    )
    
    # Create conversation memory
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        human_prefix="Customer",
        ai_prefix="Mira"
    )
    
    # Create prompt template
    prompt = ChatPromptTemplate(
        messages=[
            SystemMessagePromptTemplate.from_template(SYSTEM_INSTRUCTION),
            HumanMessagePromptTemplate.from_template("{chat_history}\n\nCustomer: {input}\n\nMira:")
        ]
    )
    
    # Create LLM chain
    chain = LLMChain(llm=llm, prompt=prompt, memory=memory, verbose=True)
    
    # Store in session
    cl.user_session.set("chain", chain)
    cl.user_session.set("memory", memory)
    
    # Send welcome message
    await cl.Message(
        content="Hi! Welcome to theindianflowers. 🌸 I'm here to help with your festive decor and orders. To get started, may I have your name, phone number, and zip code please?"
    ).send()


@cl.on_message
async def main(message: cl.Message):
    """Process user messages through the LangChain chain."""
    
    # Get chain and memory from session
    chain = cl.user_session.get("chain")
    
    if not chain:
        await cl.Message(content="Error: Chatbot not initialized. Please refresh the page.").send()
        return
    
    # Run the chain
    try:
        response = await cl.make_async(chain.run)(input=message.content)
        
        # Send response back to user
        await cl.Message(content=response).send()
        
    except Exception as e:
        error_msg = f"Error processing message: {str(e)}"
        print(f"[ERROR] {error_msg}")
        await cl.Message(content="Sorry, I encountered an error. Please try again.").send()


if __name__ == "__main__":
    # Run: chainlit run app_langchain.py
    pass
