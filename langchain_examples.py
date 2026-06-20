"""
LangChain + Ollama Example with Advanced Features
- Conversation chains with memory
- Context window management
- Multi-step reasoning with RAG-like patterns
"""

import chainlit as cl
from langchain_community.llms import Ollama
from langchain_community.memory import ConversationBufferMemory  #  Alternative
from langchain.prompts import PromptTemplate, ChatPromptTemplate # type: ignore
from langchain.chains import LLMChain, ConversationChain, SimpleSequentialChain # pyright: ignore[reportMissingImports]
from langchain.schema import HumanMessage, AIMessage, SystemMessage # type: ignore
from langchain_core.output_parsers import StrOutputParser
from typing import List

# =========================================================================
# Example 1: Simple Conversational Chain with Memory
# =========================================================================
def example_simple_conversation():
    """Basic conversation with memory persistence."""
    
    llm = Ollama(
        model="llama3",
        base_url="http://localhost:11434",
        temperature=0.7,
        num_predict=256,
    )
    
    memory = ConversationBufferMemory(return_messages=True)
    
    conversation = ConversationChain(
        llm=llm,
        memory=memory,
        verbose=True
    )
    
    # Have a multi-turn conversation
    response1 = conversation.run("My name is Alice. What's your name?")
    print(f"Bot: {response1}\n")
    
    response2 = conversation.run("What did I just tell you?")
    print(f"Bot: {response2}\n")


# =========================================================================
# Example 2: Question Answering Chain with Custom Prompt
# =========================================================================
def example_qa_chain():
    """Specialized Q&A chain with custom formatting."""
    
    llm = Ollama(
        model="llama3",
        base_url="http://localhost:11434",
        temperature=0.5,
        num_predict=512,
    )
    
    template = """Answer the question based on the context provided.

Context:
{context}

Question: {question}

Answer:"""
    
    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=template
    )
    
    chain = prompt | llm | StrOutputParser()
    
    context = "Python is a high-level programming language known for its simplicity and readability."
    question = "What is Python known for?"
    
    result = chain.invoke({"context": context, "question": question})
    print(f"Answer: {result}\n")


# =========================================================================
# Example 3: Sequential Chains for Multi-Step Tasks
# =========================================================================
def example_sequential_chain():
    """Chained operations - first summarize, then analyze."""
    
    llm = Ollama(
        model="llama3",
        base_url="http://localhost:11434",
        temperature=0.7,
    )
    
    # Step 1: Summarization chain
    summary_template = """Summarize the following text in 2-3 sentences:

{text}

Summary:"""
    
    summary_prompt = PromptTemplate(
        input_variables=["text"],
        template=summary_template
    )
    summary_chain = summary_prompt | llm | StrOutputParser()
    
    # Step 2: Sentiment analysis chain
    sentiment_template = """Analyze the sentiment of the following text:

{text}

Sentiment:"""
    
    sentiment_prompt = PromptTemplate(
        input_variables=["text"],
        template=sentiment_template
    )
    sentiment_chain = sentiment_prompt | llm | StrOutputParser()
    
    # Combine chains
    text = "I loved the movie! It was entertaining and well-made."
    
    summary = summary_chain.invoke({"text": text})
    print(f"Summary: {summary}\n")
    
    sentiment = sentiment_chain.invoke({"text": summary})
    print(f"Sentiment: {sentiment}\n")


# =========================================================================
# Example 4: Using LangChain Expression Language (LCEL)
# =========================================================================
def example_lcel_pipeline():
    """Modern LCEL pipeline with composition."""
    
    llm = Ollama(
        model="llama3",
        base_url="http://localhost:11434",
        temperature=0.7,
    )
    
    # Define pipeline: prompt -> llm -> output_parser
    template = "Tell me a joke about {topic} in one sentence."
    prompt = PromptTemplate(template=template, input_variables=["topic"])
    
    chain = prompt | llm | StrOutputParser()
    
    result = chain.invoke({"topic": "programming"})
    print(f"Joke: {result}\n")


# =========================================================================
# Example 5: Message History Management
# =========================================================================
def example_with_message_history():
    """Manually manage conversation history."""
    
    llm = Ollama(
        model="llama3",
        base_url="http://localhost:11434",
        temperature=0.7,
    )
    
    # Build conversation manually
    messages = [
        SystemMessage(content="You are a helpful assistant."),
    ]
    
    # User message 1
    messages.append(HumanMessage(content="What is the capital of France?"))
    response1 = llm.invoke(messages)
    messages.append(AIMessage(content=response1.content))
    print(f"Q1: {response1.content}\n")
    
    # User message 2 (bot has context)
    messages.append(HumanMessage(content="What is its population?"))
    response2 = llm.invoke(messages)
    messages.append(AIMessage(content=response2.content))
    print(f"Q2: {response2.content}\n")
    
    print(f"Total messages in history: {len(messages)}\n")


# =========================================================================
# Run Examples
# =========================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("LangChain + Ollama Examples")
    print("=" * 60 + "\n")
    
    print("📌 Example 1: Simple Conversation with Memory")
    print("-" * 60)
    example_simple_conversation()
    
    print("\n📌 Example 2: Question Answering Chain")
    print("-" * 60)
    example_qa_chain()
    
    print("\n📌 Example 3: Sequential Chains")
    print("-" * 60)
    example_sequential_chain()
    
    print("\n📌 Example 4: LCEL Pipeline")
    print("-" * 60)
    example_lcel_pipeline()
    
    print("\n📌 Example 5: Message History Management")
    print("-" * 60)
    example_with_message_history()
