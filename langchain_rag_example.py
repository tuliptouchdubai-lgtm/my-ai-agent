"""
LangChain RAG (Retrieval-Augmented Generation) Example
- Document loading and chunking
- Vector embeddings simulation (using simple similarity)
- Context retrieval and answer generation
"""

from langchain_community.llms import Ollama
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from typing import List, Tuple
import json

# =========================================================================
# SAMPLE KNOWLEDGE BASE (simulating product database)
# =========================================================================
PRODUCT_DOCUMENTS = [
    {
        "title": "Jasmine String (5ft)",
        "content": """
Jasmine String is a freshly strung loose jasmine strands product.
Available in both bulk quantities and individual units.
Perfect for weddings, festivals, and special celebrations.
Each strand is carefully crafted and fragrant.
Bulk pricing available for large orders.
"""
    },
    {
        "title": "Rose Petal Garland (RP32)",
        "content": """
Premium hand-crafted wedding garland made from fresh rose petals.
Model: RP32. Exclusively designed for wedding ceremonies.
Each garland is meticulously arranged with premium red and pink roses.
Delivery available to pickup locations: Houston, Tampa, Seattle.
Shelf life: 2-3 days from delivery. Store in cool, dry place.
Pricing starts at $89 for standard size.
"""
    },
    {
        "title": "Shipping Policy",
        "content": """
Pickup Locations: Houston, Tampa, Seattle, Sacramento, Pittsburgh, 
Reminderville (Ohio), Livermore, Greenwood Village (CO), 
Devon (PA 19333), Minnesota (MN 55077), Fontana (CA 92336).

Standard Shipping: 2-3 business days.
Express Shipping: 24 hours (additional $15).
Same-day delivery available for Houston and Sacramento.
Refund policy: Full refund within 7 days if not satisfied.
"""
    },
    {
        "title": "Lilly Garland (LG23)",
        "content": """
Elegant celebratory structural floral garland featuring white lilies.
Model: LG23. Ideal for festivals, anniversaries, and celebrations.
Handcrafted with premium lilies and greenery.
Measurements: 6 feet long, 8 inches wide.
Colors: White, cream, and light green varieties available.
Perfect for entrance decorations and stage backdrops.
"""
    }
]

# =========================================================================
# SIMPLE RETRIEVAL SYSTEM (without vector DB for demo)
# =========================================================================

class SimpleDocumentRetriever:
    """Basic document retriever using keyword matching."""
    
    def __init__(self, documents: List[dict]):
        self.documents = documents
        self.all_content = "\n".join([f"[{doc['title']}]\n{doc['content']}" for doc in documents])
    
    def retrieve(self, query: str, top_k: int = 2) -> str:
        """Simple keyword-based retrieval."""
        query_lower = query.lower()
        
        # Score documents by keyword matching
        scores = []
        for doc in self.documents:
            title_lower = doc['title'].lower()
            content_lower = doc['content'].lower()
            
            # Simple scoring: count keyword matches
            score = 0
            score += title_lower.count(query_lower) * 3  # Title matches weight more
            
            # Match individual words
            for word in query_lower.split():
                if len(word) > 3:  # Only meaningful words
                    score += content_lower.count(word)
            
            scores.append((doc, score))
        
        # Sort by score and return top_k
        sorted_docs = sorted(scores, key=lambda x: x[1], reverse=True)
        
        context = "\n\n".join([f"[{doc['title']}]\n{doc['content']}" 
                               for doc, _ in sorted_docs[:top_k] if _ > 0])
        
        return context if context else self.all_content[:500]  # Fallback


# =========================================================================
# RAG PIPELINE
# =========================================================================

class RAGPipeline:
    """Complete RAG system: Retrieve -> Augment -> Generate."""
    
    def __init__(self, documents: List[dict]):
        self.retriever = SimpleDocumentRetriever(documents)
        self.llm = Ollama(
            model="llama3",
            base_url="http://localhost:11434",
            temperature=0.7,
            num_predict=512,
        )
    
    def generate_response(self, query: str) -> dict:
        """Full RAG pipeline."""
        
        # Step 1: Retrieve relevant documents
        context = self.retriever.retrieve(query, top_k=2)
        
        # Step 2: Create augmented prompt
        template = """You are Mira, a helpful concierge for "theindianflowers".

Use the following context to answer the customer's question. 
If the information is not in the context, say you'll need to check with the team.

Context:
{context}

Customer Question: {question}

Your Response:"""
        
        prompt = PromptTemplate(
            template=template,
            input_variables=["context", "question"]
        )
        
        # Step 3: Generate response
        chain = prompt | self.llm | StrOutputParser()
        response = chain.invoke({"context": context, "question": query})
        
        return {
            "query": query,
            "context_used": context[:200] + "..." if len(context) > 200 else context,
            "response": response
        }


# =========================================================================
# MULTI-TURN RAG CONVERSATION
# =========================================================================

class RAGConversation:
    """Multi-turn conversation with RAG."""
    
    def __init__(self, documents: List[dict]):
        self.rag = RAGPipeline(documents)
        self.conversation_history = []
    
    def chat(self, user_message: str) -> str:
        """Add message to history and generate response."""
        
        # Generate RAG response
        result = self.rag.generate_response(user_message)
        response = result["response"]
        
        # Store in history
        self.conversation_history.append({
            "role": "user",
            "content": user_message
        })
        self.conversation_history.append({
            "role": "assistant",
            "content": response
        })
        
        return response
    
    def print_history(self):
        """Display conversation."""
        for msg in self.conversation_history:
            prefix = "You:" if msg["role"] == "user" else "Mira:"
            print(f"\n{prefix} {msg['content']}")


# =========================================================================
# EXAMPLE USAGE
# =========================================================================

def example_single_query():
    """Answer a single question with context."""
    print("\n" + "="*60)
    print("Example 1: Single Query with RAG")
    print("="*60)
    
    rag = RAGPipeline(PRODUCT_DOCUMENTS)
    
    query = "What's the price of the Rose Petal Garland?"
    result = rag.generate_response(query)
    
    print(f"Query: {result['query']}")
    print(f"\nContext Retrieved:\n{result['context_used']}")
    print(f"\nResponse:\n{result['response']}")


def example_multi_turn():
    """Multi-turn conversation with RAG."""
    print("\n" + "="*60)
    print("Example 2: Multi-Turn Conversation with RAG")
    print("="*60)
    
    conversation = RAGConversation(PRODUCT_DOCUMENTS)
    
    questions = [
        "Hi, I'm interested in products for my wedding.",
        "Tell me about the Rose Petal Garland.",
        "What cities do you deliver to?",
        "How fast is your shipping?",
    ]
    
    for question in questions:
        print(f"\n{'='*60}")
        print(f"Customer: {question}")
        print(f"{'='*60}")
        response = conversation.chat(question)
        print(f"Mira: {response}")
    
    print("\n" + "="*60)
    print("Conversation History:")
    print("="*60)
    conversation.print_history()


def example_batch_qa():
    """Answer multiple questions and compare responses."""
    print("\n" + "="*60)
    print("Example 3: Batch Q&A with RAG")
    print("="*60)
    
    rag = RAGPipeline(PRODUCT_DOCUMENTS)
    
    questions = [
        "What products do you have?",
        "Do you deliver to New York?",
        "Tell me about the Lilly Garland.",
        "What's your refund policy?",
    ]
    
    results = []
    for i, q in enumerate(questions, 1):
        print(f"\n📌 Question {i}: {q}")
        result = rag.generate_response(q)
        results.append(result)
        print(f"Answer: {result['response'][:200]}...")
    
    return results


# =========================================================================
# RUN EXAMPLES
# =========================================================================

if __name__ == "__main__":
    print("\n" + "🎯 "*20)
    print("LangChain RAG (Retrieval-Augmented Generation) Examples")
    print("🎯 "*20)
    
    # Make sure Ollama is running!
    try:
        example_single_query()
        example_multi_turn()
        example_batch_qa()
        
        print("\n" + "="*60)
        print("✅ All examples completed successfully!")
        print("="*60)
    
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("Make sure Ollama is running: ollama serve")
        print("And the model is available: ollama pull llama3")
