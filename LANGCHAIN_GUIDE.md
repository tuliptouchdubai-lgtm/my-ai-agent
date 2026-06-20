# LangChain with Ollama Examples 🚀

This workspace now includes LangChain integration with Ollama for building AI applications.

## 📁 Files Overview

| File | Purpose |
|------|---------|
| `app.py` | Original Chainlit app with raw Ollama |
| `app_langchain.py` | **Refactored with LangChain chains, memory, and prompts** |
| `langchain_examples.py` | 5 different LangChain patterns you can learn from |
| `langchain_rag_example.py` | Advanced RAG (Retrieval-Augmented Generation) example |
| `requirements.txt` | All dependencies (LangChain, Chainlit, Ollama) |

## 🛠️ Setup

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Start Ollama Server
```bash
ollama serve
ollama pull llama3  # or your preferred model
```

### 3. Run the Apps

**Option A: LangChain-based Chatbot (Recommended)**
```bash
chainlit run app_langchain.py
```

**Option B: Run Examples**
```bash
python langchain_examples.py
```

**Option C: Run RAG Example**
```bash
python langchain_rag_example.py
```

## 📚 Key LangChain Concepts

### 1. **LLM Integration**
```python
from langchain_community.llms import Ollama

llm = Ollama(model="llama3", base_url="http://localhost:11434")
response = llm.invoke("What is Python?")
```

### 2. **Memory Management**
```python
from langchain.memory import ConversationBufferMemory

memory = ConversationBufferMemory(return_messages=True)
# Automatically tracks conversation history
```

### 3. **Prompt Templates**
```python
from langchain.prompts import PromptTemplate

template = "Question: {question}\nAnswer:"
prompt = PromptTemplate(template=template, input_variables=["question"])
```

### 4. **Chains (LCEL - LangChain Expression Language)**
```python
chain = prompt | llm | output_parser
result = chain.invoke({"question": "What is AI?"})
```

### 5. **Conversation Chains**
```python
from langchain.chains import ConversationChain

conversation = ConversationChain(llm=llm, memory=memory)
response = conversation.run("Hello, how are you?")
```

## 🎯 What's Better About LangChain Version?

✅ **Memory Management** - Automatic conversation history tracking  
✅ **Modular Chains** - Reusable, composable components  
✅ **Prompt Templates** - Consistent, parameterized prompts  
✅ **Output Parsing** - Structured responses  
✅ **Sequential Operations** - Chain multiple steps together  
✅ **Error Handling** - Built-in validation  
✅ **Expression Language** - Modern, pythonic API  

## 🔄 Comparing Old vs New

### Before (Raw Ollama)
```python
import ollama
response = ollama.chat(model="llama3", messages=messages)
```

### After (LangChain)
```python
from langchain_community.llms import Ollama
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain

llm = Ollama(model="llama3")
memory = ConversationBufferMemory()
chain = ConversationChain(llm=llm, memory=memory)
response = chain.run("Your question here")
```

## 💡 Next Steps

1. **Customize the flower shop chatbot** (`app_langchain.py`)
   - Add retrieval for product database
   - Implement order tracking chains
   - Add validation for customer details

2. **Build RAG System** (see `langchain_rag_example.py`)
   - Load product documentation
   - Vector embeddings for semantic search
   - Context-aware responses

3. **Add Tools & Agents**
   - LangChain agents for multi-step tasks
   - Function calling for business logic
   - Integration with external APIs

## 📖 Resources

- [LangChain Docs](https://python.langchain.com/)
- [Ollama Models](https://ollama.ai/library)
- [Chainlit Docs](https://docs.chainlit.io/)

Happy coding! 🎉
