import ollama, re

messages = [
    {"role":"system", "content":"Test JSON output only. Respond with a JSON object containing text and suggestions."},
    {"role":"user", "content":"Hi"}
]

try:
    resp = ollama.chat(model="llama3", messages=messages, options={"temperature":0.2})
    raw = resp['message']['content']
    print('---RAW START---')
    print(raw)
    print('---RAW END---')
    m = re.search(r'(\{(?:.|\s)*\}|\[(?:.|\s)*\])', raw)
    if m:
        print('\n---EXTRACTED JSON---')
        print(m.group(1))
except Exception as e:
    print('ERROR', e)
