import ollama
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel 

class ChatRequest(BaseModel):
    prompt: str
    subject: str


app = FastAPI()




collections = {
    "Add. Maths":"Add_Maths",
    "Biology":"Biology",
    "Chemistry":"Chemistry",
    "Comp. Sci":"Comp_Sci",
    "English":"English",
    "Maths":"Maths",
    "PEH":"PEH",
    "Physics":"Physics",
    "Social. Studies":"Social_Studies"
}

def getRelatedDocs(question: str, collection, noOfResults=4):
    collection_name = collections.get(collection)

    storage = Chroma(collection_name = collection_name, 
                     embedding_function = OllamaEmbeddings(model="embeddinggemma"), 
                     persist_directory = "./vector_db")
    results = storage.similarity_search(query=question, k=noOfResults)
    
    context_parts = []
    for  document in results:
        source = document.metadata.get("source", "unknown source")

        context_parts.append(f"Source: {source}\n"
                             f"Content: {document}")
        return "\n\n".join(context_parts)

app.add_middleware(CORSMiddleware,
                   allow_origins=["http://127.0.0.1:5500"],
                   allow_credentials=True,
                   allow_methods=["GET", "POST"],
                   allow_headers=["Content-Type"])

conversation = []

@app.post("/chat")
def chat(request: ChatRequest):
    context = getRelatedDocs(question=request.prompt, collection=request.subject)

    system_prompt = """
You are an educational tutor for Senior High School students.

Your job is to help students understand concepts rather than simply
giving them answers.

When teaching:
- Explain concepts in clear, age-appropriate language.
- Break difficult concepts into smaller steps.
- Give examples.
- Ask the student questions to check understanding.
- For mathematics and science, show your working.
- Do not deliberately make up information.
- If you are uncertain, say so.
- Do not pretend that an answer is correct when you are unsure.

Now, all your responses are going to be based on the materials you will be supplied with.
Your responses are to be based solely on these materials. Do not provide any answers that do not
relate to the materials you have been given. If the question does not fall within the scope of the
learning materials you hae been supplied with, say so, and if you are to generate a response without
 the materials, make sure to inform the student when you do so.
These are the materials: 
{context} 
"""

    conversation.append({"role":"system", "content":system_prompt})
    def ai_response():
        full_result = ""
        conversation.append({"role":"user", "content":request.prompt, "subject":request.subject})
        response = ollama.chat(
            model="qwen2.5:1.5b-instruct-q5_K_M",
            messages = conversation,
            stream = True
        )
        for chunk in response:
            content = chunk["message"]["content"]

            if content:
                full_result += content
                yield content
        conversation.append({"role":"assistant", "content":full_result})

    return StreamingResponse(ai_response(), media_type="text/plain")

