# LearnAid
LearnAid is a web application powered by local Large Language models from Ollama. These AI models provide responses based on the GES-approved learning materials and, for that matter, is most helpful in guiding Senior High Schools with their studies. It is most suited for regular conversations and provides information with regards to the learning materials from which its results are derived, aiding students in their studies based on the nationally accepted curriculum. 
# How it works
Basically, this web app is a RAG chatbot (RAG stands for Retrieval-Augmented Generation). In the backend, embeddings (numerical representations of text) are created and stored in a vector database. This vector database is capable of identifying text which are related to each other or have the same meaning through their closeness in the database. And this works regardless of the wording or grammatical structuring of the 2 texts.
So text was extracted from the learner manuals and embeddings were created and stored in the vector database. A local AI model was downloaded from Ollama (an online platform from which local LLMs  can be accessed), and this model serves as the AI providing guidance to the student. But before the question is fed to the LLM, it is first converted to an embedding and sent to the vector database, which compares it to the learner manuals in order to obtain chunks of text with similar meaning. This information is then fed to the LLM with an instruction to provide answers based solely on the provided context (the text from the database similar to the question). The results obtained are the displayed on the User Interface for the student like a chat.
# Its components
There are various packages and libraries necessary to ensure that the project fully functions, and they are as follows;
1. FastAPI (the API used to connect the backend to the frontend).
2. Uvicorn (the server which runs the backend and listens for HTTP requests)
3. LangChain-Ollama (LangChain's integration with Ollama which permits us to use Ollama's embedding models - embeddinggemma was used)
4. Ollama (the package which enables us to communicate with Ollama's AI models)
5. Pydantic (the class used with FastAPI for data validation of the HTTP requests)
6. Pathlib (this enables us to handles file paths, locations and names)
7. LangChain-Chroma (LangChain's integration with ChromaDB; the vector database storing our embeddings)
8. LangChain-Community.document_loaders (PyPDFLoader was obtained from this utility to extract text from the learner manuals)
9. LangChain-Text_Splitter (The RecursiveCharacterTextSplitter method is obtained from this utility in order to divide our documents into chunks of text)
