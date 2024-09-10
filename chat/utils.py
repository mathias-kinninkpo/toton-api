from langchain.memory import ConversationBufferMemory
from .models import Conversation
from .models import Message



def persist_conversation(conversation_id, messages):
    """ Fonction pour persister l'historique des conversations dans la base de données """
    conversation = Conversation.objects.get(id=conversation_id)
    for message in messages:
        Message.objects.create(
            conversation=conversation,
            sender=message['sender'],
            content=message['content']
        )

def load_conversation(conversation_id):
    """ Charger l'historique des messages depuis la base de données """
    conversation = Conversation.objects.get(id=conversation_id)
    messages = conversation.messages.all().order_by('created_at')
    return [
        {"sender": message.sender, "content": message.content}
        for message in messages
    ]



from langchain.vectorstores import Qdrant
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from langchain.embeddings import FastEmbedEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.http import models
from langchain_qdrant import QdrantVectorStore
from langchain.prompts import PromptTemplate
import pandas as pd
import os
from langchain.llms.huggingface_pipeline import HuggingFacePipeline
from transformers import (
    pipeline,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    AutoConfig,
)
from langchain_community.llms import Ollama
from langchain.docstore.document import Document
from .llama import LlamaLLM
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))  
cached_llm = LlamaLLM()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=50)

def create_documents(df):
    documents = []
    for _, row in df.iterrows():
        chunks = text_splitter.split_text(row['context'])
        for chunk in chunks:
            documents.append({
                "text": chunk,
                "metadata": {
                    "link": row['links'],
                    "category": row['categories']
                }
            })
    return documents


csv_path = os.path.join(BASE_DIR, "dataset_after_scrapping.csv")


def persist_vectors_in_qdrant(client, collection_name="public_service_embed"):

    
   
    sentence_transformer_model = SentenceTransformer("all-MiniLM-L6-v2")
    encoder = FastEmbedEmbeddings(embedding_model=sentence_transformer_model)

    embedding_size = sentence_transformer_model.get_sentence_embedding_dimension()

    collections = client.get_collections()
    if any(collection.name == collection_name for collection in collections.collections):

        print(f"Collection '{collection_name}' already exists. Loading existing vectors...")
        vector_store = QdrantVectorStore(
            client=client,
            collection_name="public_service_embed",
            embedding=encoder,    
        )
        return vector_store

     # Si la collection n'existe pas, la créer et persister les vecteurs
    print(f"Creating collection '{collection_name}' and persisting vectors...")

    # Créer la collection dans Qdrant
    client.create_collection(
        collection_name=collection_name,
        vectors_config=models.VectorParams(
            size=embedding_size,
            distance=models.Distance.COSINE
        )
    )

      
    # Initialiser le vector store avec Langchain et Qdrant
    vector_store = Qdrant(client, collection_name=collection_name, embeddings=encoder)

    df = pd.read_csv(csv_path)

    documents = create_documents(df)

    # Extraire les textes et les métadonnées en une seule étape
    texts = [doc['text'] for doc in documents]
    metadatas = [doc['metadata'] for doc in documents]

    # Initialiser le vector store avec Langchain et Qdrant
    vector_store = Qdrant(client, collection_name=collection_name, embeddings=encoder)

    # Extraire les textes et les métadonnées en une seule étape
    texts = [doc['text'] for doc in documents]
    metadatas = [doc['metadata'] for doc in documents]

    # Ajouter tous les documents en une seule opération
    vector_store.add_texts(texts=texts, metadatas=metadatas)

    return vector_store


from langchain.chains import RetrievalQA
from .llama import LlamaLLM

# Définir le prompt template pour structurer la requête
prompt_template = PromptTemplate.from_template(
    """ 
    <s>[INST] 
    Tu es un assistant virtuel spécialisé dans les services publics. Ton objectif est de fournir des réponses claires, concises et exactes aux questions des utilisateurs en te basant sur les informations disponibles.
    Si aucune information pertinente n'est trouvée dans le contexte, indique que tu ne peux pas répondre à la question avec les données actuelles

    Voici ce que tu dois faire :
    1. Lis attentivement les informations fournies ci-dessous.
    2. Réponds de manière directe et précise à la question posée en te basant uniquement sur le contexte fourni.
    3. Si plusieurs informations pertinentes sont disponibles, synthétise-les de manière à fournir la réponse la plus complète et utile possible.
    RAPEL : Si aucune information pertinente n'est trouvée dans le contexte, indique que tu ne peux pas répondre à la question avec les données actuelles.


    
    [/INST]</s>
    

    {question}
    Contexte: {context}
    Réponse:

    
    """
)


def create_rag_chain(retriever, memory=None):
    # Set the model id to load the model from HuggingFace
    #model_id = "meta-llama/Meta-Llama-3-8B-Instruct" #context length of 262k
    # While waiting access to Llama model, you can use the falcon model to run the code.
    model_id = "beomi/gemma-ko-2b"
    HF_KEY  = os.getenv("HF_KEY")
    # # Load the default tokenizer for the selected model
    # tokenizer = AutoTokenizer.from_pretrained(model_id)
    # tokenizer.pad_token_id = tokenizer.eos_token_id

   
    # model = AutoModelForCausalLM.from_pretrained(
    #     model_id,
    #     device_map="auto",
    #     trust_remote_code=True,
    #     load_in_4bit=False
    # )

    # Wrap the model and tokenizer into a text generation pipeline
    # hf_pipeline = pipeline(
    #     "text-generation",
    #     model="beomi/gemma-ko-2b",
    #     # tokenizer=tokenizer,
    #     # temperature=0.1,
    #     # repetition_penalty=1.2,
    #     # pad_token_id=tokenizer.eos_token_id,
    #     token="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"
    # )
    #model_pipeline = HuggingFacePipeline(pipeline=hf_pipeline)
    chain = RetrievalQA.from_chain_type(
        llm=cached_llm,
        retriever=retriever,
        return_source_documents=False,
        chain_type="stuff",
        chain_type_kwargs={"prompt": prompt_template},
        #memory = memory
    )
    return chain











from langchain.schema import BaseRetriever

def search_similarity(question, client, encoder):
    hits = client.search(
    collection_name="public_service",
    query_vector=encoder.embed_query(question),
    limit=3,
    )
    return hits

def retriever(question):
    hits = search_similarity(question)
    responses = [hit.payload for hit in hits]
    return responses

class CustomQdrantRetriever(BaseRetriever):
    def __init__(self, client, encoder, collection_name):
        self.client = client
        self.encoder = encoder
        self.collection_name = collection_name

    def get_relevant_documents(self, query):
        # Utiliser les fonctions de similarité
        results = retriever(query, self.encoder, self.client, self.collection_name)
        # Retourner les documents de façon compatible avec Langchain
        return [Document(page_content=result['context'], metadata=result) for result in results]

# Créer une instance de ton retriever personnalisé
#retriever = CustomQdrantRetriever(client, encoder, collection_name="public_service_assitant")




def rag_prompt(query, responses):
    """
    Prompt RAG pour la recherche d'informations.


    Args:
        query: La question de l'utilisateur.
        responses: Liste des documents ou des réponses récupérées contenant le contexte pertinent.


    Returns:
        Le prompt RAG
    """


    # Génération du contexte à partir des réponses
    contextes = "\n".join([f"Contexte {i+1}: {response.payload['context']}" for i, response in enumerate(responses)])
    categories = ", ".join(set(response.payload['categories'] for response in responses))


    return f"""
    Tu es un assistant virtuel spécialisé dans les services publics. Ton objectif est de fournir des réponses claires, concises et exactes aux questions des utilisateurs en te basant sur les informations disponibles.


    Voici ce que tu dois faire :
    1. Lis attentivement les informations fournies ci-dessous.
    2. Réponds de manière directe et précise à la question posée en te basant uniquement sur le contexte fourni.
    3. Si plusieurs informations pertinentes sont disponibles, synthétise-les de manière à fournir la réponse la plus complète et utile possible.
    4. Si aucune information pertinente n'est trouvée dans le contexte, indique que tu ne peux pas répondre à la question avec les données actuelles.


    Informations contextuelles :
    {contextes}


    Catégories de services concernées : {categories}


    Question de l'utilisateur : {query}


    Réponse :
    """