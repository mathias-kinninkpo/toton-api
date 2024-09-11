# from langchain.memory import ConversationBufferMemory
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
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.http import models
from langchain_qdrant import QdrantVectorStore
from langchain.prompts import PromptTemplate
import pandas as pd
import os
#from langchain.llms.huggingface_pipeline import HuggingFacePipeline
# from transformers import (
#     pipeline,
#     AutoModelForCausalLM,
#     AutoTokenizer,
#     #     # pad_token_id=tokenizer.eos_token_id,
#     BitsAndBytesConfig,
#     AutoConfig,
# )
#from langchain_community.llms import Ollama
from langchain.docstore.document import Document
from .llama import LlamaLLM

BASE_DIR = os.path.dirname(os.path.abspath(__file__))  

client = QdrantClient(":memory:")
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=50)
sentence_transformer_model = SentenceTransformer("all-MiniLM-L6-v2")
encoder = FastEmbedEmbeddings(embedding_model=sentence_transformer_model)
inference_api_key = os.getenv("HF_TOKEN")

embeddings = HuggingFaceInferenceAPIEmbeddings(
    api_key=inference_api_key,
    model_name="BAAI/bge-m3"
)


def split_text(text):
    return text_splitter.split_text(text)
def create_documents(df):
    # documents = []
    # for _, row in df.iterrows():
    #     chunks = text_splitter.split_text(row['context'])
    #     for chunk in chunks:
    #         documents.append({
    #             "text": chunk,
    #             "metadata": {
    #                 "link": row['links'],
    #                 "category": row['categories']
    #             }
    #         })
    df['context_chunks'] = df['context'].apply(lambda x: split_text(x))
    return df


csv_path = os.path.join(BASE_DIR, "dataset_after_scrapping.csv")


def persist_vectors_in_qdrant(client, collection_name="public_service_embedded", encoder=encoder):

    
    
    #embedding_size = encoder.embedding_model.get_sentence_embedding_dimension()
    #embedding_size = sentence_transformer_model.get_sentence_embedding_dimension()

    collections = client.get_collections()
    if any(collection.name == collection_name for collection in collections.collections):

        print(f"Collection '{collection_name}' already exists. Loading existing vectors...")
        vector_store = QdrantVectorStore(
            client=client,
            collection_name="public_service_embedded",
            embedding=embeddings,    
        )
        return vector_store

    # Si la collection n'existe pas, la créer et persister les vecteurs
    print(f"Creating collection '{collection_name}' and persisting vectors...")
    # Créer la collection dans Qdrant
    client.create_collection(
        collection_name=collection_name,
        vectors_config=models.VectorParams(
            size=1024,
            distance=models.Distance.COSINE
        )
    )
      
    # Initialiser le vector store avec Langchain et Qdrant
    vector_store = Qdrant(client, collection_name=collection_name, embeddings=embeddings)

    df = pd.read_csv(csv_path)

    df = create_documents(df)
    # Extraire les textes et les métadonnées en une seule étape
    #texts = [doc['text'] for doc in documents]
    #metadatas = [doc['metadata'] for doc in documents]
    ## Extraire les textes et les métadonnées en une seule étape
    #texts = [doc['text'] for doc in documents]
    #metadatas = [doc['metadata'] for doc in documents]
    
    ## Ajouter tous les documents en une seule opération
    #vector_store.add_texts(texts=texts, metadatas=metadatas)
    
    texts = []
    metadatas = []

    for _, row in df.iterrows():
        for chunk in row['context_chunks']:
           # Collect the text chunks and metadata
        
            texts.append(chunk)  # Add the chunked text to the texts list
            
            # Prepare the metadata for each chunk
            metadatas.append({
                'links': row['links'], 
                'categories': row['categories'], 
            })
    vector_store.add_texts(texts=texts, metadatas=metadatas)
    return vector_store


from langchain.chains import RetrievalQA
from .llama import LlamaLLM

prompt_template = PromptTemplate.from_template(
    """ 
    <s>[INST] 
    Tu es un assistant conversationnel virtuel spécialisé dans les services publics pour le Bénin en occurence. Ton objectif est de fournir des réponses claires, concises et exactes aux questions des utilisateurs en te basant sur les informations disponibles.
    Si aucune information pertinente n'est trouvée dans le contexte, indique que tu ne peux pas répondre à la question avec les données actuelles

    Voici ce que tu dois faire :
    1. Lis attentivement les informations fournies ci-dessous.
    2. Réponds de manière directe et précise à la question posée en te basant uniquement sur le contexte fourni.
    3. Si plusieurs informations pertinentes sont disponibles, synthétise-les de manière à fournir la réponse la plus complète et utile possible.
    4. Si les informations du contexte sont non coherentes et admettent des erreurs d'hortographes, il faut bien organiser ta réponse de sorte à founir des reponse claire, corretes et concises 
    RAPEL : Si aucune information pertinente n'est trouvée dans le contexte, indique que tu ne peux pas répondre à la question dans la courtoisie!! Si ce n'est pas une question ni une demande, essai de repondre aussi avec courtoisie
    Par exemple si l'on te remercie, tu reponds normalement bien. N'oublie pas de canalyser la discution sur ton objecetif
    N'oublie pas surtout que tu est un chatbot en conversation avec un humain

    
    [/INST]</s>
    
    [INST]
    {question}
    Contexte: {context}
    Réponse:
    [/INST]
    
    """
)


def create_rag_chain(retriever, memory=None, client=client, encoder=encoder, collection_name="public_service_embed"):
    # Set the model id to load the model from HuggingFace
    #model_id = "meta-llama/Meta-Llama-3-8B-Instruct" #context length of 262k
    # While waiting access to Llama model, you can use the falcon model to run the code.
    model_id = "beomi/gemma-ko-2b"

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
    def format_context(retrieved_docs):
        """Customize how the context is presented by manipulating the retrieved documents."""
        personalized_context = ""
        for i, doc in enumerate(retrieved_docs):
            # Extract the relevant fields from the payload
            content = doc.get('payload', {}).get('content', 'No content available')
            link = doc.get('payload', {}).get('link', 'No link available')
            category = doc.get('payload', {}).get('category', 'No category available')
            
            # Format the context with link and category
            personalized_context += f"Document {i+1}:\n"
            personalized_context += f"Content: {content}\n"
            personalized_context += f"Link: {link}\n"
            personalized_context += f"Category: {category}\n"
            personalized_context += "---------------------------------\n"
        return personalized_context
    
    

    cached_llm = LlamaLLM()
    chain = RetrievalQA.from_chain_type(
        llm=cached_llm,
        retriever=retriever,
        return_source_documents=True,
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
        results = retriever(self, query)
        # Retourner les documents de façon compatible avec Langchain
        return [Document(page_content=result['context'], metadata=result['links', 'categories']) for result in results]

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