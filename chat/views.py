from rest_framework import viewsets
from rest_framework.response import Response
from rest_framework.decorators import action
from .models import Conversation, Message
from .serializers import ConversationSerializer, MessageSerializer
from langchain.memory import ConversationBufferMemory

from .utils import (
    load_conversation, 
    persist_conversation, 
    persist_vectors_in_qdrant,
    create_rag_chain
)

class ConversationViewSet(viewsets.ModelViewSet):
    """
    Viewset pour gérer les conversations.
    """
    queryset = Conversation.objects.all()
    serializer_class = ConversationSerializer

    @action(detail=True, methods=['get'])
    def messages(self, request, pk=None):
        """
        Récupérer tous les messages associés à une conversation spécifique.
        """
        conversation = self.get_object()
        messages = Message.objects.filter(conversation=conversation)
        serializer = MessageSerializer(messages, many=True)
        return Response(serializer.data)

class MessageViewSet(viewsets.ModelViewSet):
    """
    Viewset pour gérer les messages.
    """
    queryset = Message.objects.all()
    serializer_class = MessageSerializer

    def perform_create(self, serializer):
        """
        Lors de la création d'un message, associer automatiquement la conversation à ce message.
        """
        conversation = Conversation.objects.get(pk=self.request.data.get('conversation_id'))
        serializer.save(conversation=conversation)


from rest_framework.views import APIView
from rest_framework.response import Response


from sentence_transformers import SentenceTransformer
from langchain.embeddings import FastEmbedEmbeddings
from qdrant_client import QdrantClient
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
import pandas as pd
from langchain.vectorstores import Qdrant
import os


BASE_DIR = os.path.dirname(os.path.abspath(__file__))  

db_path  = os.path.join(BASE_DIR, "vector_db")

client = QdrantClient(path=db_path)

store = {}

def get_session_history(session_id: str):
    """Retrieve or create session history."""
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

class RagSearchView(APIView):

  def post(self, request):
    query = request.data.get('query')
    session_id = request.data.get('session_id')

    
    if session_id not in store:
        
        conversation, created = Conversation.objects.get_or_create(
            id=session_id,
            defaults={'name': f"Conversation starting with: {query}"}
        )
        
        store[session_id] = ChatMessageHistory()

    else:
        
        conversation = Conversation.objects.get(id=session_id)
        messages = conversation.messages.all().order_by('created_at')
        
        for message in messages:
            sender = "user" if message.sender == "user" else "bot"
            store[session_id].add_message(message)

    vectordb = persist_vectors_in_qdrant(client)
    retriever = vectordb.as_retriever(search_type="similarity", search_kwargs={"k": 5})

    chain = create_rag_chain(retriever)

    chain_with_history = RunnableWithMessageHistory(
        chain,
        get_session_history,
        input_messages_key="query",
        history_messages_key="history"
    )

    # Invoke the chain with the provided query and session-specific configuration
    config = {"configurable": {"session_id": session_id}}
    result = chain_with_history.invoke({"query": query}, config=config)

    # Extract the generated response
    response_text = result['result'].split("Réponse:")[1]

    # Persist the new user query and bot response in the database
    Message.objects.create(
        conversation=conversation,
        sender="user",
        content=query
    )
    Message.objects.create(
        conversation=conversation,
        sender="bot",
        content=response_text
    )

    # Return the generated response and updated conversation history from the database
    return Response({
        "answer": response_text,
        "chat_history": [
            {"sender": message.sender, "content": message.content} for message in conversation.messages.all().order_by('created_at')
        ]  # Return the chat history from the database
    })
