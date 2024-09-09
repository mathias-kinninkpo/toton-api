from rest_framework import serializers
from .models import Conversation, Message

class MessageSerializer(serializers.ModelSerializer):
    class Meta:
        model = Message
        fields = ['id', 'conversation', 'sender', 'content', 'created_at']
        read_only_fields = ['conversation', 'created_at']

class ConversationSerializer(serializers.ModelSerializer):
    # Liste des messages associés à la conversation
    messages = MessageSerializer(many=True, read_only=True)

    class Meta:
        model = Conversation
        fields = ['id', 'created_at', 'updated_at', 'status', 'messages']
        read_only_fields = ['created_at', 'updated_at']
