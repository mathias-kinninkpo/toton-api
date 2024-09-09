import time

class MemoryManager:
    """
    Gère la mémoire contextuelle du chatbot.
    Stocke l'historique des conversations et fournit des méthodes pour
    ajouter des messages, obtenir le contexte et nettoyer les anciens messages.
    """

    def __init__(self, max_memory=20, expiry_time=7200):
        """
        Initialise le gestionnaire de mémoire.

        :param max_memory: Nombre maximum de messages à conserver (défaut: 20)
        :param expiry_time: Durée en secondes avant qu'un message n'expire (défaut: 7200 secondes = 2 heures)
        """
        self.conversation_history = []
        self.max_memory = max_memory
        self.expiry_time = expiry_time

    def add_message(self, role, content):
        """
        Ajoute un nouveau message à l'historique des conversations.

        :param role: Rôle de l'expéditeur (ex: "user" ou "bot")
        :param content: Contenu du message
        :raises ValueError: Si max_memory est négatif ou si expiry_time est nul ou négatif
        """
        if self.max_memory <= 0 or self.expiry_time <= 0:
            raise ValueError("max_memory doit être positif et expiry_time doit être positif")
        timestamp = time.time()
        self.conversation_history.append({"role": role, "content": content, "timestamp": timestamp})
        self._clean_old_messages()

    def get_context(self):
        """
        Renvoie le contexte récent de la conversation.

        :return: Liste des messages récents formatés
        """
        return [f"{msg['role']}: {msg['content']}" for msg in self.conversation_history[-self.max_memory:]]

    def _clean_old_messages(self):
        """
        Nettoie les messages expirés et limite la taille de l'historique.
        """
        current_time = time.time()
        self.conversation_history = [
            msg for msg in self.conversation_history if current_time - msg["timestamp"] < self.expiry_time
        ]
        if len(self.conversation_history) > self.max_memory:
            self.conversation_history = self.conversation_history[-self.max_memory:]