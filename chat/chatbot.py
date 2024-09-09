from .memory_manager import MemoryManager


class Chatbot:
    """
    Représente un chatbot avec gestion de la mémoire contextuelle.
    """

    def __init__(self):
        """
        Initialise le chatbot avec un gestionnaire de mémoire.
        """
        self.memory = MemoryManager()

    def respond(self, user_input):
        """
        Génère une réponse du chatbot basée sur l'entrée de l'utilisateur et le contexte.

        :param user_input: Message de l'utilisateur
        :return: Réponse du chatbot
        """
        # Ajoute le message de l'utilisateur à la mémoire
        self.memory.add_message("user", user_input)
        
        # Récupère le contexte actuel de la conversation
        context = self.memory.get_context()
        
        # Ici, vous intégreriez votre logique de traitement du langage naturel
        # pour générer une réponse basée sur le contexte de la conversation
        response = f"Réponse basée sur le contexte : {context}"
        
        # Ajoute la réponse du bot à la mémoire
        self.memory.add_message("bot", response)
        
        return response
    
    
    
    # Exemple d'utilisation
if __name__ == "__main__":
    bot = Chatbot()

    # Simulons une conversation
    print(bot.respond("Bonjour !"))
    print(bot.respond("Comment ça va ?"))
    print(bot.respond("Quel temps fait-il aujourd'hui ?"))
    print(bot.respond("Au revoir !"))
    