import lmstudio as lms

class ChatBot:
    """
    Classe représentant un chatbot interactif utilisant lmstudio.
    """
    def __init__(self, system_message: str):
        """
        Initialise le chatbot avec un message système définissant son comportement.

        :param system_message: Message système pour guider l'IA
        """
        self.model = lms.llm()
        self.chat = lms.Chat(system_message)
    
    def run(self):
        """Démarre une boucle interactive pour échanger avec l'IA."""
        while True:
            try:
                user_input = input("You (leave blank to exit): ")
            except EOFError:
                print()
                break
            if not user_input:
                break
            
            self.chat.add_user_message(user_input)
            prediction_stream = self.model.respond_stream(
                self.chat,
                on_message=self.chat.append,
            )
            print("Bot: ", end="", flush=True)
            for fragment in prediction_stream:
                print(fragment.content, end="", flush=True)
            print()

# Exemple d'utilisation
if __name__ == "__main__":
    chatbot = ChatBot("You are a task-focused AI assistant")
    chatbot.run()