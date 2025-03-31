import lmstudio as lms
from query_chain import QueryChain
from prompt_handler import PromptHandler
from pdf_retriever import PdfRetriever

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
    
    def generate_response(self, user_input: str) -> str:
        """
        Génère une réponse du LLM pour une entrée utilisateur.

        :param user_input: Texte de l'utilisateur
        :return: Réponse du modèle
        """
        self.chat.add_user_message(user_input)
        response = ""
        for fragment in self.model.respond_stream(
            self.chat,
            on_message=self.chat.append,
        ):
            response += fragment.content
        return response

# Initialisation des composants
document_retriever = PdfRetriever("data/manuel.pdf")
prompt_handler = PromptHandler("prompts/prompt_system.txt", "prompts/prompt_context.txt")
chatbot = ChatBot("You are a task-focused AI assistant")

# Création de la QueryChain avec ChatBot 
query_chain = QueryChain(
    document_retriever,
    prompt_handler.format_docs,
    prompt_handler.prompt,
    chatbot.generate_response,  # On passe la fonction de génération ici
    prompt_handler.output_parser
)

# Exemple d'utilisation avec QueryChain
def run():
    while True:
        user_input = input("You (leave blank to exit): ")
        if not user_input:
            break

        response = query_chain.run(user_input)
        print("Bot:", response)

if __name__ == "__main__":
    run()
