from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate

class PromptHandler:
    def __init__(self, system_prompt_file: str, context_prompt_file: str):
        """
        Initialise le gestionnaire de prompt avec les fichiers de template.

        :param system_prompt_file: Chemin du fichier contenant le prompt système.
        :param context_prompt_file: Chemin du fichier contenant le prompt avec contexte.
        """
        self.prompt = self._create_prompt(system_prompt_file, context_prompt_file)
        self.output_parser = StrOutputParser()

    def _create_prompt(self, system_prompt_file, context_prompt_file):
        """
        Crée un prompt à partir des fichiers de template.

        :return: Objet ChatPromptTemplate configuré.
        """
        return ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template_file(system_prompt_file, input_variables=[]),
            SystemMessagePromptTemplate.from_template_file(context_prompt_file, input_variables=["context"]),
            HumanMessagePromptTemplate.from_template("{question}")
        ])

    @staticmethod
    def format_docs(docs):
        """
        Formate les documents récupérés en une chaîne de texte.

        :param docs: Liste de documents.
        :return: Chaîne de texte formatée.
        """
        return "\n\n".join(doc.page_content for doc in docs)
