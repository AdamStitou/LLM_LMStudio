from langchain_core.runnables import RunnablePassthrough, RunnableLambda

class QueryChain:
    """
    Chaîne de traitement pour récupérer des documents, 
    générer une réponse et parser la sortie.
    """
    def __init__(self, doc_retriever, format_doc, prompt, llm, output_parser):
        """
        Initialise la chaîne de traitement.

        :param doc_retriever: Retrieveur de documents
        :param format_doc: Fonction de formatage des documents
        :param prompt: ChatPromptTemplate (objet)
        :param llm: Modèle de langage (doit être callable ou Runnable)
        :param output_parser: Parser de sortie (doit être callable ou Runnable)
        """
        self.chain = self._build_chain(doc_retriever, format_doc, prompt, llm, output_parser)
    
    def _build_chain(self, doc_retriever, format_doc, prompt, llm, output_parser):
        """Construit la chaîne de traitement en combinant les éléments."""
        
        # Utiliser le retrieveur pour obtenir des documents pertinents
        retriever = doc_retriever.get_retriever()
        
        # Récupérer les documents pertinents (attention : ici "Some query" est un placeholder)
        documents = retriever.get_relevant_documents("Some query")  # Remplacer "Some query" par la question réelle lors de l'exécution
        formatted_docs = format_doc(documents)
        
        # Conserver le contexte pré-calculé
        context = formatted_docs
        
        # Définir une fonction qui ajoute le contexte aux entrées.
        def initial_func(inputs: dict) -> dict:
            # On attend inputs = {"question": <question de l'utilisateur>}
            # On ajoute le contexte fixe
            return {"context": context, "question": inputs.get("question")}
        
        # Définir une fonction de formatage du prompt à partir de votre ChatPromptTemplate
        def format_prompt(inputs: dict) -> str:
            # Ici, `prompt` est votre ChatPromptTemplate ; on l'utilise pour formater le prompt.
            return prompt.format(**inputs)
        
        # Convertir les fonctions en Runnables avec RunnableLambda
        runnable_initial_func = RunnableLambda(initial_func)
        runnable_format_prompt = RunnableLambda(format_prompt)

        # Construire le pipeline en s'assurant que chaque étape est callable
        return (
            runnable_initial_func
            | runnable_format_prompt  # Votre fonction qui renvoie une chaîne de caractères
            | llm
            | output_parser
        )
    
    def run(self, question):
        """Exécute la chaîne sur une question donnée."""
        return self.chain.invoke({"question": question})
