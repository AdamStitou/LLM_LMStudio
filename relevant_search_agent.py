# relevant_search_agent.py

from agent import Agent
from search_agent import SearchAgent, create_tavily_search_agent
from LLM_LM_Studio import LLMStudio

class RelevantSearchAgent(Agent):
    """
    Agent qui effectue une recherche via Tavily et utilise le LLM pour 
    sélectionner et synthétiser les résultats les plus pertinents d'après la requête.
    """
    def __init__(self, name: str, tavily_api_key: str, max_results: int = 5):
        super().__init__(name)
        # Réinstanciation du LLM pour une session isolée
        self.llm = LLMStudio()
        # Instanciation de l'agent de recherche optimisé
        self.search_agent = SearchAgent("SearchAgent", create_tavily_search_agent(tavily_api_key, max_results).api_key, max_results)
    
    def run(self, query: str) -> str:
        """
        Effectue la recherche, construit un prompt de classement et renvoie une synthèse.
        """
        # Exécute la recherche en temps réel
        raw_results = self.search_agent.run(query)
        
        if not raw_results or (isinstance(raw_results, dict) and raw_results.get("error")):
            return "Aucun résultat pertinent n'a été trouvé pour votre requête."

        # Construction d'un prompt renforcé pour que le LLM se base uniquement sur ces extraits
        ranking_prompt = (
            "Tu es un assistant qui doit répondre uniquement en te basant sur les extraits suivants, sans utiliser tes connaissances internes.\n"
            "Si les extraits sont insuffisants, indique-le clairement.\n\n"
            f"Requête de l'utilisateur : \"{query}\"\n\n"
            "Résultats bruts obtenus :\n"
        )
        
        for idx, result in enumerate(raw_results, start=1):
            url = result.get("url", "URL non disponible")
            excerpt = result.get("excerpt", "Aucun extrait")
            ranking_prompt += (
                f"Résultat {idx} :\n"
                f"- URL : {url}\n"
                f"- Extrait : {excerpt}\n\n"
            )
        
        ranking_prompt += (
            "Ta tâche : sélectionne les résultats les plus pertinents en expliquant brièvement pourquoi, "
            "puis fais une synthèse claire uniquement basée sur ces extraits."
        )
        
        answer = self.llm.respond(ranking_prompt)
        print(answer)
        return answer

