# search_agent.py

import os
from tavily import TavilyClient
from agent import Agent
from datetime import datetime, timedelta

class SearchAgent(Agent):
    """
    Agent spécialisé dans la recherche d'informations en ligne en utilisant le client officiel Tavily.
    """
    def __init__(self, name: str, api_key: str, max_results: int = 15):
        super().__init__(name)
        self.api_key = api_key
        self.max_results = max_results
        # Instanciation du client Tavily
        self.client = TavilyClient(api_key)
        # Optionnel : stocker la clé API dans l'environnement
        os.environ["TAVILY_API_KEY"] = api_key

    def run(self, query: str):
        """
        Effectue une recherche via l'API Tavily en utilisant des paramètres optimisés pour
        récupérer des résultats récents et pertinents.
        """
        # Paramètres pour obtenir des résultats récents et de qualité
        params = {
            "query": query,
            "max_results": self.max_results,
            "search_depth": "advanced",    # Extraction optimisée des extraits pertinents
            "topic": "news",               # Concentre la recherche sur l'actualité
            "days": 7,                     # Inclure uniquement les résultats des 7 derniers jours
            "time_range": "day"            # On peut préciser "day" pour des recherches ultra-récentes
        }
        
        # Effectuer la recherche via le client Tavily
        try:
            response = self.client.search(**params)
        except Exception as e:
            return {"error": "Network error", "message": str(e)}
        
        # Traitement des résultats
        processed_results = []
        for result in response.get("results", []):
            result_url = result.get("url", "URL non disponible")
            content = result.get("content", "")
            # Extraire un extrait du contenu (max 200 caractères)
            excerpt = content[:500] + "..." if len(content) > 500 else content
            # Filtrer par date si la clé "published_date" est présente
            pub_date_str = result.get("published_date", None)
            if pub_date_str:
                try:
                    # On suppose que la date est en format ISO 8601
                    pub_date = datetime.fromisoformat(pub_date_str)
                    # On garde uniquement les articles publiés dans les 7 derniers jours
                    if pub_date < datetime.now() - timedelta(days=7):
                        continue
                except Exception:
                    pass
            processed_results.append({"url": result_url, "excerpt": excerpt})
            print(processed_results)
        return processed_results

def create_tavily_search_agent(api_key: str, max_results: int = 15) -> SearchAgent:
    """
    Fonction utilitaire pour créer et configurer l'agent de recherche Tavily.
    """
    return SearchAgent("SearchAgent", api_key, max_results)
