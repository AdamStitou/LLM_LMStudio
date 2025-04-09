from agent import Agent
from relevant_search_agent import RelevantSearchAgent
from LLM_LM_Studio import LLMStudio

class AgentOrchestrator(Agent):
    """
    Orchestrateur qui gère le LLM LMStudio et un agent spécialisé,
    ici un agent de recherche pertinent.
    """
    
    def __init__(self, name: str, tavily_api_key: str):
        super().__init__(name)
        # Initialise le LLM pour les réponses directes
        self.llm = LLMStudio()
        # Initialise l'agent de recherche pertinent
        self.search_agent = RelevantSearchAgent("RelevantAgent", tavily_api_key)
    
    def run(self, prompt: str) -> str:
        """
        Si le prompt contient "search:", on extrait la requête et on délègue
        à l'agent pertinent. Sinon, le LLM répond directement.
        """
        prompt_lower = prompt.lower()
        
        if "search:" in prompt_lower:
            query_part = prompt_lower.split("search:")[1].strip()
            answer = self.search_agent.run(query_part)
            return answer
        else:
            answer = self.llm.respond(prompt)
            return answer
