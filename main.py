# main.py

from orchestrator import AgentOrchestrator

def main():
    tavily_api_key = "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"
    orchestrator = AgentOrchestrator("OrchestrateurLMStudio", tavily_api_key)
    
    print("=== Démarrage du chatbot ===")
    print("Tape 'exit' pour quitter.\n")
    
    while True:
        user_input = input("Vous: ")
        if not user_input or user_input.lower() == "exit":
            print("Fin de la session.")
            break
        
        response = orchestrator.run(user_input)
        print("Bot:", response)
        print()

if __name__ == "__main__":
    main()