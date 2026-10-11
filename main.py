from core import (
    get_or_create_session,
    get_last_session,
    process_question,
    save_exchange,
    close_resources,
)

def main():
    print("\n===============================")
    print(" AutoDrive Multi-Agent RAG v2 (avec mémoire Neo4j)")
    print("===============================")
    print("Commandes: 'exit' pour quitter, 'nouvelle' pour nouvelle session\n")

    session_id, history = get_last_session()
    if session_id:
        print(f"[SESSION RECHARGÉE] {session_id} - {len(history)} messages précédents restaurés\n")
    else:
        session_id, history = get_or_create_session()
        if session_id == "no-memory":
            print("[AVERTISSEMENT] Neo4j non disponible - fonctionnement sans mémoire\n")

    while True:
        question = input("Question : ").strip()

        if question.lower() == "exit":
            break

        if question.lower() == "nouvelle":
            session_id, history = get_or_create_session()
            if session_id == "no-memory":
                print("[AVERTISSEMENT] Neo4j non disponible - fonctionnement sans mémoire\n")
            print(f"\n[Nouvelle session démarrée: {session_id}]\n")
            continue

        if not question:
            continue

        result = process_question(question, session_id, history)

        reponse = result["reponse"]
        sources = result["sources"]
        agent_used = result["agent_utilise"]

        print("\n--- RÉPONSE ---")
        print(reponse)

        if sources:
            print("\n--- SOURCES ---")
            for src in sources:
                print(f"  - {src}")

        print(f"\n[Agent utilisé: {agent_used}]\n")

        if session_id != "no-memory":
            save_exchange(session_id, question, reponse)
            history = load_history(session_id)

    close_resources()
    print("Au revoir !")


from memory import load_history

if __name__ == "__main__":
    main()