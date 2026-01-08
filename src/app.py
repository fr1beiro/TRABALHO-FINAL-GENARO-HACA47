import json
import os
import sys

# Adiciona a pasta src ao caminho para que o pwd falso seja encontrado
sys.path.append(os.path.dirname(__file__))

from rag import create_rag

def modo_chat(qa):
    print("\n--- Assistente Acadêmico UFBA ---")
    print("Digite sua pergunta ou 'sair'")

    while True:
        pergunta = input("\nPergunta: ")
        if pergunta.lower() == "sair":
            break

        try:
            # CORREÇÃO: Usando invoke com a chave 'query'
            resposta = qa.invoke({"query": pergunta})
            print("\nResposta:", resposta["result"])
        except Exception as e:
            print(f"\nErro ao processar: {e}")

def modo_json(qa, caminho_input):
    if not os.path.exists(caminho_input):
        print(f"Erro: O arquivo {caminho_input} não foi encontrado.")
        return

    with open(caminho_input, "r", encoding="utf-8") as f:
        perguntas = json.load(f)

    resultados = []
    print(f"\nProcessando {len(perguntas)} perguntas...\n")

    for i, item in enumerate(perguntas, 1):
        pergunta = item["pergunta"]
        print(f"[{i}] {pergunta}")

        try:
            # CORREÇÃO: Garantindo que a chave 'query' seja enviada
            # Isso resolve o erro 'Missing some input keys' das suas imagens
            resposta_ia = qa.invoke({"query": pergunta})
            texto_resposta = resposta_ia["result"]

            resultados.append({
                "id": i,
                "pergunta": pergunta,
                "resposta_referencia": item.get("resposta", ""),
                "resposta_ia_ufba": texto_resposta
            })
            print("✔ Concluído.")
        except Exception as e:
            print(f"❌ Erro na pergunta {i}: {e}")
            resultados.append({"id": i, "pergunta": pergunta, "erro": str(e)})

    caminho_saida = "data/respostas_finais.json"
    with open(caminho_saida, "w", encoding="utf-8") as f:
        json.dump(resultados, f, indent=4, ensure_ascii=False)

    print(f"\n✔ Processo concluído! Respostas salvas em: {caminho_saida}")

def main():
    print("Carregando sistema RAG UFBA...")
    qa = create_rag()

    print("\n1 - Chat interativo\n2 - Responder perguntas do JSON")
    opcao = input("Opção: ")

    if opcao == "1":
        modo_chat(qa)
    elif opcao == "2":
        # Verificando se o arquivo está como .JSON ou .json conforme sua pasta
        caminho = "data/perguntas.JSON" 
        modo_json(qa, caminho)
    else:
        print("Opção inválida.")

if __name__ == "__main__":
    main()