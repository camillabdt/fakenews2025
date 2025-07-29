import requests
import pandas as pd
import time

# === CONFIGURAÇÕES ===

API_KEYS = [
      # Substitua pela sua chave de API Groq
]

MODELOS = {
    "qwen": "qwen-qwq-32b",
    "deepseek": "deepseek-r1-distill-llama-70b",
   "gemma": "gemma2-9b-it",
    "llama": "llama-3.1-8b-instant",

}

URL = 'https://api.groq.com/openai/v1/chat/completions'

PROMPT_EXPLICACAO = (
    "Leia o texto abaixo e diga se ele parece uma notícia falsa ou verdadeira. "
    "Depois, explique por quê. Aponte sinais, indicadores ou elementos no conteúdo "
    "que sugerem que seja fake news (ou que reforcem sua veracidade).\n\n"
    "Texto:\n{}"
)

# === FUNÇÃO DE CONSULTA COM EXPLICAÇÃO ===

def call_groq_explicacao(prompt_text, model):
    for key in API_KEYS:
        headers = {"Authorization": f"Bearer {key}"}
        body = {
            "model": model,
            "messages": [
                {"role": "system", "content": "Você é um especialista em checagem de fatos. Analise o texto e explique com base em indícios."},
                {"role": "user", "content": prompt_text}
            ],
            "temperature": 0.7
        }

        try:
            response = requests.post(URL, headers=headers, json=body)
            if response.status_code == 200:
                reply = response.json()['choices'][0]['message']['content'].strip()
                return reply
            else:
                print(f"[{key}] Erro {response.status_code}: {response.text}")
        except Exception as e:
            print(f"Erro com chave {key}: {e}")
        time.sleep(2)
    return "Erro: sem resposta"

# === EXECUÇÃO ===

def analisar_com_explicacoes(input_csv):
    df = pd.read_csv(input_csv)
    df.columns = df.columns.str.strip()

    if 'text' not in df.columns:
        raise ValueError("❌ A coluna 'text' não foi encontrada no CSV.")

    for nome_modelo, modelo in MODELOS.items():
        print(f"\n🔍 Executando análises com {nome_modelo}...")
        explicacoes = []

        for i, row in df.iterrows():
            prompt = PROMPT_EXPLICACAO.format(row['text'])
            print(f"[{i+1}/{len(df)}] Analisando...")
            explicacao = call_groq_explicacao(prompt, modelo)
            explicacoes.append(explicacao)

        df[f'explicacao_{nome_modelo}'] = explicacoes

    df.to_csv("explicacoes_fakenews.csv", index=False)
    print("\n✅ Arquivo 'explicacoes_fakenews.csv' gerado com explicações dos modelos.")

# === PONTO DE ENTRADA ===

if __name__ == "__main__":
    analisar_com_explicacoes("1novo.csv")
