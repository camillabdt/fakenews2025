import requests
import pandas as pd
import time
import os

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

PROMPT_FAKE_FEATURES = (
'''
Analise o texto de entrada abaixo e avalie a presença das seguintes 16 features de fake news, atribuindo a cada uma uma feature de 0 (ausente) a 10 (explicitamente presente), conforme definições:
- Na coluna 1, adicione um ID incremental no formato #01, #02, etc.];
- Na coluna 2, adicione a pontuação para a feature "Exagero/Sensacionalismo", que se refere ao uso de linguagem hiperbólica (ex: "a mais venenosa do mundo", "literalmente admitiu");
- Na coluna 3, adicione a pontuação para a feature "Falta de Fontes Confiáveis", que se refere à ausência de referências verificáveis ou uso de plataformas duvidosas (ex: "Truth Feed", "Milo.com");
- Na coluna 4, adicione a pontuação para a feature "Linguagem Emocional/Pejorativa", que se refere ao uso de termos carregados de emoção ou adjetivos desqualificadores (ex: "Gun Nuts", "fascist antifa");
- Na coluna 5, adicione a pontuação para a feature "Dados Imprecisos/Vagos", que se refere ao uso de números/estatísticas sem fonte ou contextos não verificáveis (ex: "514 misdemeanors", "várias vítimas");
- Na coluna 6, adicione a pontuação para a feature "Viés/Narrativa Tendenciosa", que se refere ao posicionamento unilateral sem contra-argumentos (ex: "extremistas de direita");
- Na coluna 7, adicione a pontuação para a feature "Falta de Contexto", que se refere ao uso de informações isoladas sem histórico relevante (ex: "golpe na Turquia" sem datas/detalhes);
- Na coluna 8, adicione a pontuação para a feature "Generalizações/Stereótipos", que se refere ao uso de afirmações amplas não sustentadas (ex: "todos os artistas brancos foram excluídos");
- Na coluna 9, adicione a pontuação para a feature "Apelo à Urgência/Medo", que se refere à criação artificial de emergência (ex: "LIVE FEED", "risco de morte iminente");
- Na coluna 10, adicione a pontuação para a feature "Uso de Fontes Duvidosas", que se refere à citação a plataformas conhecidas por desinformação (ex: "Truth Feed", artigos de opinião como fatos);
- Na coluna 11, adicione a pontuação para a feature "Contradições Lógicas", que se refere a inconsistências internas (ex: "fascist antifa");
- Na coluna 12, adicione a pontuação para a feature "Erros Gramaticais/Formais", que se refere a falhas básicas de redação (ex: "Turkey s coup");
- Na coluna 13, adicione a pontuação para a feature "Seletividade Factual", que se refere à omissão intencional de contextos relevantes (ex: citar apenas precedentes que apoiam a narrativa);
- Na coluna 14, adicione a pontuação para a feature "Acusação/Responsabilização", que se refere à atribuição de culpa sem prova (ex: "estão prevenindo o Supremo Tribunal de funcionar");
- Na coluna 15, adicione a pontuação para a feature "Simplificação Excessiva", que se refere à redução de temas complexos a slogans (ex: associar hip-hop diretamente ao racismo);
- Na coluna 16, adicione a pontuação para a feature "Apelo a Teorias da Conspiração", que se refere à alegações não comprovadas (ex: "celebridades estão caladas por conspiração");
- Na coluna 17, adicione a pontuação para a feature "Agenda Política Explícita", que se refere à defesa clara de posicionamento ideológico (ex: "expulsar republicanos por gerações");
- Na coluna 18, adicione uma justificativa breve para cada feature com pontuação ≥5, utilizando trechos relevantes do texto, de forma bem concisa.

- Formate a saída em CSV sem cabeçalho, na ordem numérica das features, sem incluir qualquer outro texto ou formatação adicional, onde cada pontuação é separada por vírgula e cada justificativa é separada por ponto e vírgula.

Texto:
{}'''
)

def call_model(prompt_text, model_name, api_key):
    headers = {"Authorization": f"Bearer {api_key}"}
    body = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": "Você é um especialista em detecção de fake news."},
            {"role": "user", "content": prompt_text}
        ],
        "temperature": 0.7
    }

    try:
        response = requests.post(URL, headers=headers, json=body)
        if response.status_code == 200:
            return response.json()['choices'][0]['message']['content'].strip()
        else:
            return f"[HTTP {response.status_code}] {response.text}"
    except Exception as e:
        return f"[EXCEPTION] {e}"

def analisar_noticia(input_csv):
    df = pd.read_csv(input_csv)
    df.columns = df.columns.str.strip()

    if 'text' not in df.columns:
        raise ValueError("❌ A coluna 'text' não foi encontrada no CSV.")

    n = 750  # número de linhas a ignorar

    for idx, row in df.iloc[n:].iterrows():
        texto = row['text']
        print(f"\n🚀 Analisando linha {idx + 1}/{len(df)}")
        print("📰 Texto:")
        print(texto[:200] + "...\n" if len(texto) > 200 else texto)

        prompt = PROMPT_FAKE_FEATURES.format(texto)

        for i, (apelido, modelo) in enumerate(MODELOS.items()):
            api_key = API_KEYS[i % len(API_KEYS)]
            print(f"🔎 Modelo: {modelo}")
            resposta = call_model(prompt, modelo, api_key)

            # Define o nome do arquivo por modelo
            csv_filename = f"{apelido}_fake.csv"
            linha = f"{apelido},{repr(texto)},{resposta.strip()}"
            
            # Escreve cabeçalho se o arquivo ainda não existir
            if not os.path.exists(csv_filename):
                with open(csv_filename, "w", encoding="utf-8") as f:
                    f.write("modelo,texto,resposta\n")
            
            with open(csv_filename, "a", encoding="utf-8") as f:
                f.write(linha + "\n")

            print(f"✅ Adicionado ao arquivo: {csv_filename}")
            time.sleep(10)

# === PONTO DE ENTRADA ===
if __name__ == "__main__":
    analisar_noticia("1.csv")
