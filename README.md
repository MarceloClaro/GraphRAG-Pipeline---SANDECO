
# GraphRAG Pipeline Visualizer: Framework de Auditoria e Recuperação Aumentada por Grafos

![Status](https://img.shields.io/badge/Status-Auditoria_Técnica_Qualis_A1-blue?style=for-the-badge)
![Tech Stack](https://img.shields.io/badge/Stack-React_|_Gemini_2.0_|_D3.js_|_TensorFlow-indigo?style=for-the-badge)
![Innovation](https://img.shields.io/badge/Innovation-Triangulated_Supervision-orange?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

**Autor Responsável:** Prof. Marcelo Claro Laranjeira

---

## 📑 1. Resumo Executivo e Inovação

Este repositório hospeda a implementação de referência de uma pipeline **Multi-Stage GraphRAG (Graph-based Retrieval-Augmented Generation)**. O sistema transcende as limitações do RAG tradicional ("Naive RAG") e introduz duas inovações críticas para o nível **Qualis A1**:

1.  **Extração Exaustiva e Híbrida:** Uma estratégia de *Chunking* que combina segmentação estrutural (por artigos/seções) com janelas deslizantes (*sliding windows*) para garantir **100% de cobertura textual**, recuperando inclusive notas de rodapé e textos "órfãos" frequentemente descartados por parsers comuns.
2.  **Mitigação de Viés de Auto-Treinamento (Triangulated Supervision):** Uma técnica nova de refinamento de embeddings que impede que a CNN aprenda os erros ("alucinações") de classificação da IA, triangulando sinais de Rótulo, Adjacência Temporal e Overlap Léxico.

---

## 🏗️ 2. Arquitetura do Sistema e Fluxo de Dados

O diagrama abaixo ilustra o fluxo rigoroso de tensores, destacando a validação cruzada no treinamento da CNN.

```mermaid
graph TD
    A[PDF Bruto] -->|Extração 100%| B(Chunks Exaustivos)
    B -->|Gemini 2.0: NER & Classificação| C{Enriquecimento Semântico}
    C -->|Input Augmentation| D[High-Fidelity Embeddings]
    
    subgraph "Inovação: Triangulated Supervision"
    D -->|Signal 1: AI Label| E{Validador de Tripleto}
    D -->|Signal 2: Temporal Adjacency| E
    D -->|Signal 3: Lexical Overlap| E
    E -->|Consenso >= 2 Sinais| F[CNN 1D - Triplet Loss]
    end
    
    subgraph "Topologia & Grafo"
    F -->|K-Means++ & Silhouette| G[Clusterização Semântica]
    G -->|Link Prediction| H[Grafo de Conhecimento]
    end
    
    subgraph "RAG Lab (Inferência)"
    User[Query do Usuário] -->|HyDE Generator| I[Documento Hipotético]
    I -->|Vector Retrieval| J[Candidatos Top-K]
    J -->|CRAG Evaluator| K{Juiz de Relevância}
    K -->|Rejeitado| L[Descarte]
    K -->|Aprovado| M[Nós Âncora]
    M -->|Graph Traversal| N[Expansão de Vizinhos]
    N -->|Contexto Estendido| O[Geração Final (Agentic)]
    end
```

---

## 🔬 3. Detalhamento Técnico das Etapas

### 3.1. Ingestão Exaustiva (100% Extraction)
A pipeline rejeita o descarte arbitrário de dados. Implementamos um **Chunking Híbrido**:
1.  **Segmentação Estrutural:** Regex primário para capturar hierarquias legais (Artigos, Incisos).
2.  **Sliding Window Fallback:** Se a estrutura falhar, uma janela deslizante de 1000 tokens (com overlap de 200) varre o resíduo.
3.  **Fusão de Ruído:** Fragmentos menores que 5 caracteres não são descartados, mas fundidos ao chunk anterior, garantindo que pontuações e numerações de página não quebrem a continuidade semântica.

### 3.2. Mitigação de Viés: Triangulated Supervision
Em pipelines tradicionais, se a IA classifica erroneamente um texto A como "Lei" e B como "Lei", a CNN aprende a aproximá-los, reforçando o erro (Viés de Auto-Treinamento).
Nossa solução **HAC (Hybrid Anchor Consistency)** redefine a mineração de positivos na Triplet Loss. Um par $(A, P)$ só é positivo se satisfizer a lógica:

$$ Score(A, P) = \mathbb{I}(Label_A = Label_P) + \mathbb{I}(Adj_A \approx Adj_P) + \mathbb{I}(Jaccard(A, P) > \tau) $$

A atualização de pesos ocorre apenas se:
1.  $Score \ge 2$ (Concordância de múltiplos sinais); OU
2.  $Adj_A - Adj_P = 1$ (Vizinhança imediata, fluxo de tópico forte).

Isso ancora o modelo na **realidade física do documento** (adjacência) e na **realidade léxica** (palavras compartilhadas), ignorando alucinações de classificação isoladas.

### 3.3. Refinamento Neural (Embeddings)
*   **Função de Perda:** $L = \max(d(A,P) - d(A,N) + \alpha, 0)$
*   **Otimizador:** AdamW com Decaimento de Peso para regularização.

---

## 🧠 4. Lab RAG: Técnicas Avançadas

A Etapa 5 ("Lab RAG") executa uma cadeia auditável:

### 4.1. HyDE (Hypothetical Document Embeddings)
O sistema alucina intencionalmente uma "Resposta Ideal" (fake) usando um LLM. O vetor desta resposta hipotética serve como proxy para buscar documentos reais, superando a lacuna semântica entre perguntas curtas e documentos técnicos.

### 4.2. CRAG (Corrective RAG)
Um **LLM Juiz** avalia cada chunk recuperado vetorialmente. Chunks com score de relevância $< 0.5$ são descartados antes da geração, limpando o contexto de ruídos ("Red Herrings").

### 4.3. GraphRAG (Recuperação Topológica)
Utiliza os nós validados pelo CRAG como "sementes" para caminhar no grafo. Recupera vizinhos de 1º grau conectados por arestas semânticas fortes, capturando contextos que não compartilham palavras-chave diretas com a query (ex: inferência indireta).

---

## 📊 5. Métricas de Auditoria (Qualis A1)

O sistema gera um **Relatório Técnico** contendo métricas rigorosas:

| Métrica | Definição Matemática | Interpretação no Contexto RAG |
| :--- | :--- | :--- |
| **Modularidade ($Q$)** | $Q = \frac{1}{2m} \sum_{ij} (A_{ij} - \frac{k_i k_j}{2m}) \delta(c_i, c_j)$ | Mede a separabilidade temática. $Q > 0.4$ indica clusters de conhecimento robustos. |
| **Densidade ($\rho$)** | $\rho = \frac{2|E|}{|V|(|V|-1)}$ | Controla o risco de "Hairball" (excesso de conexões confusas). Ideal: $0.05 < \rho < 0.15$. |
| **Silhouette Score** | $S = \frac{b-a}{\max(a,b)}$ | Valida a coesão dos clusters antes da construção do grafo. |

---

## 🚀 6. Guia de Reprodutibilidade

### Pré-requisitos
*   Node.js v18+
*   Chave de API Google Gemini (`GEMINI_API_KEY`).

### Instalação
```bash
npm install
export API_KEY="sua_chave_gemini"
npm start
```

### Protocolo de Teste (Lab RAG)
1.  **Upload:** Carregue PDFs. O sistema extrairá 100% do texto.
2.  **Grafo:** Treine a CNN. Observe no console/status se a perda diminui, indicando que a Supervisão Triangulada está convergindo.
3.  **Lab RAG:** Faça uma pergunta. Verifique os logs para ver o HyDE gerando a hipótese e o CRAG filtrando o lixo.

---

## 👨‍💻 7. Créditos e Autoria

**Desenvolvimento e Concepção Arquitetural:**
**Prof. Marcelo Claro Laranjeira**

*Este software é uma ferramenta de auditoria técnica destinada à validação de arquiteturas RAG em nível acadêmico e industrial (Qualis A1).*
