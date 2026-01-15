
# GraphRAG Pipeline Visualizer: Framework de Auditoria e Recuperação Aumentada por Grafos

![Status](https://img.shields.io/badge/Status-Auditoria_Técnica_Qualis_A1-blue?style=for-the-badge)
![Tech Stack](https://img.shields.io/badge/Stack-React_|_Gemini_2.0_|_D3.js_|_TensorFlow-indigo?style=for-the-badge)
![RAG Methods](https://img.shields.io/badge/Methods-HyDE_|_CRAG_|_GraphRAG_|_Agentic-purple?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

**Autor Responsável:** Prof. Marcelo Claro Laranjeira

---

## 📑 1. Resumo Executivo

Este repositório hospeda a implementação de referência de uma pipeline **Multi-Stage GraphRAG (Graph-based Retrieval-Augmented Generation)**. Diferentemente de sistemas RAG tradicionais ("Naive RAG"), que dependem exclusivamente de busca vetorial (*vector search*) em um espaço latente plano, esta arquitetura orquestra um **Grafo de Conhecimento Semântico** enriquecido por **Agentes Cognitivos**.

O sistema integra o estado da arte em LLMs (**Google Gemini 2.0 Flash/Embedding-004**) com técnicas avançadas de **Metric Learning (CNN + Triplet Loss)** e **Teoria Espectral de Grafos**. O objetivo primário é a mitigação rigorosa de alucinações estocásticas através de validação cruzada de recuperação (CRAG) e expansão topológica de contexto (GraphRAG).

---

## 🏗️ 2. Arquitetura do Sistema e Fluxo de Dados

O diagrama abaixo ilustra o fluxo de tensores e informações desde a ingestão do documento bruto até a inferência generativa final.

```mermaid
graph TD
    A[PDF Bruto] -->|Extração & Limpeza| B(Chunks Hierárquicos)
    B -->|Gemini 2.0: NER & Classificação| C{Enriquecimento Semântico}
    C -->|Input Augmentation| D[High-Fidelity Embeddings]
    
    subgraph "Metric Learning (Refinamento)"
    D -->|Triplet Loss| E[CNN 1D - Feature Extraction]
    E -->|Otimização AdamW| F[Espaço Vetorial Ajustado]
    end
    
    subgraph "Topologia & Grafo"
    F -->|K-Means++ & Silhouette| G[Clusterização Semântica]
    G -->|Link Prediction: Jaccard + Overlap| H[Grafo de Conhecimento]
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

## 🔬 3. Detalhamento Técnico das Etapas (Pipeline Stages)

### 3.1. Etapa 1: Ingestão e Enriquecimento Semântico (Upload)
A pipeline rejeita o *Naive Chunking* (corte arbitrário por tokens). Implementamos um **Chunking Hierárquico Orientado a Estrutura**:
1.  **Segmentação Lógica:** Identificação de Artigos, Seções e Parágrafos baseada em Regex estrutural.
2.  **Agente de Limpeza (Data Cleaning Agent):** Um LLM processa cada fragmento para remover ruídos de OCR e normalizar Unicode.
3.  **Extração de Metadados:**
    *   **Classificação Taxonômica:** (ex: "Definição Jurídica", "Procedimento Técnico").
    *   **NER (Named Entity Recognition):** Extração de 3-5 palavras-chave canônicas.

### 3.2. Etapa 2: Vetorização e Refinamento Neural (Embeddings)
Transformação do texto em vetores de 768 dimensões.
*   **Input Augmentation:** O vetor não é gerado apenas do texto cru.
    $$Input = [Tipo] \oplus [Rótulo] \oplus [Keywords] \oplus [Conteúdo]$$
*   **Refinamento CNN (Triplet Loss):** Aplicamos uma **Rede Neural Convolucional 1D** treinada em tempo real (browser-side) para distorcer o espaço vetorial, aproximando conceitos similares e afastando distintos.
    *   **Função de Perda:** $L = \max(d(A,P) - d(A,N) + \alpha, 0)$
    *   **Validação:** Cross-Validation 80/20 (Treino/Validação) para evitar overfitting.

### 3.3. Etapa 3 & 4: Clusterização e Construção do Grafo
A topologia não é aleatória; é determinística baseada em propriedades semânticas.
*   **Clusterização:** K-Means++ validado por **Silhouette Score** ($S > 0.5$ ideal).
*   **Definição de Arestas (Links):** A conexão $W_{u,v}$ entre dois nós é calculada por uma função híbrida:
    $$W_{u,v} = 0.6 \cdot \text{Overlap}(K_u, K_v) + 0.4 \cdot \text{Jaccard}(K_u, K_v)$$
    *   Onde $K$ são os conjuntos de palavras-chave extraídas pela IA. Isso captura tanto a similaridade vocabular quanto a inclusão semântica.

---

## 🧠 4. Lab RAG: Técnicas Avançadas Implementadas

A Etapa 5 ("Lab RAG") não é uma simples consulta. Ela executa uma cadeia de pensamento (*Chain of Thought*) complexa, auditável via logs de engenharia.

### 4.1. HyDE (Hypothetical Document Embeddings)
*   **Conceito:** A query do usuário é frequentemente curta e pobre semanticamente.
*   **Implementação:** O sistema alucina intencionalmente uma "Resposta Ideal" (mas fake) usando um LLM.
*   **Vantagem:** O vetor desta resposta hipotética está muito mais próximo do vetor do documento real do que a pergunta original estaria.

### 4.2. CRAG (Corrective RAG)
*   **Conceito:** Recuperação vetorial traz ruído ("False Positives").
*   **Implementação:** Um **LLM Juiz** avalia cada chunk recuperado.
    *   *Input:* Query + Chunk.
    *   *Output:* Score de Relevância (0.0 a 1.0).
    *   *Ação:* Chunks com score $< 0.5$ são descartados antes de poluírem o contexto do gerador final.

### 4.3. GraphRAG (Recuperação Topológica)
*   **Conceito:** A resposta pode estar no "vizinho" do documento encontrado, não no documento em si.
*   **Implementação:**
    1.  Identificamos os nós "âncora" validados pelo CRAG.
    2.  Navegamos pelas arestas do grafo para recuperar os **vizinhos de 1º grau** (1-hop neighbors).
    3.  Este contexto estendido permite inferências laterais que a busca vetorial ignora.

### 4.4. Agentic RAG & Memória
*   **Conceito:** Manutenção de estado e autonomia.
*   **Implementação:** O sistema mantém um histórico de chat (`ChatHistory`) que é injetado recursivamente no prompt final, permitindo perguntas de seguimento ("E sobre o que falamos antes?").

---

## 📊 5. Métricas de Auditoria (Qualis A1)

O sistema gera um **Relatório Técnico** contendo métricas rigorosas de Ciência de Redes:

| Métrica | Definição Matemática | Interpretação no Contexto RAG |
| :--- | :--- | :--- |
| **Modularidade ($Q$)** | $Q = \frac{1}{2m} \sum_{ij} (A_{ij} - \frac{k_i k_j}{2m}) \delta(c_i, c_j)$ | Mede a separabilidade temática. $Q > 0.4$ indica que os documentos formam clusters de conhecimento distintos e coerentes. |
| **Densidade ($\rho$)** | $\rho = \frac{2|E|}{|V|(|V|-1)}$ | Indica a saturação de informações. Grafos muito densos ("Hairball") causam confusão no LLM; grafos esparsos perdem conexões. |
| **Centralidade de Autovetor** | $Ax = \lambda x$ | Identifica os documentos "Hub" (autoridades) que conectam múltiplos temas. Essencial para resumir domínios. |

---

## 🚀 6. Guia de Reprodutibilidade

### Pré-requisitos
*   Node.js v18+
*   Chave de API Google Gemini (`GEMINI_API_KEY`) com acesso aos modelos `gemini-2.0-flash-exp` e `text-embedding-004`.

### Instalação
```bash
# 1. Instalar dependências
npm install

# 2. Configurar Ambiente (Opcional, a chave pode ser inserida no código se necessário para dev local)
export API_KEY="sua_chave_gemini"

# 3. Executar Pipeline
npm start
```

### Protocolo de Teste (Lab RAG)
1.  **Upload:** Carregue PDFs complexos (ex: Legislação, Papers).
2.  **Enriquecimento:** Execute a IA para gerar metadados.
3.  **Grafo:** Gere embeddings, treine a CNN e construa o grafo.
4.  **Lab RAG:** Acesse a aba "Lab RAG".
    *   Digite uma pergunta complexa que exija síntese.
    *   Observe o log lateral: HyDE -> Retrieval -> CRAG -> Graph Expansion.
    *   Verifique se a resposta final cita os nós expandidos pelo grafo.

---

## ⚠️ 7. Limitações e Considerações Éticas

1.  **Custo de API:** O método Agentic/CRAG multiplica o número de chamadas ao LLM (1 chamada por chunk recuperado para avaliação).
2.  **Latência:** A cadeia HyDE + CRAG adiciona latência significativa (~5-10s) em prol da precisão.
3.  **Viés de Treinamento:** A CNN ajusta os pesos baseada nas *labels* geradas pela própria IA na etapa 1. Erros de classificação inicial podem se propagar.

---

## 👨‍💻 8. Créditos e Autoria

**Desenvolvimento e Concepção Arquitetural:**
**Prof. Marcelo Claro Laranjeira**

*Este software é uma ferramenta de auditoria técnica destinada à validação de arquiteturas RAG em nível acadêmico e industrial (Qualis A1).*
