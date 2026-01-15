
# GraphRAG Pipeline Visualizer: Framework de Auditoria e Recuperação Aumentada por Grafos

![Status](https://img.shields.io/badge/Status-Auditoria_Técnica_Qualis_A1-blue?style=for-the-badge)
![Tech Stack](https://img.shields.io/badge/Stack-React_|_Gemini_1.5_Flash_|_Text--Embedding--004-indigo?style=for-the-badge)
![Robustness](https://img.shields.io/badge/Robustness-Heuristic_Fallback_|_CDN_Worker-success?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

**Autor Responsável:** Prof. Marcelo Claro Laranjeira

---

## 📑 1. Resumo Executivo e Inovação

Este repositório hospeda a implementação de referência de uma pipeline **Multi-Stage GraphRAG (Graph-based Retrieval-Augmented Generation)**. O sistema foi atualizado para atingir o nível de robustez **Qualis A1**, introduzindo mecanismos de tolerância a falhas e modelos de última geração:

1.  **Arquitetura Resiliente (Zero-Fail):** Implementação de um sistema de **Fallback Heurístico (Regex)** que assume o controle quando a IA atinge limites de taxa ou filtros de segurança, garantindo 100% de continuidade na ingestão de documentos legais complexos (Vade Mecum, Diários Oficiais).
2.  **State-of-the-Art Embeddings:** Migração para o modelo **Google `text-embedding-004`**, oferecendo vetores de alta fidelidade (768 dimensões) otimizados para tarefas de recuperação semântica e clusterização.
3.  **Mitigação de Viés (Triangulated Supervision):** Refinamento de embeddings via CNN com Triplet Loss, utilizando sinais híbridos (Rótulo + Adjacência Temporal + Overlap Léxico) para evitar overfitting em alucinações.

---

## 🏗️ 2. Arquitetura do Sistema e Fluxo de Dados

O diagrama abaixo ilustra o fluxo rigoroso, destacando a nova camada de segurança heurística.

```mermaid
graph TD
    A[PDF Bruto] -->|PDF.js (CDN Worker)| B(Chunks Exaustivos)
    B --> C{IA Disponível & Segura?}
    C -->|Sim| D[Gemini 1.5 Flash: NER & Classificação]
    C -->|Não / Erro| E[Fallback Heurístico (Regex Jurídico)]
    D --> F{Enriquecimento Semântico}
    E --> F
    F -->|Input Augmentation| G[Text-Embedding-004]
    
    subgraph "Inovação: Triangulated Supervision"
    G -->|Signal 1: AI Label| H{Validador de Tripleto}
    G -->|Signal 2: Temporal Adjacency| H
    G -->|Signal 3: Lexical Overlap| H
    H -->|Consenso >= 2 Sinais| I[CNN 1D - Triplet Loss]
    end
    
    subgraph "Topologia & Grafo"
    I -->|K-Means++ & Silhouette| J[Clusterização Semântica]
    J -->|Link Prediction| K[Grafo de Conhecimento]
    end
    
    subgraph "RAG Lab (Inferência)"
    User[Query do Usuário] -->|HyDE Generator| L[Documento Hipotético]
    L -->|Vector Retrieval| M[Candidatos Top-K]
    M -->|CRAG Evaluator| N{Juiz de Relevância}
    N -->|Rejeitado| O[Descarte]
    N -->|Aprovado| P[Nós Âncora]
    P -->|Graph Traversal| Q[Expansão de Vizinhos]
    Q -->|Contexto Estendido| R[Geração Final]
    end
```

---

## 🔬 3. Detalhamento Técnico das Etapas

### 3.1. Ingestão Robusta (Robust Ingestion)
*   **Worker Dedicado:** Configuração explícita do `pdf.worker.min.js` via CDNJS para contornar problemas de *fake worker* em ambientes ESM.
*   **Segmentação Híbrida:** 
    *   *Camada 1 (AI):* Gemini 1.5 Flash analisa e classifica chunks (Artigos, Incisos).
    *   *Camada 2 (Regex):* Em caso de falha da API (429/500/Safety), um motor Regex de alta precisão identifica estruturas do Direito Brasileiro (Art., §, Capítulos), garantindo que nenhum chunk seja perdido.

### 3.2. Vetorização de Alta Fidelidade
Utilização do modelo **`text-embedding-004`**. Ao contrário de modelos genéricos, este modelo captura nuances semânticas finas necessárias para distinguir conceitos jurídicos próximos (ex: "Furto" vs "Roubo").

### 3.3. Refinamento Neural (HAC - Hybrid Anchor Consistency)
A CNN aprende a aproximar vetores não apenas por rótulos (que podem estar errados), mas pela **triangulação** com a realidade física do documento (proximidade de parágrafos) e realidade léxica (palavras-chave compartilhadas).

---

## 🧠 4. Lab RAG: Pipeline de Inferência

A Etapa 5 ("Lab RAG") executa uma cadeia auditável completa:

1.  **HyDE (Hypothetical Document Embeddings):** Gera uma resposta alucinada ideal para converter a query do usuário em um vetor compatível com o domínio documental.
2.  **Retrieval Híbrido:** Busca vetorial (Cosseno) + Filtro de Metadados.
3.  **CRAG (Corrective RAG):** Um "LLM Juiz" avalia os chunks recuperados. Scores $< 0.6$ são descartados para evitar contaminação do contexto.
4.  **GraphRAG (Expansão Topológica):** Explora vizinhos de 1º grau no grafo para capturar contextos adjacentes que não possuem palavras-chave diretas com a pergunta.

---

## 📊 5. Métricas de Auditoria (Qualis A1)

O sistema gera um **Relatório Técnico** contendo métricas rigorosas:

| Métrica | Definição Matemática | Interpretação no Contexto RAG |
| :--- | :--- | :--- |
| **Modularidade ($Q$)** | $Q = \frac{1}{2m} \sum_{ij} (A_{ij} - \frac{k_i k_j}{2m}) \delta(c_i, c_j)$ | Mede a separabilidade temática. $Q > 0.4$ indica clusters de conhecimento robustos. |
| **Densidade ($\rho$)** | $\rho = \frac{2|E|}{|V|(|V|-1)}$ | Controla o risco de "Hairball". Mantido entre 0.05 e 0.15 para navegabilidade ideal. |
| **Silhouette Score** | $S = \frac{b-a}{\max(a,b)}$ | Valida a coesão dos clusters gerados pelo embedding `text-embedding-004`. |

---

## 🚀 6. Guia de Reprodutibilidade

### Pré-requisitos
*   Node.js v18+
*   Chave de API Google Gemini (`GEMINI_API_KEY`) com acesso aos modelos `gemini-1.5-flash` e `text-embedding-004`.

### Instalação
```bash
npm install
export API_KEY="sua_chave_gemini"
npm start
```

### Protocolo de Teste (Lab RAG)
1.  **Upload:** Carregue PDFs complexos (ex: leis, contratos).
2.  **Enriquecimento:** Observe o fallback entrar em ação se houver instabilidade na API.
3.  **Vetorização:** Confirme o uso do modelo `gemini-004` nos logs.
4.  **Lab RAG:** Execute perguntas complexas e verifique o *trace* de execução (HyDE -> Retrieval -> CRAG -> Graph).

---

## 👨‍💻 7. Créditos e Autoria

**Desenvolvimento e Concepção Arquitetural:**
**Prof. Marcelo Claro Laranjeira**

*Este software é uma ferramenta de auditoria técnica destinada à validação de arquiteturas RAG em nível acadêmico e industrial (Qualis A1).*
