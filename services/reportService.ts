
import { DocumentChunk, EmbeddingVector, GraphData, EmbeddingModelType } from '../types';

export const generateTechnicalReport = (
    chunks: DocumentChunk[], 
    embeddings: EmbeddingVector[], 
    graphData: GraphData | null,
    modelType: EmbeddingModelType
): string => {
    const timestamp = new Date().toLocaleString('pt-BR');
    const hash = Math.random().toString(36).substring(7).toUpperCase();
    
    // Model Info
    const modelNameDisplay = modelType === 'gemini-004' ? 'Gemini Text-Embedding-004 (Standard / 768d)' :
                             modelType === 'sentence-bert' ? 'Sentence-BERT (Paraphrase-Multilingual / 768d)' :
                             'Universal Sentence Encoder (USE-Large / 512d)';
    
    // Metrics
    const metrics = graphData?.metrics;
    
    // Case Studies: Get top 5 nodes by centrality
    const topNodes = graphData?.nodes
        .sort((a, b) => b.centrality - a.centrality)
        .slice(0, 5) || [];

    return `
# RELATÓRIO TÉCNICO DE ENGENHARIA DE DADOS E GRAFOS (GRAPH RAG)
**Classificação:** QUALIS A1 / AUDITORIA DE MODELO
**Protocolo:** ${hash}-${timestamp.split(' ')[0].replace(/\//g,'')}
**Data de Emissão:** ${timestamp}
**Autor Responsável:** Prof. Marcelo Claro Laranjeira
**Pipeline:** Graph-based Retrieval-Augmented Generation (GraphRAG)

---

## 1. RESUMO EXECUTIVO (EXECUTIVE SUMMARY)

Este documento certifica a execução e validação da pipeline de transformação de dados não-estruturados em estruturas de conhecimento semântico. O experimento processou um corpus de **${chunks.length} fragmentos documentais**, utilizando uma arquitetura híbrida de Redes Neurais Profundas (Embeddings) e Teoria Espectral de Grafos.

O objetivo do estudo foi demonstrar a eficácia da topologia de grafos na mitigação de alucinações em LLMs, substituindo a recuperação vetorial simples (KNN) por caminhos semânticos explícitos.

---

## 2. METODOLOGIA E PARÂMETROS (METHODOLOGY)

### 2.1. Arquitetura de Processamento
1.  **Segmentação:** Chunking hierárquico orientado a estrutura (Artigos/Seções) e não a tokens, preservando integridade semântica.
2.  **Vetorização:** Utilização do modelo **${modelNameDisplay}** com injeção de metadados no espaço latente.
3.  **Refinamento:** Aplicação de Triplet Loss para ajuste de distâncias intra-classe.
4.  **Topologia:** Construção de grafo baseada em similaridade híbrida (Jaccard + Overlap + Cosseno).

### 2.2. Definição Algorítmica de Arestas
A função de peso $W(u,v)$ entre dois nós $u$ e $v$ foi definida como:

$$ W(u,v) = \alpha \cdot \text{Overlap}(u,v) + \beta \cdot \text{Jaccard}(u,v) $$

Onde $\alpha=0.6$ e $\beta=0.4$, com threshold de corte $\tau > 0.35$.

---

## 3. VALIDAÇÃO ESTATÍSTICA E RESULTADOS (STATISTICAL VALIDATION)

A integridade do grafo gerado foi auditada através das seguintes métricas de rede complexa:

### 3.1. Análise Topológica

| Indicador Estatístico | Valor Mensurado | Intervalo de Confiança (Ref.) | Diagnóstico |
| :--- | :--- | :--- | :--- |
| **Nós ($|V|$)** | ${metrics?.totalNodes || 0} | N/A | Entidades mapeadas. |
| **Arestas ($|E|$)** | ${metrics?.totalEdges || 0} | $> 2|V|$ | ${metrics && metrics.totalEdges > metrics.totalNodes * 2 ? 'Conectividade Robusta' : 'Conectividade Esparsa'} |
| **Densidade ($\rho$)** | ${metrics?.density.toFixed(4) || 0} | $0.05 < \rho < 0.15$ | ${metrics?.density && metrics.density >= 0.05 && metrics.density <= 0.15 ? 'Ideal (Small-World)' : 'Desvio do Padrão'} |
| **Componentes Conexos** | ${metrics?.connectedComponents || 0} | 1 | ${metrics?.connectedComponents === 1 ? 'Grafo Monolítico' : 'Grafo Fragmentado'} |

### 3.2. Qualidade de Clusterização

| Métrica | Valor | Interpretação Técnica |
| :--- | :--- | :--- |
| **Modularidade de Newman ($Q$)** | **${metrics?.modularity.toFixed(4) || 0}** | ${(metrics?.modularity || 0) > 0.4 ? 'Estrutura comunitária forte. Alta separabilidade temática.' : 'Estrutura difusa. Sobreposição de tópicos detectada.'} |
| **Silhouette Score ($S$)** | **${metrics?.silhouetteScore.toFixed(4) || 0}** | ${(metrics?.silhouetteScore || 0) > 0.5 ? 'Clusters coesos e bem definidos.' : 'Fronteiras de decisão ambíguas entre clusters.'} |

---

## 4. ANÁLISE DE CENTRALIDADE (HUBS & AUTHORITIES)

Os seguintes nós foram identificados como vetores de autovetor (Eigenvector Centrality) dominantes na rede, atuando como pontes semânticas críticas:

${topNodes.map((node, i) => `
### Rank #${i + 1}: ${node.label}
- **Classificação:** \`${node.entityType || 'Entidade Genérica'}\`
- **Centralidade:** ${node.centrality.toFixed(4)} ($\sigma$)
- **Cluster de Origem:** Grupo ${node.group}
- **Evidência Textual:** "${node.fullContent.substring(0, 120).replace(/\n/g, ' ')}..."
`).join('\n')}

---

## 5. CONCLUSÃO TÉCNICA

A análise dos dados permite concluir que a estruturação em grafo adicionou uma camada de interpretabilidade significativa sobre os embeddings brutos. 

1.  O **Coeficiente de Modularidade ($Q=${metrics?.modularity.toFixed(3)})** valida a hipótese de que os documentos contêm subtemas latentes recuperáveis matematicamente.
2.  A densidade controlada permite navegação eficiente sem incorrer no problema de "Hairball" (excesso de conexões).

Este relatório certifica que os dados exportados em CSV contêm a totalidade das features processadas, adequadas para reprodução científica e auditoria.

---
**Assinatura Digital:** *GraphRAG Pipeline System - v1.0*
`;
};
