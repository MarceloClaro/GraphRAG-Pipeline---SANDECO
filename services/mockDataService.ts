import { DocumentChunk, EmbeddingVector, ClusterPoint, GraphData, GraphNode, GraphLink } from '../types';

// Helper to generate random ID
const uuid = () => Math.random().toString(36).substring(2, 9);

// --- 1. Real Chunking Strategy ---
export const processRealPDFsToChunks = (rawDocs: { filename: string, text: string }[]): DocumentChunk[] => {
  const chunks: DocumentChunk[] = [];

  rawDocs.forEach(doc => {
    // Dividir por parágrafos duplos (estrutura comum em extração de PDF limpa)
    // Se não houver parágrafos claros, dividir por sentenças grosseiras (~500 chars)
    let rawChunks = doc.text.split(/\n\n+/);
    
    if (rawChunks.length < 2 && doc.text.length > 500) {
        // Fallback para textos densos sem quebras
        rawChunks = doc.text.match(/.{1,1000}(?:\s|$)/g) || [doc.text];
    }

    rawChunks.forEach((textPart, index) => {
      const cleanContent = textPart.trim();
      if (cleanContent.length > 50) { // Ignorar chunks muito pequenos (cabeçalhos/rodapés soltos)
        chunks.push({
          id: `chk_${doc.filename.substring(0,5)}_${index}_${uuid()}`,
          source: doc.filename,
          content: cleanContent,
          tokens: cleanContent.split(' ').length
        });
      }
    });
  });

  return chunks;
};

// --- 2. Embedding Simulation (com dados reais de entrada) ---
// Como não temos um backend Python aqui, simulamos o vetor, mas mantemos o conteúdo real.
export const generateEmbeddingsFromChunks = (chunks: DocumentChunk[]): EmbeddingVector[] => {
  return chunks.map(chunk => ({
    id: chunk.id,
    vector: Array.from({ length: 5 }, () => Math.random()), // Vetor simulado para UI
    contentSummary: chunk.content.substring(0, 50) + '...',
    fullContent: chunk.content
  }));
};

// --- 3. Clustering (Heurística simples baseada em tamanho ou hash simples para "sentir" real) ---
export const generateClustersFromEmbeddings = (embeddings: EmbeddingVector[]): ClusterPoint[] => {
  return embeddings.map((emb, i) => {
    // Simulando clusterização baseada no tamanho do texto (como proxy de complexidade)
    // Em um sistema real, isso viria do UMAP(Embeddings)
    const len = emb.fullContent.length;
    const clusterId = len < 300 ? 0 : len < 600 ? 1 : 2; 
    
    // Posições com base no cluster para visualização separada
    const baseX = clusterId === 0 ? 10 : clusterId === 1 ? 50 : 90;
    const baseY = clusterId === 0 ? 80 : clusterId === 1 ? 20 : 50;

    return {
      id: emb.id,
      x: baseX + (Math.random() * 20 - 10),
      y: baseY + (Math.random() * 20 - 10),
      clusterId: clusterId,
      label: `Chunk ${i + 1}`,
      fullContent: emb.fullContent
    };
  });
};

// --- 4. Graph Generation (com Jaccard Similarity REAL) ---
// Calcula arestas baseadas em sobreposição de palavras reais
export const generateGraphFromClusters = (clusters: ClusterPoint[]): GraphData => {
  const nodes: GraphNode[] = clusters.map(c => ({
    id: c.id,
    label: c.label,
    group: c.clusterId,
    fullContent: c.fullContent,
    centrality: 0 // Será calculado abaixo
  }));

  const links: GraphLink[] = [];

  // Função auxiliar para calcular similaridade de Jaccard (palavras em comum)
  const getJaccardSimilarity = (str1: string, str2: string) => {
    const set1 = new Set(str1.toLowerCase().split(/\W+/).filter(w => w.length > 3));
    const set2 = new Set(str2.toLowerCase().split(/\W+/).filter(w => w.length > 3));
    
    if (set1.size === 0 || set2.size === 0) return 0;

    const intersection = new Set([...set1].filter(x => set2.has(x)));
    const union = new Set([...set1, ...set2]);
    
    return intersection.size / union.size;
  };

  // Criar arestas baseadas no conteúdo REAL
  for (let i = 0; i < nodes.length; i++) {
    for (let j = i + 1; j < nodes.length; j++) {
      const nodeA = nodes[i];
      const nodeB = nodes[j];
      
      const similarity = getJaccardSimilarity(nodeA.fullContent, nodeB.fullContent);
      
      // Threshold: Se tiverem mais de 5% de palavras únicas em comum ou forem do mesmo grupo artificial
      if (similarity > 0.05) {
        links.push({
          source: nodeA.id,
          target: nodeB.id,
          value: similarity,
          type: similarity > 0.15 ? 'semantico' : 'co-ocorrencia'
        });
      }
    }
  }

  // Calcular centralidade básica (Grau)
  const degreeMap: Record<string, number> = {};
  links.forEach(l => {
    degreeMap[l.source] = (degreeMap[l.source] || 0) + 1;
    degreeMap[l.target] = (degreeMap[l.target] || 0) + 1;
  });

  nodes.forEach(n => {
    n.centrality = (degreeMap[n.id] || 0) / nodes.length; // Normalizado simples
  });

  return { nodes, links };
};