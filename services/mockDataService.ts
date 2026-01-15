
import { DocumentChunk, EmbeddingVector, ClusterPoint, GraphData, GraphNode, GraphLink, GraphMetrics, EmbeddingModelType } from '../types';

// Helper to generate random ID
const uuid = () => Math.random().toString(36).substring(2, 9);

// Helper to generate random due date (next 1-14 days)
const getRandomDueDate = () => {
  const today = new Date();
  const futureDate = new Date(today);
  futureDate.setDate(today.getDate() + Math.floor(Math.random() * 14) + 1);
  return futureDate.toLocaleDateString('pt-BR');
};

// Helper function to identify hierarchy within a window
const identifyEntityHierarchy = (text: string): { type: string, label: string } => {
  const cleanText = text.trim();
  
  // Regexes de alta precisão para capturar o início do bloco
  const header = cleanText.substring(0, 150);

  if (/(?:CAPÍTULO|TITULO|LIVRO)\s+[IVXLCDM\d]+/i.test(header)) {
    const match = header.match(/(?:CAPÍTULO|TITULO|LIVRO)\s+[IVXLCDM\d]+/i);
    return { type: 'ESTRUTURA_MACRO', label: match ? match[0].toUpperCase() : 'CAPÍTULO' };
  }
  if (/(?:Art\.|Artigo)\s*[\d\.]+/i.test(header)) {
    const match = header.match(/(?:Art\.|Artigo)\s*[\d\.]+(?:º|°)?/i);
    return { type: 'ARTIGO', label: match ? match[0] : 'Artigo' };
  }
  if (/(?:§|Parágrafo)\s*/i.test(header)) {
    const match = header.match(/(?:§\s*[\d\.]+(?:º|°)?|Parágrafo\s+único)/i);
    return { type: 'PARAGRAFO', label: match ? match[0] : '§' };
  }
  if (/^[IVXLCDM]+\s*[\.\-\–]/.test(header)) {
    const match = header.match(/^[IVXLCDM]+/);
    return { type: 'INCISO', label: match ? `Inciso ${match[0]}` : 'Inciso' };
  }
  
  // Inferência genérica de tópico
  const firstWords = cleanText.split(/\s+/).slice(0, 5).join(' ');
  if (firstWords === firstWords.toUpperCase() && firstWords.length > 10) {
      return { type: 'TITULO_SECAO', label: firstWords.substring(0, 30) };
  }

  return { type: 'FRAGMENTO_TEXTO', label: 'Texto Geral' };
};

// --- 1. Real Chunking Strategy (Deterministic High-Granularity Sliding Window) ---
// Ajustado para máxima recuperação (High Recall) -> Mais chunks, menores, mais overlap.
export const processRealPDFsToChunks = (rawDocs: { filename: string, text: string }[]): DocumentChunk[] => {
  const chunks: DocumentChunk[] = [];
  
  // CONFIGURAÇÃO DE ALTA GRANULARIDADE (GraphRAG Needs)
  // 400 chars ~= 60-80 palavras. Ideal para nós de grafo densos.
  const WINDOW_SIZE = 400;  
  const OVERLAP = 100;      
  // Mínimo reduzido para capturar títulos soltos ou frases curtas importantes
  const MIN_CHUNK_SIZE = 15; 

  rawDocs.forEach(doc => {
    const filenameSafe = doc.filename.replace(/[^a-zA-Z0-9]/g, '').substring(0, 5);
    
    // Normalização preservando fluxo
    const fullText = doc.text.replace(/\s+/g, ' ').trim();
    
    if (fullText.length === 0) return;

    let cursor = 0;
    let chunkIndex = 0;

    console.log(`[Chunker] Iniciando processamento de ${doc.filename} (${fullText.length} caracteres)`);

    // Loop "Hard Sliding Window"
    // Garante iterar até o último caractere
    while (cursor < fullText.length) {
        const end = Math.min(cursor + WINDOW_SIZE, fullText.length);
        const content = fullText.slice(cursor, end).trim();

        // Lógica de Preservação Total:
        // Se o conteúdo existe, ele é um chunk. Não descartamos nada > MIN_CHUNK_SIZE.
        // Se for < MIN_CHUNK_SIZE mas for o FINAL do arquivo, mantemos para não perder a conclusão.
        const isLastChunk = end === fullText.length;
        
        if (content.length >= MIN_CHUNK_SIZE || (content.length > 0 && isLastChunk)) {
            
            // Identificação de Metadados
            const { type, label } = identifyEntityHierarchy(content);
            
            // Label refinement
            let finalLabel = label;
            if (type === 'FRAGMENTO_TEXTO') {
                const preview = content.substring(0, 30).replace(/[^\w\s]/gi, '');
                finalLabel = `${preview}...`;
            }

            chunks.push({
                id: `chk_${filenameSafe}_${chunkIndex}_${uuid()}`,
                source: doc.filename,
                content: content,
                tokens: content.split(/\s+/).length,
                dueDate: getRandomDueDate(),
                entityType: type,
                entityLabel: finalLabel,
                keywords: [] 
            });

            chunkIndex++;
        }

        if (end === fullText.length) break;

        // Avanço Determinístico
        cursor += (WINDOW_SIZE - OVERLAP);
    }
  });
  
  console.log(`[Chunker] Extração Finalizada: ${chunks.length} chunks gerados.`);
  return chunks;
};

// --- 2. Embedding Simulation ---
export const generateEmbeddingsFromChunks = (chunks: DocumentChunk[], modelType: EmbeddingModelType): EmbeddingVector[] => {
  const dimensions = 768;
  const modelName = 'Gemini Text-Embedding-004';

  return chunks.map(chunk => {
    const seed = chunk.content.length;
    const pseudoVector = new Array(dimensions).fill(0).map((_, i) => {
        const val = Math.sin(seed * (i + 1)) * Math.cos(seed); 
        return (val + 1) / 2;
    });

    return {
      id: chunk.id,
      vector: pseudoVector,
      contentSummary: chunk.content.substring(0, 50) + '...',
      fullContent: chunk.content,
      dueDate: chunk.dueDate,
      entityType: chunk.entityType,
      entityLabel: chunk.entityLabel,
      keywords: chunk.keywords,
      modelUsed: modelName
    };
  });
};

// --- MATH HELPERS ---
const euclideanDistance = (a: number[], b: number[]) => {
  let sum = 0;
  const len = a.length > 50 ? 50 : a.length;
  for (let i = 0; i < len; i++) {
    const diff = a[i] - b[i];
    sum += diff * diff;
  }
  return Math.sqrt(sum);
};

const addVectors = (a: number[], b: number[]) => {
  const len = a.length;
  const res = new Array(len);
  for(let i = 0; i < len; i++) res[i] = a[i] + b[i];
  return res;
};

const divideVector = (a: number[], scalar: number) => {
  const len = a.length;
  const res = new Array(len);
  for(let i = 0; i < len; i++) res[i] = a[i] / scalar;
  return res;
};

// --- K-MEANS & METRICS ---
interface KMeansResult { centroids: number[][]; assignments: number[]; inertia: number; }

const runKMeans = (vectors: number[][], k: number, maxIterations = 20): KMeansResult => {
  if (vectors.length === 0) return { centroids: [], assignments: [], inertia: 0 };
  if (k > vectors.length) k = vectors.length;
  const reducedVectors = vectors.map(v => v.slice(0, 5));
  
  let centroids = reducedVectors.slice(0, k); 
  let assignments = new Array(vectors.length).fill(0);
  let prevAssignments = new Array(vectors.length).fill(-1);

  for (let iter = 0; iter < maxIterations; iter++) {
    let changed = false;
    assignments = reducedVectors.map((vec, idx) => {
      let minDist = Infinity;
      let clusterIdx = 0;
      centroids.forEach((centroid, cIdx) => {
        const dist = euclideanDistance(vec, centroid);
        if (dist < minDist) { minDist = dist; clusterIdx = cIdx; }
      });
      if (clusterIdx !== prevAssignments[idx]) changed = true;
      return clusterIdx;
    });

    if (!changed) break;
    prevAssignments = [...assignments];

    const sums = Array(k).fill(null).map(() => Array(reducedVectors[0].length).fill(0));
    const counts = Array(k).fill(0);

    reducedVectors.forEach((vec, idx) => {
      const clusterIdx = assignments[idx];
      sums[clusterIdx] = addVectors(sums[clusterIdx], vec);
      counts[clusterIdx]++;
    });

    centroids = sums.map((sum, idx) => {
      if (counts[idx] === 0) return centroids[idx];
      return divideVector(sum, counts[idx]);
    });
  }
  return { centroids, assignments, inertia: 0 };
};

export const calculateSilhouetteScore = (vectors: number[][], assignments: number[], k: number): number => {
  if (k < 2) return 0;
  const n = Math.min(vectors.length, 200); 
  const sampledVectors = vectors.slice(0, n).map(v => v.slice(0, 10)); 
  
  let totalScore = 0;
  for (let i = 0; i < n; i++) {
    const ownCluster = assignments[i];
    let a_i = 0; let ownCount = 0;
    let b_i = Infinity;
    const clusterDistances: Record<number, {sum: number, count: number}> = {};

    for (let j = 0; j < n; j++) {
      if (i === j) continue;
      const dist = euclideanDistance(sampledVectors[i], sampledVectors[j]);
      const otherCluster = assignments[j];

      if (otherCluster === ownCluster) {
        a_i += dist; ownCount++;
      } else {
        if (!clusterDistances[otherCluster]) clusterDistances[otherCluster] = { sum: 0, count: 0 };
        clusterDistances[otherCluster].sum += dist; clusterDistances[otherCluster].count++;
      }
    }
    if (ownCount > 0) a_i /= ownCount;
    for (const cKey in clusterDistances) {
      const c = clusterDistances[cKey];
      const meanDist = c.sum / c.count;
      if (meanDist < b_i) b_i = meanDist;
    }
    if (b_i === Infinity) b_i = 0;
    const s_i = Math.max(a_i, b_i) === 0 ? 0 : (b_i - a_i) / Math.max(a_i, b_i);
    totalScore += s_i;
  }
  return totalScore / n;
};

// --- 3. Advanced Clustering Logic ---
export let currentSilhouetteScore = 0; 

export const generateClustersFromEmbeddings = (embeddings: EmbeddingVector[]): ClusterPoint[] => {
  const vectors = embeddings.map(e => e.vector);
  
  if (vectors.length < 3) return embeddings.map((emb, i) => ({ ...emb, x: Math.random()*100, y: Math.random()*100, clusterId: 0, label: emb.entityLabel || `Chunk ${i}` }));

  // Run K-Means
  const k = Math.min(5, Math.ceil(Math.sqrt(vectors.length/2)));
  const { assignments, centroids } = runKMeans(vectors, k);
  currentSilhouetteScore = calculateSilhouetteScore(vectors, assignments, k);
  
  // Projection for Visualization
  return embeddings.map((emb, i) => {
    const clusterId = assignments[i];
    const angle = (clusterId / k) * 2 * Math.PI;
    const radius = 30 + Math.random() * 20;
    const baseX = Math.cos(angle) * radius + 50;
    const baseY = Math.sin(angle) * radius + 50;

    return {
      id: emb.id,
      clusterId: clusterId, 
      x: baseX + (Math.random() * 10 - 5),
      y: baseY + (Math.random() * 10 - 5),
      label: emb.entityLabel || `Chunk ${i}`,
      fullContent: emb.fullContent,
      dueDate: emb.dueDate,
      entityType: emb.entityType,
      entityLabel: emb.entityLabel,
      keywords: emb.keywords
    };
  });
};

// --- 4. Graph Generation ---
export const generateGraphFromClusters = (clusters: ClusterPoint[]): GraphData => {
  const nodes: GraphNode[] = clusters.map(c => ({
    id: c.id,
    label: c.entityLabel || c.label,
    group: c.clusterId === -1 ? 99 : c.clusterId,
    fullContent: c.fullContent,
    centrality: 0, 
    dueDate: c.dueDate,
    entityType: c.entityType,
    keywords: c.keywords
  }));

  const nodeKeywordSets = nodes.map(n => 
    new Set(n.keywords?.map(k => k.toLowerCase().trim()) || [])
  );

  const keywordToNodeIndices: Record<string, number[]> = {};
  nodeKeywordSets.forEach((set, nodeIdx) => {
    set.forEach(kw => {
        if (!keywordToNodeIndices[kw]) keywordToNodeIndices[kw] = [];
        keywordToNodeIndices[kw].push(nodeIdx);
    });
  });

  const linksMap = new Map<string, GraphLink>();

  const addLink = (idxA: number, idxB: number, weightBase: number, confidenceBase: number, type: 'semantico' | 'co-ocorrencia' | 'hierarquico') => {
      if (idxA === idxB) return;
      const key = idxA < idxB ? `${nodes[idxA].id}-${nodes[idxB].id}` : `${nodes[idxB].id}-${nodes[idxA].id}`;
      
      const existing = linksMap.get(key);
      if (existing) {
          existing.value = Math.min(1, existing.value + (weightBase * 0.5));
          existing.confidence = Math.min(1, existing.confidence + (confidenceBase * 0.2));
          if (type === 'hierarquico') existing.type = 'hierarquico';
          else if (type === 'semantico' && existing.type === 'co-ocorrencia') existing.type = 'semantico';
      } else {
          linksMap.set(key, {
              source: nodes[idxA].id,
              target: nodes[idxB].id,
              value: weightBase,
              confidence: confidenceBase,
              type: type
          });
      }
  };

  // PHASE A: Semantic (Keyword Overlap + Jaccard)
  Object.values(keywordToNodeIndices).forEach(indices => {
      if (indices.length < 2) return;
      if (indices.length > nodes.length * 0.6) return;

      for (let i = 0; i < indices.length; i++) {
          for (let j = i + 1; j < indices.length; j++) {
              const u = indices[i];
              const v = indices[j];
              
              const setA = nodeKeywordSets[u];
              const setB = nodeKeywordSets[v];
              
              if (setA.size === 0 || setB.size === 0) continue;

              let intersection = 0;
              const smallerSet = setA.size < setB.size ? setA : setB;
              const largerSet = setA.size < setB.size ? setB : setA;
              
              for (const k of smallerSet) {
                  if (largerSet.has(k)) intersection++;
              }
              
              if (intersection === 0) continue;

              const union = setA.size + setB.size - intersection;
              const minSize = Math.min(setA.size, setB.size);

              const jaccard = intersection / union;
              const overlapCoeff = intersection / minSize;
              const confidence = (overlapCoeff * 0.6) + (jaccard * 0.4);

              if (confidence > 0.35) {
                  addLink(u, v, confidence * 0.8, confidence, 'semantico');
              }
          }
      }
  });

  // PHASE B: Structural (Cluster Homophily)
  const nodesByCluster: Record<number, number[]> = {};
  nodes.forEach((n, idx) => {
      if (!nodesByCluster[n.group]) nodesByCluster[n.group] = [];
      nodesByCluster[n.group].push(idx);
  });

  Object.values(nodesByCluster).forEach(indices => {
      if (indices.length < 2) return;
      for (let i = 0; i < indices.length; i++) {
          for (let j = i + 1; j < indices.length; j++) {
               const u = indices[i];
               const v = indices[j];
               const sameType = nodes[u].entityType === nodes[v].entityType;
               const confidence = sameType ? 0.6 : 0.3;
               const weight = sameType ? 0.4 : 0.2;
               addLink(u, v, weight, confidence, 'co-ocorrencia');
          }
      }
  });

  const links = Array.from(linksMap.values()).filter(l => l.confidence > 0.3);
  const edgeCount = links.length;
  const n = nodes.length;
  const density = n > 1 ? (2 * edgeCount) / (n * (n - 1)) : 0;
  
  const degreeMap: Record<string, number> = {};
  links.forEach(l => {
    degreeMap[l.source] = (degreeMap[l.source] || 0) + 1;
    degreeMap[l.target] = (degreeMap[l.target] || 0) + 1;
  });

  let totalDegree = 0;
  nodes.forEach(node => {
    const deg = degreeMap[node.id] || 0;
    node.centrality = deg / (n - 1 || 1);
    totalDegree += deg;
  });

  const avgDegree = n > 0 ? totalDegree / n : 0;
  let edgesWithinClusters = 0;
  links.forEach(l => {
    const sourceGroup = nodes.find(n => n.id === l.source)?.group;
    const targetGroup = nodes.find(n => n.id === l.target)?.group;
    if (sourceGroup !== undefined && sourceGroup === targetGroup) edgesWithinClusters++;
  });
  
  const modularity = edgeCount > 0 ? (edgesWithinClusters / edgeCount) - Math.pow(1 / (nodes.length || 1), 2) : 0;

  const metrics: GraphMetrics = {
      density,
      avgDegree,
      modularity,
      silhouetteScore: currentSilhouetteScore,
      totalNodes: n,
      totalEdges: edgeCount,
      connectedComponents: 1 
  };

  return { nodes, links, metrics };
};
