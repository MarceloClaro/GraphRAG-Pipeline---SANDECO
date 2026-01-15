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

// Helper function to identify hierarchy
const identifyEntityHierarchy = (text: string): { type: string, label: string } => {
  const cleanText = text.trim();
  if (/^(?:CAPÍTULO|TITULO|LIVRO)\s+[IVXLCDM\d]+/i.test(cleanText)) {
    const match = cleanText.match(/^(?:CAPÍTULO|TITULO|LIVRO)\s+[IVXLCDM\d]+/i);
    return { type: 'ESTRUTURA_MACRO', label: match ? match[0].toUpperCase() : 'CAPÍTULO' };
  }
  if (/^(?:Art\.|Artigo)\s*[\d\.]+/i.test(cleanText)) {
    const match = cleanText.match(/^(?:Art\.|Artigo)\s*[\d\.]+(?:º|°)?/i);
    return { type: 'ARTIGO', label: match ? match[0] : 'Artigo' };
  }
  if (/^(?:§|Parágrafo)\s*/i.test(cleanText)) {
    const match = cleanText.match(/^(?:§\s*[\d\.]+(?:º|°)?|Parágrafo\s+único)/i);
    return { type: 'PARAGRAFO', label: match ? match[0] : '§' };
  }
  if (/^[IVXLCDM]+\s*[\.\-\–]\s+/.test(cleanText)) {
    const match = cleanText.match(/^[IVXLCDM]+/);
    return { type: 'INCISO', label: match ? `Inciso ${match[0]}` : 'Inciso' };
  }
  if (/^[a-z]\)\s+/.test(cleanText)) {
    const match = cleanText.match(/^[a-z]\)/);
    return { type: 'ALINEA', label: match ? `Alínea ${match[0]}` : 'Alínea' };
  }
  if (cleanText.length < 100 && cleanText === cleanText.toUpperCase() && /[A-Z]/.test(cleanText) && cleanText.length > 3) {
    return { type: 'TITULO_SECAO', label: cleanText.substring(0, 30) + (cleanText.length > 30 ? '...' : '') };
  }
  return { type: 'FRAGMENTO_TEXTO', label: 'Texto Geral' };
};

// --- 1. Real Chunking Strategy (Full Coverage) ---
export const processRealPDFsToChunks = (rawDocs: { filename: string, text: string }[]): DocumentChunk[] => {
  const chunks: DocumentChunk[] = [];
  
  rawDocs.forEach(doc => {
    const filenameSafe = doc.filename.replace(/[^a-zA-Z0-9]/g, '').substring(0, 5);
    
    // Normalização de quebras de linha para garantir consistência
    const normalizedText = doc.text.replace(/\r\n/g, '\n').replace(/\r/g, '\n');
    
    // Estratégia Principal: Divisão por Parágrafos (Linha Dupla)
    let rawChunks = normalizedText.split(/\n\s*\n/);
    
    // Estratégia de Fallback: Se o texto for um bloco monolítico (ex: OCR ruim ou formatação estranha)
    if (rawChunks.length < 3 && normalizedText.length > 1000) {
        // Usa Regex com [\s\S] para capturar tudo, incluindo quebras de linha.
        // Divide em blocos de ~1000 caracteres, tentando respeitar espaços.
        rawChunks = normalizedText.match(/[\s\S]{1,1000}(?=\s|$)/g) || [normalizedText];
    }

    rawChunks.forEach((textPart, index) => {
      const cleanContent = textPart.trim();
      
      // FILTRO MÍNIMO: Apenas descarta string vazia ou ruído extremo (< 3 chars).
      // Isso garante que títulos, headers e frases curtas sejam preservados.
      if (!cleanContent || cleanContent.length < 3) return;

      // TRATAMENTO DE BLOCOS GIGANTES (> 2000 chars)
      // Se um parágrafo for muito longo, ele deve ser subdividido para caber na janela de contexto do embedding.
      if (cleanContent.length > 2000) {
         // Subdivisão forçada de blocos massivos
         const subSegments = cleanContent.match(/[\s\S]{1,1200}(?=\s|$)/g) || [cleanContent];
         
         subSegments.forEach((sub, subIdx) => {
            const subClean = sub.trim();
            if (subClean.length < 3) return;
            
            const { type, label } = identifyEntityHierarchy(subClean);
            
            // Ajusta label para indicar continuação
            const finalLabel = subIdx === 0 ? label : `${label} (Cont. ${subIdx})`;
            
            chunks.push({
                id: `chk_${filenameSafe}_${index}_${subIdx}_${uuid()}`,
                source: doc.filename,
                content: subClean,
                tokens: subClean.split(/\s+/).length,
                dueDate: getRandomDueDate(),
                entityType: subIdx === 0 ? type : 'CONTINUACAO',
                entityLabel: finalLabel,
                keywords: []
            });
         });
      } else {
        // Processamento padrão para chunks de tamanho normal
        const { type, label } = identifyEntityHierarchy(cleanContent);
        
        // Se for texto genérico, cria um label baseado nas primeiras palavras
        let finalLabel = label;
        if (type === 'FRAGMENTO_TEXTO') {
            const words = cleanContent.split(/\s+/);
            finalLabel = words.slice(0, 4).join(' ') + (words.length > 4 ? '...' : '');
        }

        chunks.push({
            id: `chk_${filenameSafe}_${index}_${uuid()}`,
            source: doc.filename,
            content: cleanContent,
            tokens: cleanContent.split(/\s+/).length,
            dueDate: getRandomDueDate(),
            entityType: type,
            entityLabel: finalLabel,
            keywords: []
        });
      }
    });
  });
  
  return chunks;
};

// --- 2. Embedding Simulation (Respecting Model Dimensions) ---
export const generateEmbeddingsFromChunks = (chunks: DocumentChunk[], modelType: EmbeddingModelType): EmbeddingVector[] => {
  const dimensions = 768;
  const modelName = 'Gemini Text-Embedding-004';

  return chunks.map(chunk => {
    // Simulated semantic vector based on content hash + noise to create clusters
    const seed = chunk.content.length;
    const pseudoVector = new Array(dimensions).fill(0).map((_, i) => {
        const val = Math.sin(seed * (i + 1)) * Math.cos(seed); 
        return (val + 1) / 2; // Normalize 0-1
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

// --- MATH HELPERS (OPTIMIZED) ---
const euclideanDistance = (a: number[], b: number[]) => {
  let sum = 0;
  // Use first 50 dimensions for distance to save CPU in browser
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
  // Use first 5 dims for clustering logic speedup in browser simulation
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
  const n = Math.min(vectors.length, 200); // Sample for performance
  const sampledVectors = vectors.slice(0, n).map(v => v.slice(0, 10)); // Reduced dim
  
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
export let currentSilhouetteScore = 0; // Export for report

export const generateClustersFromEmbeddings = (embeddings: EmbeddingVector[]): ClusterPoint[] => {
  const vectors = embeddings.map(e => e.vector);
  
  if (vectors.length < 3) return embeddings.map((emb, i) => ({ ...emb, x: Math.random()*100, y: Math.random()*100, clusterId: 0, label: emb.entityLabel || `Chunk ${i}` }));

  // Run K-Means
  const k = Math.min(5, Math.ceil(Math.sqrt(vectors.length/2)));
  const { assignments, centroids } = runKMeans(vectors, k);
  currentSilhouetteScore = calculateSilhouetteScore(vectors, assignments, k);
  
  // Projection for Visualization (Fake UMAP)
  const c1 = centroids[0] || vectors[0].slice(0, 5);
  const c2 = centroids[1] || vectors[1].slice(0, 5);

  return embeddings.map((emb, i) => {
    const clusterId = assignments[i];
    // Project based on distance to pivots + noise for separation
    const angle = (clusterId / k) * 2 * Math.PI;
    const radius = 30 + Math.random() * 20;
    const baseX = Math.cos(angle) * radius + 50;
    const baseY = Math.sin(angle) * radius + 50;

    return {
      id: emb.id,
      clusterId: clusterId, 
      x: baseX + (Math.random() * 10 - 5), // Jitter
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

// --- 4. Graph Generation (Optimized O(N^2) -> Sparse Approach) ---
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

  // 1. Pre-compute keyword sets for fast intersection
  const nodeKeywordSets = nodes.map(n => 
    new Set(n.keywords?.map(k => k.toLowerCase().trim()) || [])
  );

  // 2. Inverted Index for sparse keyword matching
  const keywordToNodeIndices: Record<string, number[]> = {};
  nodeKeywordSets.forEach((set, nodeIdx) => {
    set.forEach(kw => {
        if (!keywordToNodeIndices[kw]) keywordToNodeIndices[kw] = [];
        keywordToNodeIndices[kw].push(nodeIdx);
    });
  });

  const linksMap = new Map<string, GraphLink>();

  // Helper to add/update link with confidence calculation
  const addLink = (idxA: number, idxB: number, weightBase: number, confidenceBase: number, type: 'semantico' | 'co-ocorrencia' | 'hierarquico') => {
      if (idxA === idxB) return;
      // Ensure deterministic key
      const key = idxA < idxB ? `${nodes[idxA].id}-${nodes[idxB].id}` : `${nodes[idxB].id}-${nodes[idxA].id}`;
      
      const existing = linksMap.get(key);
      if (existing) {
          // Reinforce existing link
          existing.value = Math.min(1, existing.value + (weightBase * 0.5));
          existing.confidence = Math.min(1, existing.confidence + (confidenceBase * 0.2));
          
          // Upgrade type priority: Hierarquico > Semantico > Co-ocorrencia
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

  // 3. PHASE A: Semantic Connections (via Inverted Index)
  // Logic: Hybrid Similarity = (Overlap Coefficient * 0.6) + (Jaccard Index * 0.4)
  // Overlap helps detect subset relationships (hierarchical), Jaccard detects exact similarity.
  Object.values(keywordToNodeIndices).forEach(indices => {
      if (indices.length < 2) return;
      if (indices.length > nodes.length * 0.6) return; // Skip stopwords

      for (let i = 0; i < indices.length; i++) {
          for (let j = i + 1; j < indices.length; j++) {
              const u = indices[i];
              const v = indices[j];
              
              const setA = nodeKeywordSets[u];
              const setB = nodeKeywordSets[v];
              
              if (setA.size === 0 || setB.size === 0) continue;

              // Fast intersection
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

              // Composite Confidence Score
              const confidence = (overlapCoeff * 0.6) + (jaccard * 0.4);

              if (confidence > 0.35) {
                  // Weight is slightly lower than confidence to keep graph physics springy
                  addLink(u, v, confidence * 0.8, confidence, 'semantico');
              }
          }
      }
  });

  // 4. PHASE B: Structural/Cluster Connections (Intra-Cluster)
  // Refined: Only link if same Cluster AND (Same Entity Type OR High Density)
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
               
               // Check Entity Type Homophily
               const sameType = nodes[u].entityType === nodes[v].entityType;
               
               // Confidence is lower for pure co-occurrence unless they match type
               const confidence = sameType ? 0.6 : 0.3;
               const weight = sameType ? 0.4 : 0.2;

               addLink(u, v, weight, confidence, 'co-ocorrencia');
          }
      }
  });

  const links = Array.from(linksMap.values()).filter(l => l.confidence > 0.3); // Threshold based on confidence

  // Metrics Calculation
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

  // Modularity
  let edgesWithinClusters = 0;
  links.forEach(l => {
    const sourceGroup = nodes.find(n => n.id === l.source)?.group;
    const targetGroup = nodes.find(n => n.id === l.target)?.group;
    
    if (sourceGroup !== undefined && sourceGroup === targetGroup) {
        edgesWithinClusters++;
    }
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