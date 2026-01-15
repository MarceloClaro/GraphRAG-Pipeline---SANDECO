import { DocumentChunk, EmbeddingVector, ClusterPoint, GraphData, GraphNode, GraphLink } from '../types';

// Helper to generate random ID
const uuid = () => Math.random().toString(36).substring(2, 9);

// Helper to generate random due date (next 1-14 days)
const getRandomDueDate = () => {
  const today = new Date();
  const futureDate = new Date(today);
  futureDate.setDate(today.getDate() + Math.floor(Math.random() * 14) + 1);
  return futureDate.toLocaleDateString('pt-BR');
};

// Helper function to identify hierarchy based on regex patterns (Fallback se IA não for usada)
const identifyEntityHierarchy = (text: string): { type: string, label: string } => {
  const cleanText = text.trim();
  
  // 1. CAPÍTULOS / TÍTULOS (Geralmente UPPERCASE curto ou "Capítulo X")
  if (/^(?:CAPÍTULO|TITULO|LIVRO)\s+[IVXLCDM\d]+/i.test(cleanText)) {
    const match = cleanText.match(/^(?:CAPÍTULO|TITULO|LIVRO)\s+[IVXLCDM\d]+/i);
    return { type: 'ESTRUTURA_MACRO', label: match ? match[0].toUpperCase() : 'CAPÍTULO' };
  }
  
  // 2. ARTIGOS (Ex: "Art. 1º", "Artigo 5")
  if (/^(?:Art\.|Artigo)\s*[\d\.]+/i.test(cleanText)) {
    const match = cleanText.match(/^(?:Art\.|Artigo)\s*[\d\.]+(?:º|°)?/i);
    return { type: 'ARTIGO', label: match ? match[0] : 'Artigo' };
  }

  // 3. PARÁGRAFOS (Ex: "§ 1º", "Parágrafo único")
  if (/^(?:§|Parágrafo)\s*/i.test(cleanText)) {
    const match = cleanText.match(/^(?:§\s*[\d\.]+(?:º|°)?|Parágrafo\s+único)/i);
    return { type: 'PARAGRAFO', label: match ? match[0] : '§' };
  }

  // 4. INCISOS (Números Romanos no início: I -, II -, III)
  // Regex simples para capturar I, II, III, IV, etc. seguido de ponto ou traço
  if (/^[IVXLCDM]+\s*[\.\-\–]\s+/.test(cleanText)) {
    const match = cleanText.match(/^[IVXLCDM]+/);
    return { type: 'INCISO', label: match ? `Inciso ${match[0]}` : 'Inciso' };
  }

  // 5. ALÍNEAS (Letras minúsculas seguidas de parenteses: a), b) )
  if (/^[a-z]\)\s+/.test(cleanText)) {
    const match = cleanText.match(/^[a-z]\)/);
    return { type: 'ALINEA', label: match ? `Alínea ${match[0]}` : 'Alínea' };
  }
  
  // 6. Cabeçalhos de texto (Linhas curtas totalmente em maiúsculas que não caíram na regra 1)
  if (cleanText.length < 60 && cleanText === cleanText.toUpperCase() && /[A-Z]/.test(cleanText)) {
    return { type: 'TITULO_SECAO', label: cleanText.substring(0, 20) };
  }

  // Padrão
  return { type: 'FRAGMENTO_TEXTO', label: 'Texto Geral' };
};

// --- 1. Real Chunking Strategy ---
export const processRealPDFsToChunks = (rawDocs: { filename: string, text: string }[]): DocumentChunk[] => {
  const chunks: DocumentChunk[] = [];

  rawDocs.forEach(doc => {
    // Dividir por parágrafos duplos para separar blocos lógicos
    let rawChunks = doc.text.split(/\n\n+/);
    
    // Fallback para textos densos
    if (rawChunks.length < 2 && doc.text.length > 500) {
        rawChunks = doc.text.match(/.{1,1000}(?:\s|$)/g) || [doc.text];
    }

    rawChunks.forEach((textPart, index) => {
      const cleanContent = textPart.trim();
      if (cleanContent.length > 20) { // Ignorar ruído muito curto
        
        // Identificar Hierarquia (Inicial - pode ser melhorada pela IA depois)
        const { type, label } = identifyEntityHierarchy(cleanContent);
        
        // Se for "Texto Geral", tentamos dar um label mais útil baseado no início do texto
        let finalLabel = label;
        if (type === 'FRAGMENTO_TEXTO') {
            const firstWords = cleanContent.split(' ').slice(0, 3).join(' ');
            finalLabel = `${firstWords}...`;
        }

        chunks.push({
          id: `chk_${doc.filename.substring(0,3)}_${index}_${uuid()}`,
          source: doc.filename,
          content: cleanContent,
          tokens: cleanContent.split(' ').length,
          dueDate: getRandomDueDate(),
          entityType: type,
          entityLabel: finalLabel,
          keywords: [] // Inicialmente vazio, preenchido pelo Gemini
        });
      }
    });
  });

  return chunks;
};

// --- 2. Embedding Simulation (Enhanced for Clustering) ---
// Gera vetores que não são puramente aleatórios, mas influenciados pelo conteúdo
export const generateEmbeddingsFromChunks = (chunks: DocumentChunk[]): EmbeddingVector[] => {
  return chunks.map(chunk => {
    // Hash simples do conteúdo para gerar sementes "semânticas"
    let hash = 0;
    for (let i = 0; i < chunk.content.length; i++) {
      hash = ((hash << 5) - hash) + chunk.content.charCodeAt(i);
      hash |= 0;
    }
    
    // Simular 5 dimensões baseadas em características do texto
    const len = chunk.content.length;
    const words = chunk.content.split(' ').length;
    const hasNumbers = /\d/.test(chunk.content) ? 1 : 0;
    const isQuestion = chunk.content.includes('?') ? 1 : 0;
    
    // Normalização bruta para 0-1
    const v1 = (hash % 100) / 100;
    const v2 = Math.min(len / 1000, 1);
    const v3 = Math.min(words / 200, 1);
    const v4 = hasNumbers;
    const v5 = isQuestion;

    return {
      id: chunk.id,
      vector: [v1, v2, v3, v4, v5],
      contentSummary: chunk.content.substring(0, 50) + '...',
      fullContent: chunk.content,
      dueDate: chunk.dueDate,
      entityType: chunk.entityType,
      entityLabel: chunk.entityLabel,
      keywords: chunk.keywords
    };
  });
};

// --- MATH HELPERS FOR CLUSTERING ---

const euclideanDistance = (a: number[], b: number[]) => {
  let sum = 0;
  for (let i = 0; i < a.length; i++) {
    sum += Math.pow(a[i] - b[i], 2);
  }
  return Math.sqrt(sum);
};

const addVectors = (a: number[], b: number[]) => a.map((val, i) => val + b[i]);
const divideVector = (a: number[], scalar: number) => a.map(val => val / scalar);

// --- K-MEANS IMPLEMENTATION ---

interface KMeansResult {
  centroids: number[][];
  assignments: number[];
  inertia: number;
}

const runKMeans = (vectors: number[][], k: number, maxIterations = 20): KMeansResult => {
  if (vectors.length === 0) return { centroids: [], assignments: [], inertia: 0 };
  if (k > vectors.length) k = vectors.length;

  // 1. Initialize Centroids (Randomly pick k vectors)
  let centroids = vectors.slice(0, k); 
  let assignments = new Array(vectors.length).fill(0);
  let prevAssignments = new Array(vectors.length).fill(-1);

  for (let iter = 0; iter < maxIterations; iter++) {
    // 2. Assign points to nearest centroid
    let changed = false;
    assignments = vectors.map((vec, idx) => {
      let minDist = Infinity;
      let clusterIdx = 0;
      centroids.forEach((centroid, cIdx) => {
        const dist = euclideanDistance(vec, centroid);
        if (dist < minDist) {
          minDist = dist;
          clusterIdx = cIdx;
        }
      });
      if (clusterIdx !== prevAssignments[idx]) changed = true;
      return clusterIdx;
    });

    if (!changed) break;
    prevAssignments = [...assignments];

    // 3. Update Centroids
    const sums = Array(k).fill(null).map(() => Array(vectors[0].length).fill(0));
    const counts = Array(k).fill(0);

    vectors.forEach((vec, idx) => {
      const clusterIdx = assignments[idx];
      sums[clusterIdx] = addVectors(sums[clusterIdx], vec);
      counts[clusterIdx]++;
    });

    centroids = sums.map((sum, idx) => {
      if (counts[idx] === 0) return centroids[idx]; // Keep old if empty
      return divideVector(sum, counts[idx]);
    });
  }

  // Calculate Inertia (WCSS)
  let inertia = 0;
  vectors.forEach((vec, idx) => {
    inertia += Math.pow(euclideanDistance(vec, centroids[assignments[idx]]), 2);
  });

  return { centroids, assignments, inertia };
};

// --- SILHOUETTE SCORE ---

const calculateSilhouetteScore = (vectors: number[][], assignments: number[], k: number): number => {
  if (k < 2) return 0;
  
  let totalScore = 0;
  const n = vectors.length;

  for (let i = 0; i < n; i++) {
    const ownCluster = assignments[i];
    
    // a(i): Mean distance to points in same cluster
    let a_i = 0;
    let ownCount = 0;
    
    // b(i): Min mean distance to points in other clusters
    let b_i = Infinity;
    
    // Pre-calculate distances to other clusters
    const clusterDistances: Record<number, {sum: number, count: number}> = {};

    for (let j = 0; j < n; j++) {
      if (i === j) continue;
      const dist = euclideanDistance(vectors[i], vectors[j]);
      const otherCluster = assignments[j];

      if (otherCluster === ownCluster) {
        a_i += dist;
        ownCount++;
      } else {
        if (!clusterDistances[otherCluster]) clusterDistances[otherCluster] = { sum: 0, count: 0 };
        clusterDistances[otherCluster].sum += dist;
        clusterDistances[otherCluster].count++;
      }
    }

    if (ownCount > 0) a_i /= ownCount;

    for (const cKey in clusterDistances) {
      const c = clusterDistances[cKey];
      const meanDist = c.sum / c.count;
      if (meanDist < b_i) b_i = meanDist;
    }

    if (b_i === Infinity) b_i = 0; // Should not happen if k > 1

    const s_i = Math.max(a_i, b_i) === 0 ? 0 : (b_i - a_i) / Math.max(a_i, b_i);
    totalScore += s_i;
  }

  return totalScore / n;
};

// --- DBSCAN IMPLEMENTATION ---

const runDBSCAN = (vectors: number[][], epsilon: number, minPts: number): number[] => {
  const n = vectors.length;
  const labels = new Array(n).fill(-2); // -2: undefined, -1: noise
  let clusterId = 0;

  const getNeighbors = (idx: number) => {
    const neighbors = [];
    for (let i = 0; i < n; i++) {
      if (i !== idx && euclideanDistance(vectors[idx], vectors[i]) <= epsilon) {
        neighbors.push(i);
      }
    }
    return neighbors;
  };

  for (let i = 0; i < n; i++) {
    if (labels[i] !== -2) continue;

    const neighbors = getNeighbors(i);
    if (neighbors.length < minPts) {
      labels[i] = -1; // Noise
    } else {
      labels[i] = clusterId;
      let seedSet = [...neighbors];
      
      let k = 0;
      while (k < seedSet.length) {
        const currentIdx = seedSet[k];
        if (labels[currentIdx] === -1) labels[currentIdx] = clusterId; // Change noise to border
        if (labels[currentIdx] !== -2) {
            k++;
            continue;
        }

        labels[currentIdx] = clusterId;
        const currentNeighbors = getNeighbors(currentIdx);
        if (currentNeighbors.length >= minPts) {
          seedSet = [...seedSet, ...currentNeighbors];
        }
        k++;
      }
      clusterId++;
    }
  }
  return labels;
};

// --- 3. Advanced Clustering Logic ---
export const generateClustersFromEmbeddings = (embeddings: EmbeddingVector[]): ClusterPoint[] => {
  const vectors = embeddings.map(e => e.vector);
  
  if (vectors.length < 3) {
    return embeddings.map((emb, i) => ({
      id: emb.id,
      x: Math.random() * 100,
      y: Math.random() * 100,
      clusterId: 0,
      label: emb.entityLabel || `Chunk ${i + 1}`,
      fullContent: emb.fullContent,
      dueDate: emb.dueDate,
      entityType: emb.entityType,
      entityLabel: emb.entityLabel,
      keywords: emb.keywords
    }));
  }

  console.log("Iniciando otimização de K-Means...");
  let bestK = 2;
  let bestScore = -1;
  let bestAssignments: number[] = [];
  let bestCentroids: number[][] = [];

  const maxK = Math.min(6, vectors.length);
  
  for (let k = 2; k <= maxK; k++) {
    const { assignments, centroids } = runKMeans(vectors, k);
    const score = calculateSilhouetteScore(vectors, assignments, k);
    
    if (score > bestScore) {
      bestScore = score;
      bestK = k;
      bestAssignments = assignments;
      bestCentroids = centroids;
    }
  }
  
  const dbscanLabels = runDBSCAN(vectors, 0.35, 2); 
  
  return embeddings.map((emb, i) => {
    const kMeansCluster = bestAssignments[i];
    const isNoise = dbscanLabels[i] === -1;
    
    const centroid = bestCentroids[kMeansCluster];
    
    const anchorX = (centroid[0] * 100) + (centroid[2] * 50) + (kMeansCluster * 40);
    const anchorY = (centroid[1] * 100) + (centroid[3] * 50) + ((kMeansCluster % 2) * 40);

    return {
      id: emb.id,
      clusterId: isNoise ? -1 : kMeansCluster, 
      x: anchorX + (Math.random() * 15 - 7.5),
      y: anchorY + (Math.random() * 15 - 7.5),
      label: emb.entityLabel || (isNoise ? `[Ruído] Chunk ${i}` : `Chunk ${i} (G${kMeansCluster})`),
      fullContent: emb.fullContent,
      dueDate: emb.dueDate,
      entityType: emb.entityType,
      entityLabel: emb.entityLabel,
      keywords: emb.keywords
    };
  });
};

// --- 4. Graph Generation (Enhanced with Gemini Keywords) ---
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

  // Helper para interseção de keywords (Melhorado com IA)
  const getKeywordIntersection = (keys1?: string[], keys2?: string[]) => {
      if (!keys1 || !keys2 || keys1.length === 0 || keys2.length === 0) return 0;
      const set1 = new Set(keys1.map(k => k.toLowerCase()));
      const set2 = new Set(keys2.map(k => k.toLowerCase()));
      const intersection = [...set1].filter(x => set2.has(x));
      return intersection.length / Math.max(set1.size, set2.size); // Normalizado
  };

  // Criar arestas
  for (let i = 0; i < nodes.length; i++) {
    for (let j = i + 1; j < nodes.length; j++) {
      const nodeA = nodes[i];
      const nodeB = nodes[j];
      
      // 1. Similaridade Textual (Fallback)
      const textSimilarity = getJaccardSimilarity(nodeA.fullContent, nodeB.fullContent);
      
      // 2. Similaridade de Entidades (IA)
      const entitySimilarity = getKeywordIntersection(nodeA.keywords, nodeB.keywords);

      // Peso final (dá mais valor às entidades encontradas pela IA se existirem)
      const finalWeight = (entitySimilarity * 0.7) + (textSimilarity * 0.3);
      
      // Threshold adaptativo
      if (finalWeight > 0.05) {
        links.push({
          source: nodeA.id,
          target: nodeB.id,
          value: finalWeight,
          type: entitySimilarity > 0.2 ? 'semantico' : 'co-ocorrencia'
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