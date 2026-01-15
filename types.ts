export enum PipelineStage {
  UPLOAD = 'UPLOAD',
  EMBEDDINGS = 'EMBEDDINGS',
  CLUSTERING = 'CLUSTERING',
  GRAPH = 'GRAPH'
}

export interface DocumentChunk {
  id: string;
  source: string;
  content: string; // O texto completo (Qualis A1 rigor)
  tokens: number;
}

export interface EmbeddingVector {
  id: string;
  vector: number[]; // Simulado, exibido truncado
  contentSummary: string;
  fullContent: string;
}

export interface ClusterPoint {
  id: string;
  x: number;
  y: number;
  clusterId: number;
  label: string;
  fullContent: string;
}

export interface GraphNode {
  id: string;
  label: string;
  group: number;
  fullContent: string;
  centrality: number;
}

export interface GraphLink {
  source: string;
  target: string;
  value: number; // Peso da aresta
  type: 'semantico' | 'co-ocorrencia' | 'hierarquico';
}

export interface GraphData {
  nodes: GraphNode[];
  links: GraphLink[];
}