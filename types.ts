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
  dueDate?: string;
  entityType?: string; // Ex: ARTIGO, CAPITULO, INCISO, TEXTO
  entityLabel?: string; // Ex: Art. 1º, Cap. II, Introdução
  keywords?: string[]; // Entidades identificadas por IA
}

export interface EmbeddingVector {
  id: string;
  vector: number[]; // Simulado, exibido truncado
  contentSummary: string;
  fullContent: string;
  dueDate?: string;
  entityType?: string;
  entityLabel?: string;
  keywords?: string[];
}

export interface ClusterPoint {
  id: string;
  x: number;
  y: number;
  clusterId: number;
  label: string;
  fullContent: string;
  dueDate?: string;
  entityType?: string;
  entityLabel?: string;
  keywords?: string[];
}

export interface GraphNode {
  id: string;
  label: string;
  group: number;
  fullContent: string;
  centrality: number;
  dueDate?: string;
  entityType?: string;
  keywords?: string[];
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