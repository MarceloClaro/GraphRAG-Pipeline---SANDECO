
import { GoogleGenAI, Type } from "@google/genai";
import { DocumentChunk, EmbeddingVector } from "../types";

// Inicializa o cliente Gemini
// Assume que process.env.API_KEY está disponível
const ai = new GoogleGenAI({ apiKey: process.env.API_KEY });

// Utiliza gemini-1.5-flash para maior estabilidade JSON e prevenção de loops
const modelName = 'gemini-1.5-flash'; 
const embeddingModelName = 'text-embedding-004';

interface GeminiChunkResponse {
  cleaned_text?: string;
  entity_type: string;
  entity_label: string;
  keywords: string[];
}

// Helper para delay (promessa de espera)
const delay = (ms: number) => new Promise(resolve => setTimeout(resolve, ms));

// Wrapper de Retry com Backoff Exponencial
async function retryOperation<T>(operation: () => Promise<T>, maxRetries: number = 4, initialDelay: number = 2000): Promise<T> {
  let lastError: any;
  
  for (let i = 0; i < maxRetries; i++) {
    try {
      return await operation();
    } catch (error: any) {
      lastError = error;
      
      const isRateLimit = error.message?.includes('429') || error.status === 429 || error.code === 429 || error.message?.includes('Quota exceeded');
      const isServerOverload = error.message?.includes('503') || error.status === 503;
      const isInternal = error.message?.includes('500') || error.status === 500;
      
      if (isRateLimit || isServerOverload || isInternal) {
        const waitTime = initialDelay * Math.pow(2, i) + (Math.random() * 1000);
        console.warn(`[Gemini API] Erro Temporário (${i + 1}/${maxRetries}). Aguardando ${Math.round(waitTime)}ms...`);
        await delay(waitTime);
        continue;
      }
      throw error; 
    }
  }
  throw lastError;
}

// --- HYDE: HYPOTHETICAL DOCUMENT EMBEDDING ---
export const generateHyDEAnswer = async (query: string): Promise<string> => {
    try {
        const response = await retryOperation(async () => {
            return await ai.models.generateContent({
                model: modelName,
                contents: `
Atue como um jurista sênior.
Query: "${query}"
Tarefa: Escreva um parágrafo ideal que responderia a esta pergunta, usando linguagem técnica, citando leis hipotéticas ou doutrina.
Objetivo: Usar este texto para busca semântica.
`,
            });
        });
        return response.text || "";
    } catch (e) {
        console.error("Erro HyDE:", e);
        return query; 
    }
};

// --- CRAG: CORRECTIVE RAG EVALUATOR ---
export const evaluateChunkRelevance = async (query: string, chunkContent: string): Promise<{relevant: boolean, score: number, reasoning: string}> => {
    try {
        const response = await retryOperation(async () => {
            return await ai.models.generateContent({
                model: modelName,
                contents: `
Role: RAG Relevance Judge.
Query: "${query}"
Context: "${chunkContent.substring(0, 500)}..."
Task: Is the Context relevant to answer the Query?
Output JSON: { "score": 0.0-1.0, "relevant": bool, "reasoning": "string" }
`,
                config: { responseMimeType: "application/json" }
            });
        });
        const result = JSON.parse(response.text || "{}");
        return {
            relevant: result.relevant === true || (result.score || 0) > 0.6,
            score: result.score ?? 0,
            reasoning: result.reasoning ?? "N/A"
        };
    } catch (e) {
        return { relevant: true, score: 0.5, reasoning: "Evaluation Failed" };
    }
};

// --- SINGLE EMBEDDING ---
export const generateSingleEmbedding = async (text: string): Promise<number[]> => {
    try {
        const result = await retryOperation(async () => {
            return await ai.models.embedContent({
                model: embeddingModelName,
                contents: text.substring(0, 2048), // Truncate for embedding model safety
            });
        });
        return result.embedding?.values || [];
    } catch (e) {
        return new Array(768).fill(0);
    }
};

// --- RAG GENERATION ---
export const generateRAGResponse = async (query: string, context: string, chatHistory: any[]): Promise<string> => {
    try {
        const historyText = chatHistory.slice(-4).map(m => `${m.role}: ${m.content}`).join('\n');
        const response = await retryOperation(async () => {
            return await ai.models.generateContent({
                model: modelName,
                contents: `
CONTEXTO:
${context}

HISTÓRICO:
${historyText}

PERGUNTA: ${query}

RESPOSTA (Baseada ESTRITAMENTE no contexto acima):
`
            });
        });
        return response.text || "Não foi possível gerar a resposta.";
    } catch (e) {
        return "Erro de geração.";
    }
};

/**
 * Processa um único chunk.
 * ESTRATÉGIA DE ROBUSTEZ:
 * 1. Tenta limpeza completa + metadados.
 * 2. Se falhar (ex: loop infinito), tenta APENAS metadados (mantendo texto original).
 */
export const analyzeChunkWithGemini = async (chunk: DocumentChunk): Promise<DocumentChunk> => {
  // Input Safety
  if (!chunk.content || chunk.content.length < 5) return chunk;
  const safeContent = chunk.content.slice(0, 1000); // Limit input context

  // Attempt 1: Full Processing
  try {
    const prompt = `
      INPUT: "${safeContent}"
      TASK:
      1. Clean format (remove newlines/hyphens).
      2. Classify (Artigo, Inciso, Texto).
      3. Label (ex: "Art. 5").
      4. Extract 3 keywords.
      
      OUTPUT JSON STRICTLY:
      { "cleaned_text": "string", "entity_type": "string", "entity_label": "string", "keywords": ["k1", "k2"] }
    `;

    const response = await retryOperation(async () => {
      return await ai.models.generateContent({
        model: modelName,
        contents: prompt,
        config: {
          temperature: 0.1,
          maxOutputTokens: 1024,
          responseMimeType: "application/json"
        }
      });
    }, 2, 1000);

    const text = response.text || "{}";
    
    // Safety check for hallucination loops
    if (text.length > 5000) throw new Error("Loop detected");

    const result = JSON.parse(text) as GeminiChunkResponse;

    return {
      ...chunk,
      content: result.cleaned_text || chunk.content,
      entityType: result.entity_type || "Texto",
      entityLabel: result.entity_label || "Auto",
      keywords: result.keywords || []
    };

  } catch (fullError) {
    // Attempt 2: Metadata Only Fallback
    try {
        // console.warn(`[Gemini Fallback] Chunk ${chunk.id.slice(-4)} cleaning failed. Fetching metadata only.`);
        const fallbackPrompt = `
          Text: "${safeContent}"
          Task: Extract metadata only.
          JSON: { "entity_type": "string", "entity_label": "string", "keywords": ["string"] }
        `;
        
        const fallbackResponse = await retryOperation(async () => {
            return await ai.models.generateContent({
                model: modelName,
                contents: fallbackPrompt,
                config: { temperature: 0.1, responseMimeType: "application/json" }
            });
        }, 1, 1000);
        
        const meta = JSON.parse(fallbackResponse.text || "{}");
        return {
            ...chunk,
            // Keep original content since cleaning failed
            entityType: meta.entity_type || "Texto",
            entityLabel: meta.entity_label || "Auto",
            keywords: meta.keywords || []
        };
    } catch (finalError) {
        console.error(`[Gemini] Failed to enrich chunk ${chunk.id.slice(0,8)}. Keeping original.`);
        return chunk;
    }
  }
};

export const enhanceChunksWithAI = async (chunks: DocumentChunk[], onProgress: (progress: number) => void): Promise<DocumentChunk[]> => {
  const enhancedChunks: DocumentChunk[] = [];
  const batchSize = 3; 
  
  for (let i = 0; i < chunks.length; i += batchSize) {
    const batch = chunks.slice(i, i + batchSize);
    
    const results = await Promise.all(batch.map(async (c, idx) => {
        await delay(idx * 200); // Stagger requests
        return analyzeChunkWithGemini(c);
    }));
    
    enhancedChunks.push(...results);
    const progress = Math.round(((i + batch.length) / chunks.length) * 100);
    onProgress(progress);
  }
  return enhancedChunks;
};

export const generateRealEmbeddingsWithGemini = async (chunks: DocumentChunk[], onProgress: (progress: number) => void): Promise<EmbeddingVector[]> => {
  const embeddings: EmbeddingVector[] = [];
  const batchSize = 5; // Increased batch size for embeddings
  
  for (let i = 0; i < chunks.length; i += batchSize) {
    const batch = chunks.slice(i, i + batchSize);
    
    const batchResults = await Promise.all(batch.map(async (chunk, idx) => {
        await delay(idx * 100); 

        try {
            // Contexto rico para o embedding
            const richContent = `Type: ${chunk.entityType}\nLabel: ${chunk.entityLabel}\nContent: ${chunk.content}`;
            
            const result = await retryOperation(async () => {
                return await ai.models.embedContent({
                    model: embeddingModelName,
                    contents: richContent, 
                });
            }, 3, 1000);
            
            const vector = result.embedding?.values || [];
            
            return {
                id: chunk.id,
                vector: vector,
                contentSummary: chunk.content.substring(0, 50) + '...',
                fullContent: chunk.content,
                dueDate: chunk.dueDate,
                entityType: chunk.entityType,
                entityLabel: chunk.entityLabel,
                keywords: chunk.keywords,
                modelUsed: `Gemini ${embeddingModelName}`
            };
        } catch (e: any) {
            // Fallback Vector (Zeroed) to preserve data integrity instead of dropping
            return {
                id: chunk.id,
                vector: new Array(768).fill(0), 
                contentSummary: chunk.content.substring(0, 50) + '...',
                fullContent: chunk.content,
                dueDate: chunk.dueDate,
                entityType: chunk.entityType,
                entityLabel: chunk.entityLabel,
                keywords: chunk.keywords,
                modelUsed: 'ERROR'
            };
        }
    }));
    embeddings.push(...batchResults);
    const progress = Math.round(((i + batch.length) / chunks.length) * 100);
    onProgress(progress);
  }
  return embeddings;
};
