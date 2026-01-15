
import { GoogleGenAI } from "@google/genai";
import { DocumentChunk, EmbeddingVector } from "../types";

// Inicializa o cliente Gemini
const ai = new GoogleGenAI({ apiKey: process.env.API_KEY });

// Configuração dos Modelos
const modelName = 'gemini-3-flash-preview'; 
const embeddingModelName = 'text-embedding-004';

// --- CIRCUIT BREAKER STATE ---
// Se true, interrompe chamadas à API para evitar erros em cascata e bloqueios
let globalQuotaExceeded = false;

interface GeminiChunkResponse {
  cleaned_text?: string;
  entity_type: string;
  entity_label: string;
  keywords: string[];
}

const delay = (ms: number) => new Promise(resolve => setTimeout(resolve, ms));

// Wrapper de Retry com detecção de Cota
async function retryOperation<T>(operation: () => Promise<T>, maxRetries: number = 3, initialDelay: number = 2000): Promise<T> {
  if (globalQuotaExceeded) throw new Error("Circuit Breaker Open: Quota Exceeded");

  let lastError: any;
  
  for (let i = 0; i < maxRetries; i++) {
    try {
      return await operation();
    } catch (error: any) {
      lastError = error;
      
      const isRateLimit = error.message?.includes('429') || error.status === 429 || error.code === 429 || error.message?.includes('Quota exceeded') || error.message?.includes('Resource has been exhausted');
      const isServerOverload = error.message?.includes('503') || error.status === 503;
      
      if (isRateLimit) {
        console.warn(`[Gemini API] Cota atingida (Tentativa ${i + 1}/${maxRetries}).`);
        if (i === maxRetries - 1) {
            // Se falhou na última tentativa por cota, ativa o Circuit Breaker
            console.error("[Gemini API] COTA EXCEDIDA CRITICAMENTE. ATIVANDO MODO HEURÍSTICO GLOBAL.");
            globalQuotaExceeded = true;
        } else {
            // Backoff agressivo para rate limit
            const waitTime = initialDelay * Math.pow(3, i) + (Math.random() * 2000);
            await delay(waitTime);
            continue;
        }
      } else if (isServerOverload) {
        const waitTime = initialDelay * Math.pow(2, i);
        await delay(waitTime);
        continue;
      }
      
      throw error; 
    }
  }
  throw lastError;
}

// --- HEURISTIC FALLBACK (Regex) ---
const heuristicEnrichment = (chunk: DocumentChunk): DocumentChunk => {
    const cleanContent = chunk.content.trim();
    const firstLine = cleanContent.split(/\r?\n/)[0] || "";
    
    let label = "Texto";
    let type = "Conteúdo";
    
    if (/^Art\.\s*\d+/i.test(firstLine)) {
         type = "Artigo";
         label = firstLine.match(/^Art\.\s*\d+[º°]?/i)?.[0] || "Artigo";
    } else if (/^§\s*\d+/i.test(firstLine) || /^Parágrafo/i.test(firstLine)) {
         type = "Parágrafo";
         label = firstLine.match(/^§\s*\d+[º°]?/i)?.[0] || "§";
    } else if (/^[IVXLCDM]+\s*[\.\-]/i.test(firstLine)) {
         type = "Inciso";
         label = firstLine.split(/[\.\-]/)[0];
    } else if (/^CAP[IÍ]TULO/i.test(firstLine)) {
         type = "Capítulo";
         label = firstLine;
    } else {
         label = firstLine.substring(0, 25) + (firstLine.length > 25 ? "..." : "");
    }

    return {
        ...chunk,
        entityType: type,
        entityLabel: label,
        keywords: ["Extração Heurística", "Modo Offline"]
    };
};

export const generateHyDEAnswer = async (query: string): Promise<string> => {
    if (globalQuotaExceeded) return query; // Fail fast
    try {
        const response = await retryOperation(async () => {
            return await ai.models.generateContent({
                model: modelName,
                contents: `Atue como um jurista. Query: "${query}". Escreva um parágrafo de resposta ideal técnica.`,
            });
        });
        return response.text || "";
    } catch (e) {
        return query; 
    }
};

export const evaluateChunkRelevance = async (query: string, chunkContent: string): Promise<{relevant: boolean, score: number, reasoning: string}> => {
    if (globalQuotaExceeded) return { relevant: true, score: 0.5, reasoning: "Circuit Breaker Active" };
    try {
        const response = await retryOperation(async () => {
            return await ai.models.generateContent({
                model: modelName,
                contents: `Query: "${query}"\nContext: "${chunkContent.substring(0, 500)}"\nIs relevant? JSON: { "score": 0.0-1.0, "relevant": bool }`,
                config: { responseMimeType: "application/json" }
            });
        });
        const result = JSON.parse(response.text || "{}");
        return {
            relevant: result.relevant === true || (result.score || 0) > 0.6,
            score: result.score ?? 0,
            reasoning: "AI Evaluation"
        };
    } catch (e) {
        return { relevant: true, score: 0.5, reasoning: "Evaluation Failed" };
    }
};

export const generateSingleEmbedding = async (text: string): Promise<number[]> => {
    if (globalQuotaExceeded) return new Array(768).fill(0);
    try {
        const result = await retryOperation(async () => {
            return await ai.models.embedContent({
                model: embeddingModelName,
                contents: text.substring(0, 2048),
            });
        });
        return result.embedding?.values || [];
    } catch (e) {
        return new Array(768).fill(0);
    }
};

export const generateRAGResponse = async (query: string, context: string, chatHistory: any[]): Promise<string> => {
    if (globalQuotaExceeded) return "O limite de requisições da IA foi excedido. Não é possível gerar uma nova resposta agora.";
    try {
        const historyText = chatHistory.slice(-4).map(m => `${m.role}: ${m.content}`).join('\n');
        const response = await retryOperation(async () => {
            return await ai.models.generateContent({
                model: modelName,
                contents: `CONTEXTO:\n${context}\n\nHISTÓRICO:\n${historyText}\n\nPERGUNTA: ${query}\n\nRESPOSTA:`
            });
        });
        return response.text || "Erro na geração.";
    } catch (e) {
        return "Erro de geração.";
    }
};

export const analyzeChunkWithGemini = async (chunk: DocumentChunk): Promise<DocumentChunk> => {
  // CIRCUIT BREAKER CHECK
  if (globalQuotaExceeded) {
      return heuristicEnrichment(chunk);
  }

  if (!chunk.content || chunk.content.length < 5) return chunk;
  const safeContent = chunk.content.slice(0, 1000);

  try {
    const prompt = `
      INPUT: "${safeContent}"
      TASK: Classify (Artigo, Inciso, Texto) and Label. Extract 3 keywords.
      OUTPUT JSON: { "cleaned_text": "string", "entity_type": "string", "entity_label": "string", "keywords": ["k1"] }
    `;

    const response = await retryOperation(async () => {
      return await ai.models.generateContent({
        model: modelName,
        contents: prompt,
        config: {
          temperature: 0.1,
          responseMimeType: "application/json"
        }
      });
    }, 2, 2000); // Mais delay inicial

    const text = response.text || "{}";
    const result = JSON.parse(text) as GeminiChunkResponse;

    return {
      ...chunk,
      content: result.cleaned_text || chunk.content,
      entityType: result.entity_type || "Texto",
      entityLabel: result.entity_label || "Auto",
      keywords: result.keywords || []
    };

  } catch (error) {
    // Se falhar (incluindo Circuit Breaker ativando agora), usa Heurística
    return heuristicEnrichment(chunk);
  }
};

export const enhanceChunksWithAI = async (chunks: DocumentChunk[], onProgress: (progress: number) => void): Promise<DocumentChunk[]> => {
  // Reset circuit breaker on new run, unless it was a hard block? 
  // Better to reset and try again, but maybe flag sticks for session.
  // globalQuotaExceeded = false; // Uncomment to force retry on new button click
  
  const enhancedChunks: DocumentChunk[] = [];
  // Reduzir batch size para evitar 429 agressivo
  const batchSize = 2; 
  
  for (let i = 0; i < chunks.length; i += batchSize) {
    const batch = chunks.slice(i, i + batchSize);
    
    // Se o Circuit Breaker abriu, processa o resto síncronamente (rápido)
    if (globalQuotaExceeded) {
        const results = batch.map(c => heuristicEnrichment(c));
        enhancedChunks.push(...results);
        onProgress(Math.round(((i + batch.length) / chunks.length) * 100));
        await delay(5); // Pequeno respiro para UI
        continue;
    }

    const results = await Promise.all(batch.map(async (c, idx) => {
        await delay(idx * 500); // Maior stagger entre requests do mesmo batch
        return analyzeChunkWithGemini(c);
    }));
    
    enhancedChunks.push(...results);
    const progress = Math.round(((i + batch.length) / chunks.length) * 100);
    onProgress(progress);
    
    // Delay entre batches
    await delay(1000);
  }
  return enhancedChunks;
};

export const generateRealEmbeddingsWithGemini = async (chunks: DocumentChunk[], onProgress: (progress: number) => void): Promise<EmbeddingVector[]> => {
  const embeddings: EmbeddingVector[] = [];
  const batchSize = 3; 
  
  for (let i = 0; i < chunks.length; i += batchSize) {
    const batch = chunks.slice(i, i + batchSize);
    
    // Fallback rápido se cota estourou
    if (globalQuotaExceeded) {
         const results = batch.map(chunk => ({
                id: chunk.id,
                vector: new Array(768).fill(0).map(() => Math.random() * 0.1), // Noise vector
                contentSummary: chunk.content.substring(0, 50) + '...',
                fullContent: chunk.content,
                dueDate: chunk.dueDate,
                entityType: chunk.entityType,
                entityLabel: chunk.entityLabel,
                keywords: chunk.keywords,
                modelUsed: 'Heuristic/Offline'
         }));
         embeddings.push(...results);
         onProgress(Math.round(((i + batch.length) / chunks.length) * 100));
         await delay(5);
         continue;
    }

    const batchResults = await Promise.all(batch.map(async (chunk, idx) => {
        await delay(idx * 300); 

        try {
            const richContent = `Type: ${chunk.entityType}\nLabel: ${chunk.entityLabel}\nContent: ${chunk.content}`;
            
            const result = await retryOperation(async () => {
                return await ai.models.embedContent({
                    model: embeddingModelName,
                    contents: richContent, 
                });
            }, 2, 2000);
            
            return {
                id: chunk.id,
                vector: result.embedding?.values || [],
                contentSummary: chunk.content.substring(0, 50) + '...',
                fullContent: chunk.content,
                dueDate: chunk.dueDate,
                entityType: chunk.entityType,
                entityLabel: chunk.entityLabel,
                keywords: chunk.keywords,
                modelUsed: `Gemini ${embeddingModelName}`
            };
        } catch (e: any) {
            // Em caso de erro individual ou CB ativando no meio do batch
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
    await delay(500);
  }
  return embeddings;
};
