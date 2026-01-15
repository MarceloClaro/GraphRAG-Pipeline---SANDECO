import { GoogleGenAI, Type } from "@google/genai";
import { DocumentChunk, EmbeddingVector } from "../types";

// Inicializa o cliente Gemini
// Assume que process.env.API_KEY está disponível
const ai = new GoogleGenAI({ apiKey: process.env.API_KEY });

// Switch to gemini-2.0-flash-exp as gemini-3-flash-preview (from prompt guidelines) 
// might not be available to all API keys yet, causing 404.
const modelName = 'gemini-2.0-flash-exp';
const embeddingModelName = 'text-embedding-004';

interface GeminiChunkResponse {
  cleaned_text: string;
  entity_type: string;
  entity_label: string;
  keywords: string[];
}

// Helper for delay
const delay = (ms: number) => new Promise(resolve => setTimeout(resolve, ms));

// Retry wrapper with exponential backoff
async function retryOperation<T>(operation: () => Promise<T>, maxRetries: number = 3, initialDelay: number = 2000): Promise<T> {
  let lastError: any;
  
  for (let i = 0; i < maxRetries; i++) {
    try {
      return await operation();
    } catch (error: any) {
      lastError = error;
      
      // Check for 429 (Resource Exhausted) or 503 (Service Unavailable)
      const isRateLimit = error.message?.includes('429') || error.status === 429 || error.code === 429;
      const isServerOverload = error.message?.includes('503') || error.status === 503;
      
      // Also catch 404 if it's a transient model loading issue, though usually it's permanent.
      // We mainly retry on rate limits here.

      if (isRateLimit || isServerOverload) {
        const waitTime = initialDelay * Math.pow(2, i); // 2s, 4s, 8s
        console.warn(`Gemini API throttled (Attempt ${i + 1}/${maxRetries}). Retrying in ${waitTime}ms...`);
        await delay(waitTime);
        continue;
      }
      
      // Throw other errors immediately (e.g., 400 Bad Request, 404 Not Found)
      throw error; 
    }
  }
  throw lastError;
}

/**
 * Processa um único chunk usando Gemini para limpar, classificar e extrair entidades.
 */
export const analyzeChunkWithGemini = async (chunk: DocumentChunk): Promise<DocumentChunk> => {
  try {
    const prompt = `
      Você é um especialista em processamento de documentos legais e acadêmicos (Data Cleaning & NLP).
      
      Sua tarefa é processar o seguinte fragmento de texto extraído de um PDF:
      "${chunk.content}"
      
      Realize as seguintes operações:
      1. **Limpeza**: Remova quebras de linha desnecessárias, hifens de fim de linha, números de página soltos e caracteres estranhos. O texto deve ficar fluido e legível.
      2. **Classificação Hierárquica**: Identifique o tipo do fragmento. (Ex: ARTIGO, INCISO, PARAGRAFO, CAPITULO, DEFINICAO, CONCEITO, EMENTA, BIBLIOGRAFIA).
      3. **Rotulagem**: Crie um rótulo curto e identificável (Ex: "Art. 5º", "Definição de RAG", "Conclusão").
      4. **Extração de Entidades**: Liste as 3 a 5 principais palavras-chave ou entidades técnicas presentes neste fragmento para criação de um grafo de conhecimento.
      
      Retorne APENAS o JSON.
    `;

    // Wrap API call with retry logic
    const response = await retryOperation(async () => {
      return await ai.models.generateContent({
        model: modelName,
        contents: prompt,
        config: {
          responseMimeType: "application/json",
          responseSchema: {
            type: Type.OBJECT,
            properties: {
              cleaned_text: { type: Type.STRING },
              entity_type: { type: Type.STRING },
              entity_label: { type: Type.STRING },
              keywords: { 
                type: Type.ARRAY,
                items: { type: Type.STRING }
              }
            }
          }
        }
      });
    });

    const result = JSON.parse(response.text || "{}") as GeminiChunkResponse;

    return {
      ...chunk,
      content: result.cleaned_text || chunk.content, // Atualiza com texto limpo
      entityType: result.entity_type || chunk.entityType,
      entityLabel: result.entity_label || chunk.entityLabel,
      keywords: result.keywords || []
    };

  } catch (error: any) {
    console.error(`Erro ao processar chunk ${chunk.id.substring(0,8)} com Gemini (${modelName}):`, error.message || error);
    return chunk;
  }
};

/**
 * Processa uma lista de chunks em paralelo (com limite de concorrência para evitar rate limit).
 */
export const enhanceChunksWithAI = async (chunks: DocumentChunk[], onProgress: (progress: number) => void): Promise<DocumentChunk[]> => {
  const enhancedChunks: DocumentChunk[] = [];
  const batchSize = 3; 
  
  for (let i = 0; i < chunks.length; i += batchSize) {
    const batch = chunks.slice(i, i + batchSize);
    const results = await Promise.all(batch.map(c => analyzeChunkWithGemini(c)));
    enhancedChunks.push(...results);
    
    if (i + batchSize < chunks.length) await delay(1000); 
    
    const progress = Math.round(((i + batch.length) / chunks.length) * 100);
    onProgress(progress);
  }
  
  return enhancedChunks;
};

/**
 * Gera embeddings reais usando o modelo text-embedding-004 do Gemini.
 */
export const generateRealEmbeddingsWithGemini = async (chunks: DocumentChunk[], onProgress: (progress: number) => void): Promise<EmbeddingVector[]> => {
  const embeddings: EmbeddingVector[] = [];
  const batchSize = 10; // Embeddings endpoint usually has higher throughput
  
  for (let i = 0; i < chunks.length; i += batchSize) {
    const batch = chunks.slice(i, i + batchSize);
    
    // Process batch in parallel but carefully
    const batchResults = await Promise.all(batch.map(async (chunk) => {
        try {
            // Construct a rich content string that includes metadata for higher fidelity embeddings
            // This ensures the vector captures the classification (Type, Label) and Keywords, not just raw text.
            const richContent = `
Tipo: ${chunk.entityType || 'Texto Genérico'}
Rótulo: ${chunk.entityLabel || 'Sem Rótulo'}
Palavras-Chave: ${chunk.keywords?.join(', ') || ''}

Conteúdo:
${chunk.content}
            `.trim();

            const result = await retryOperation(async () => {
                return await ai.models.embedContent({
                    model: embeddingModelName,
                    content: richContent, 
                });
            });
            
            // Return full vector
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
                modelUsed: `Gemini ${embeddingModelName} (High-Fidelity)`
            };
        } catch (e) {
            console.error("Falha ao gerar embedding para chunk", chunk.id, e);
            // Fallback para vetor zerado ou aleatório em caso de falha crítica para não parar pipeline
            // Use 768 dimensions as that is standard for text-embedding-004
            return {
                id: chunk.id,
                vector: new Array(768).fill(0).map(() => Math.random()),
                contentSummary: chunk.content.substring(0, 50) + '...',
                fullContent: chunk.content,
                dueDate: chunk.dueDate,
                entityType: chunk.entityType,
                entityLabel: chunk.entityLabel,
                keywords: chunk.keywords,
                modelUsed: 'ERROR_FALLBACK'
            };
        }
    }));
    
    embeddings.push(...batchResults);
    
    // Rate limit safeguard
    if (i + batchSize < chunks.length) await delay(500);

    const progress = Math.round(((i + batch.length) / chunks.length) * 100);
    onProgress(progress);
  }

  return embeddings;
};