
import { GoogleGenAI, Type } from "@google/genai";
import { DocumentChunk, EmbeddingVector } from "../types";

// Inicializa o cliente Gemini
// Assume que process.env.API_KEY está disponível
const ai = new GoogleGenAI({ apiKey: process.env.API_KEY });

// Utiliza o modelo flash-exp para maior velocidade e menor latência
const modelName = 'gemini-2.0-flash-exp';
const embeddingModelName = 'text-embedding-004';

interface GeminiChunkResponse {
  cleaned_text: string;
  entity_type: string;
  entity_label: string;
  keywords: string[];
}

// Helper para delay (promessa de espera)
const delay = (ms: number) => new Promise(resolve => setTimeout(resolve, ms));

// Wrapper de Retry com Backoff Exponencial
async function retryOperation<T>(operation: () => Promise<T>, maxRetries: number = 6, initialDelay: number = 2000): Promise<T> {
  let lastError: any;
  
  for (let i = 0; i < maxRetries; i++) {
    try {
      return await operation();
    } catch (error: any) {
      lastError = error;
      
      // Identifica erros de cota (429) ou sobrecarga de servidor (503) ou erro interno (500)
      const isRateLimit = error.message?.includes('429') || error.status === 429 || error.code === 429 || error.message?.includes('Quota exceeded');
      const isServerOverload = error.message?.includes('503') || error.status === 503;
      const isInternal = error.message?.includes('500') || error.status === 500;
      
      if (isRateLimit || isServerOverload || isInternal) {
        // Backoff exponencial
        const waitTime = initialDelay * Math.pow(2, i) + (Math.random() * 1000);
        console.warn(`[Gemini API] Erro de Cota/Servidor (Tentativa ${i + 1}/${maxRetries}). Aguardando ${Math.round(waitTime)}ms... Detalhe: ${error.message}`);
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
Você é um especialista jurídico e acadêmico.
Tarefa: Gere um parágrafo hipotético ideal que responderia à pergunta abaixo.
Não responda a pergunta diretamente, apenas escreva o TEXTO que conteria a resposta em um documento oficial, lei ou artigo científico.
Pergunta: "${query}"
Resposta Hipotética (Estilo Jurídico/Acadêmico):
                `,
            });
        });
        return response.text || "";
    } catch (e) {
        console.error("Erro HyDE:", e);
        return query; // Fallback para a query original
    }
};

// --- CRAG: CORRECTIVE RAG EVALUATOR ---
export const evaluateChunkRelevance = async (query: string, chunkContent: string): Promise<{relevant: boolean, score: number, reasoning: string}> => {
    try {
        const response = await retryOperation(async () => {
            return await ai.models.generateContent({
                model: modelName,
                contents: `
Atue como um juiz corretivo de RAG (CRAG).
Query: "${query}"
Documento Recuperado: "${chunkContent.substring(0, 500)}..."

Avalie se o documento contém informações úteis para responder à query.
Retorne JSON: { "score": (0.0 a 1.0), "relevant": (boolean, true se score > 0.5), "reasoning": "curta justificativa" }
                `,
                config: {
                    responseMimeType: "application/json"
                }
            });
        });
        const result = JSON.parse(response.text || "{}");
        return {
            relevant: result.relevant ?? false,
            score: result.score ?? 0,
            reasoning: result.reasoning ?? "N/A"
        };
    } catch (e) {
        console.error("Erro CRAG:", e);
        return { relevant: true, score: 0.5, reasoning: "Falha na avaliação, mantendo por segurança." };
    }
};

// --- SINGLE EMBEDDING (FOR QUERY) ---
export const generateSingleEmbedding = async (text: string): Promise<number[]> => {
    try {
        const result = await retryOperation(async () => {
            return await ai.models.embedContent({
                model: embeddingModelName,
                contents: text, 
            });
        });
        return result.embedding?.values || [];
    } catch (e) {
        console.error("Erro Embedding Query:", e);
        return new Array(768).fill(0);
    }
};

// --- FINAL GENERATION (AGENTIC/MEMORY) ---
export const generateRAGResponse = async (
    query: string, 
    context: string, 
    chatHistory: {role: string, content: string}[]
): Promise<string> => {
    try {
        const historyText = chatHistory.map(m => `${m.role.toUpperCase()}: ${m.content}`).join('\n');
        
        const response = await retryOperation(async () => {
            return await ai.models.generateContent({
                model: modelName,
                contents: `
Histórico da Conversa:
${historyText}

Contexto Recuperado (GraphRAG + Vector):
${context}

Usuário: ${query}

Instruções:
1. Responda com base ESTRITAMENTE no contexto.
2. Cite as fontes (ex: [Art. 5º]) se disponíveis.
3. Se o contexto não for suficiente, diga que não sabe, não alucine.
4. Seja conciso e direto.
                `
            });
        });
        return response.text || "Erro na geração.";
    } catch (e) {
        return "Erro ao gerar resposta final.";
    }
};

/**
 * Processa um único chunk usando Gemini para limpar, classificar e extrair entidades.
 */
export const analyzeChunkWithGemini = async (chunk: DocumentChunk): Promise<DocumentChunk> => {
  try {
    const prompt = `
      Você é um especialista em processamento de documentos legais e acadêmicos (Data Cleaning & NLP).
      
      Sua tarefa é processar o seguinte fragmento de texto extraído de um PDF:
      "${chunk.content}"
      
      Realize as seguintes operações com rigor acadêmico:
      1. **Limpeza**: Remova quebras de linha desnecessárias, hifens de fim de linha, números de página soltos e caracteres estranhos.
      2. **Classificação Hierárquica**: Identifique o tipo do fragmento. (Ex: ARTIGO, INCISO, PARAGRAFO, CAPITULO, DEFINICAO).
      3. **Rotulagem**: Crie um rótulo curto e identificável (Ex: "Art. 5º", "Definição de RAG").
      4. **Extração de Entidades**: Liste as 3 a 5 principais palavras-chave.
      
      Retorne APENAS o JSON.
    `;

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
              keywords: { type: Type.ARRAY, items: { type: Type.STRING } }
            }
          }
        }
      });
    }, 5, 3000);

    const result = JSON.parse(response.text || "{}") as GeminiChunkResponse;

    return {
      ...chunk,
      content: result.cleaned_text || chunk.content, 
      entityType: result.entity_type || chunk.entityType,
      entityLabel: result.entity_label || chunk.entityLabel,
      keywords: result.keywords || []
    };

  } catch (error: any) {
    console.error(`Erro processar chunk ${chunk.id.substring(0,8)}:`, error.message);
    return chunk;
  }
};

export const enhanceChunksWithAI = async (chunks: DocumentChunk[], onProgress: (progress: number) => void): Promise<DocumentChunk[]> => {
  const enhancedChunks: DocumentChunk[] = [];
  const batchSize = 3; 
  
  for (let i = 0; i < chunks.length; i += batchSize) {
    const batch = chunks.slice(i, i + batchSize);
    
    const results = await Promise.all(batch.map(async (c, idx) => {
        await delay(idx * 500); 
        return analyzeChunkWithGemini(c);
    }));
    
    enhancedChunks.push(...results);
    if (i + batchSize < chunks.length) await delay(2000); 
    const progress = Math.round(((i + batch.length) / chunks.length) * 100);
    onProgress(progress);
  }
  return enhancedChunks;
};

export const generateRealEmbeddingsWithGemini = async (chunks: DocumentChunk[], onProgress: (progress: number) => void): Promise<EmbeddingVector[]> => {
  const embeddings: EmbeddingVector[] = [];
  const batchSize = 3; 
  
  for (let i = 0; i < chunks.length; i += batchSize) {
    const batch = chunks.slice(i, i + batchSize);
    
    const batchResults = await Promise.all(batch.map(async (chunk, idx) => {
        await delay(idx * 600); 

        try {
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
                    contents: richContent, 
                });
            }, 6, 4000);
            
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
        } catch (e: any) {
            console.error(`Falha no embedding chunk ${chunk.id}:`, e.message);
            return {
                id: chunk.id,
                vector: new Array(768).fill(0).map(() => Math.random()), // Last resort fallback
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
    if (i + batchSize < chunks.length) await delay(2500);
    const progress = Math.round(((i + batch.length) / chunks.length) * 100);
    onProgress(progress);
  }
  return embeddings;
};
