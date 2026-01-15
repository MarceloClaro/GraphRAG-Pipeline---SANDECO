
import React, { useState, useRef } from 'react';
import { PipelineStage, DocumentChunk, EmbeddingVector, ClusterPoint, GraphData, EmbeddingModelType, CNNHyperParameters, TrainingMetrics, RAGStepLog, ChatMessage } from './types';
import { 
  processRealPDFsToChunks, 
  generateClustersFromEmbeddings, 
  generateGraphFromClusters 
} from './services/mockDataService';
import { enhanceChunksWithAI, generateRealEmbeddingsWithGemini, generateHyDEAnswer, generateSingleEmbedding, evaluateChunkRelevance, generateRAGResponse } from './services/geminiService';
import { trainCNNWithTripletLoss, cosineSimilarity } from './services/cnnRefinementService';
import { extractTextFromPDF } from './services/pdfService';
import { downloadCSV } from './services/exportService';
import { generateTechnicalReport } from './services/reportService';
import PipelineProgress from './components/PipelineProgress';
import FullContentModal from './components/FullContentModal';
import ForceGraph, { ForceGraphRef } from './components/charts/ForceGraph';
import GraphMetricsDashboard from './components/GraphMetricsDashboard';
import ClusterAnalysisPanel from './components/ClusterAnalysisPanel';
import { ScatterChart, Scatter, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell } from 'recharts';

const App: React.FC = () => {
  // State
  const [stage, setStage] = useState<PipelineStage>(PipelineStage.UPLOAD);
  const [chunks, setChunks] = useState<DocumentChunk[]>([]);
  const [embeddings, setEmbeddings] = useState<EmbeddingVector[]>([]);
  const [clusters, setClusters] = useState<ClusterPoint[]>([]);
  const [graphData, setGraphData] = useState<GraphData | null>(null);
  const [highlightedClusters, setHighlightedClusters] = useState<number[]>([]);
  
  // Refs
  const graphRef = useRef<ForceGraphRef>(null);

  // Settings
  const embeddingModel: EmbeddingModelType = 'gemini-004';
  
  // CNN
  const [cnnParams, setCnnParams] = useState<CNNHyperParameters>({
    margin: 0.2,
    learningRate: 0.005,
    epochs: 15,
    miningStrategy: 'hard',
    optimizer: 'adamw'
  });
  const [trainingMetrics, setTrainingMetrics] = useState<TrainingMetrics | null>(null);
  
  // Processing
  const [isProcessing, setIsProcessing] = useState(false);
  const [processingStatus, setProcessingStatus] = useState("Processando...");
  const [uploadError, setUploadError] = useState<string | null>(null);
  const [aiEnhanced, setAiEnhanced] = useState(false);
  
  // Modal
  const [modalOpen, setModalOpen] = useState(false);
  const [modalContent, setModalContent] = useState({ title: '', text: '' });
  const [reportOpen, setReportOpen] = useState(false);
  const [reportText, setReportText] = useState('');

  // RAG Lab State
  const [userQuery, setUserQuery] = useState('');
  const [ragLogs, setRagLogs] = useState<RAGStepLog[]>([]);
  const [chatHistory, setChatHistory] = useState<ChatMessage[]>([]);
  const [isRagThinking, setIsRagThinking] = useState(false);

  // --- Handlers existentes (Upload, CNN, Cluster, Graph) ---
  const handleFileUpload = async (event: React.ChangeEvent<HTMLInputElement>) => {
    if (!event.target.files || event.target.files.length === 0) return;
    setIsProcessing(true);
    setProcessingStatus("Lendo PDF e extraindo texto...");
    setUploadError(null);
    setAiEnhanced(false);
    try {
      const files = Array.from(event.target.files);
      const extractedDocs = [];
      for (const file of files) {
        if (file.type === "application/pdf") {
          const processed = await extractTextFromPDF(file);
          extractedDocs.push(processed);
        } else {
           const text = await file.text();
           extractedDocs.push({ filename: file.name, text: text, pageCount: 1 });
        }
      }
      const generatedChunks = processRealPDFsToChunks(extractedDocs);
      if (generatedChunks.length === 0) {
        setUploadError("Nenhum conteúdo de texto pôde ser extraído dos arquivos.");
      } else {
        setChunks(generatedChunks);
        setStage(PipelineStage.UPLOAD);
      }
    } catch (err) {
      console.error(err);
      setUploadError("Erro ao processar arquivos.");
    } finally {
      setIsProcessing(false);
    }
  };

  const handleEnhanceWithAI = async () => {
    if (chunks.length === 0) return;
    setIsProcessing(true);
    setProcessingStatus("Gemini AI: Limpando texto e identificando entidades...");
    try {
      const enhanced = await enhanceChunksWithAI(chunks, (progress) => {
        setProcessingStatus(`Gemini AI: Processando chunks... ${progress}%`);
      });
      setChunks(enhanced);
      setAiEnhanced(true);
    } catch (err) {
      console.error("Erro na IA", err);
      setUploadError("Falha ao conectar com Gemini API.");
    } finally {
      setIsProcessing(false);
    }
  };

  const handleProcessEmbeddings = async () => {
    setIsProcessing(true);
    setProcessingStatus("Inicializando modelo de embedding...");
    setTrainingMetrics(null); 
    await new Promise(r => setTimeout(r, 100));
    try {
        setProcessingStatus("Gerando Embeddings Reais via Gemini API (High-Fidelity)...");
        const embeds = await generateRealEmbeddingsWithGemini(chunks, (progress) => {
            setProcessingStatus(`Gerando vetores (Gemini-004)... ${progress}%`);
        });
        setEmbeddings(embeds);
        setStage(PipelineStage.EMBEDDINGS);
    } catch (e) {
        console.error(e);
        setUploadError("Erro na geração de embeddings.");
    } finally {
        setIsProcessing(false);
    }
  };

  const handleRunCNNTraining = async () => {
      if (embeddings.length === 0) return;
      setIsProcessing(true);
      setProcessingStatus("Inicializando CNN com Triplet Loss...");
      try {
          await trainCNNWithTripletLoss(embeddings, cnnParams, (metrics, updatedEmbeddings) => {
              setProcessingStatus(`Epoch ${metrics.currentEpoch}/${cnnParams.epochs} | Train Loss: ${metrics.trainLoss.toFixed(4)} | Val Loss: ${metrics.valLoss.toFixed(4)}`);
              setTrainingMetrics(metrics);
              setEmbeddings(updatedEmbeddings); 
          });
          setProcessingStatus("Treinamento concluído.");
      } catch (err) {
          console.error("Erro no treinamento CNN", err);
          setUploadError("Falha no refinamento CNN.");
      } finally {
          setIsProcessing(false);
      }
  };

  const handleRunClustering = () => {
    setIsProcessing(true);
    setProcessingStatus("Calculando clusters (K-Means)...");
    setTimeout(() => {
        const clusterPoints = generateClustersFromEmbeddings(embeddings);
        setClusters(clusterPoints);
        setStage(PipelineStage.CLUSTERING);
        setIsProcessing(false);
    }, 800);
  };

  const handleBuildGraph = () => {
    setIsProcessing(true);
    setProcessingStatus("Construindo grafo de conhecimento...");
    setTimeout(() => {
        const graph = generateGraphFromClusters(clusters);
        setGraphData(graph);
        setStage(PipelineStage.GRAPH);
        setIsProcessing(false);
    }, 1000);
  };

  const handleGenerateReport = () => {
    const report = generateTechnicalReport(chunks, embeddings, graphData, embeddingModel);
    setReportText(report);
    setReportOpen(true);
  };

  const handleUnifiedExport = () => {
    if (!graphData || chunks.length === 0) {
        alert("O grafo precisa ser construído.");
        return;
    }
    const unifiedData = chunks.map(chunk => {
        const embedding = embeddings.find(e => e.id === chunk.id);
        const cluster = clusters.find(c => c.id === chunk.id);
        const graphNode = graphData.nodes.find(n => n.id === chunk.id);
        const degree = graphData.links.filter(l => l.source === chunk.id || l.target === chunk.id).length;
        const vectorSample = embedding ? `[${embedding.vector.slice(0, 5).map(v => v.toFixed(4)).join('; ')}...]` : 'N/A';
        return {
            Chunk_ID: chunk.id,
            Arquivo: chunk.source,
            Tipo_IA: chunk.entityType || 'N/A',
            Rotulo: chunk.entityLabel || 'N/A',
            Palavras_Chave: chunk.keywords ? chunk.keywords.join('; ') : '',
            Conteudo_Preview: chunk.content.substring(0, 250).replace(/(\r\n|\n|\r)/gm, " "),
            Tokens: chunk.tokens,
            Provedor_IA: "Google Gemini",
            Modelo_Embedding: embedding?.modelUsed || embeddingModel,
            Dim_Embedding: embedding?.vector.length || 0,
            Vetor_Sample: vectorSample,
            Cluster_ID: cluster ? cluster.clusterId : -1,
            Cluster_X: cluster ? cluster.x.toFixed(4) : 0,
            Cluster_Y: cluster ? cluster.y.toFixed(4) : 0,
            Grafo_Grupo: graphNode ? graphNode.group : -1,
            Grafo_Centralidade: graphNode ? graphNode.centrality.toFixed(5) : 0,
            Grau_Arestas: degree,
            Palavras_Grafo: graphNode?.keywords ? graphNode.keywords.join('; ') : '',
            Etapa_Atual: "GRAFO_CONCLUIDO"
        };
    });
    downloadCSV(unifiedData, `GraphRAG_Dataset_Completo_${new Date().toISOString().split('T')[0]}.csv`);
  };

  const openModal = (title: string, content: string) => {
    setModalContent({ title, text: content });
    setModalOpen(true);
  };

  // --- RAG LAB LOGIC (NO SIMULATION) ---
  const handleRAGQuery = async () => {
      if (!userQuery.trim() || !graphData) return;
      setIsRagThinking(true);
      setRagLogs([]); // Reset logs
      
      const logs: RAGStepLog[] = [];
      const addLog = (l: RAGStepLog) => { logs.push(l); setRagLogs([...logs]); };

      try {
          // 1. HyDE
          addLog({ step: 'HYDE', description: 'Gerando Documento Hipotético Ideal...', status: 'warning' });
          const hydeDoc = await generateHyDEAnswer(userQuery);
          addLog({ step: 'HYDE', description: 'HyDE Gerado', data: hydeDoc, status: 'success' });

          // 2. Vector Retrieval (Embedding + Cosine)
          addLog({ step: 'RETRIEVAL', description: 'Vetorizando Query & HyDE...', status: 'warning' });
          const queryVector = await generateSingleEmbedding(`${userQuery}\n${hydeDoc}`);
          
          const similarities = embeddings.map(emb => ({
              id: emb.id,
              score: cosineSimilarity(queryVector, emb.vector),
              content: emb.fullContent,
              label: emb.entityLabel
          })).sort((a, b) => b.score - a.score).slice(0, 6); // Top 6 candidates
          
          addLog({ step: 'RETRIEVAL', description: `Top 6 Chunks Recuperados (Score Máx: ${similarities[0]?.score.toFixed(4)})`, data: similarities.map(s => `${s.label} (${s.score.toFixed(2)})`), status: 'success' });

          // 3. CRAG (Corrective RAG)
          addLog({ step: 'CRAG', description: 'Avaliando relevância dos chunks (LLM Judge)...', status: 'warning' });
          const validChunks: any[] = [];
          for (const chunk of similarities) {
              const evalResult = await evaluateChunkRelevance(userQuery, chunk.content);
              if (evalResult.relevant) {
                  validChunks.push(chunk);
              }
          }
          addLog({ step: 'CRAG', description: `Filtragem: ${validChunks.length}/${similarities.length} chunks aprovados.`, status: validChunks.length > 0 ? 'success' : 'error' });

          // 4. GraphRAG Expansion
          addLog({ step: 'GRAPHRAG', description: 'Explorando vizinhos no Grafo...', status: 'warning' });
          const contextSet = new Set<string>();
          validChunks.forEach(c => contextSet.add(c.content)); // Add Vector Results

          const expandedNodes: string[] = [];
          validChunks.forEach(c => {
               // Find Graph Node
               const node = graphData.nodes.find(n => n.id === c.id);
               if (node) {
                   // Find neighbors in edges
                   const neighbors = graphData.links
                    .filter(l => l.source === node.id || l.target === node.id)
                    .map(l => l.source === node.id ? l.target : l.source);
                   
                   // Add neighbor content (1-hop)
                   neighbors.forEach(nid => {
                       const neighborNode = graphData.nodes.find(n => n.id === nid);
                       if (neighborNode && !contextSet.has(neighborNode.fullContent)) {
                           contextSet.add(neighborNode.fullContent);
                           expandedNodes.push(neighborNode.label);
                       }
                   });
               }
          });
          
          if (expandedNodes.length > 0) {
              addLog({ step: 'GRAPHRAG', description: `Expansão: +${expandedNodes.length} nós adicionados via topologia.`, data: expandedNodes.slice(0, 5), status: 'success' });
          } else {
              addLog({ step: 'GRAPHRAG', description: 'Nenhuma expansão topológica relevante encontrada.', status: 'warning' });
          }

          // 5. Generation (Memory + Agentic)
          addLog({ step: 'GENERATION', description: 'Sintetizando resposta final com contexto...', status: 'warning' });
          const finalContext = Array.from(contextSet).join('\n---\n');
          const answer = await generateRAGResponse(userQuery, finalContext, chatHistory);
          
          // Update Chat History (Memory)
          const newHistory: ChatMessage[] = [
              ...chatHistory,
              { role: 'user', content: userQuery, timestamp: new Date().toISOString() },
              { role: 'assistant', content: answer, timestamp: new Date().toISOString() }
          ];
          setChatHistory(newHistory);
          addLog({ step: 'GENERATION', description: 'Resposta Gerada.', status: 'success' });

      } catch (err) {
          console.error(err);
          addLog({ step: 'GENERATION', description: 'Falha crítica no fluxo RAG.', status: 'error' });
      } finally {
          setIsRagThinking(false);
          setUserQuery('');
      }
  };

  return (
    <div className="min-h-screen bg-slate-50 text-slate-900 font-sans flex flex-col">
      <header className="bg-slate-900 text-white shadow-lg sticky top-0 z-40">
        <div className="max-w-7xl mx-auto px-4 py-4 flex justify-between items-center">
          <div className="flex items-center space-x-3">
            <div className="bg-indigo-500 p-2 rounded-lg">
              <svg className="w-6 h-6 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
              </svg>
            </div>
            <div>
              <h1 className="text-xl font-bold tracking-tight">GraphRAG Pipeline Visualizer</h1>
              <p className="text-xs text-slate-400">
                Autor: <strong>Prof. Marcelo Claro Laranjeira</strong> | Powered by Gemini AI (Real Data & RAG)
              </p>
            </div>
          </div>
          <div className="text-right hidden sm:block">
             <span className="bg-gradient-to-r from-emerald-500 to-teal-600 text-white text-xs px-2 py-1 rounded shadow-sm">Qualis A1 Mode</span>
          </div>
        </div>
      </header>

      <main className="max-w-7xl mx-auto px-4 py-8 flex-grow">
        <PipelineProgress currentStage={stage} />

        <div className="bg-white rounded-xl shadow-sm border border-slate-200 min-h-[600px] p-6 relative">
          
          {isProcessing && (
             <div className="absolute inset-0 bg-white/90 backdrop-blur-sm z-50 flex flex-col items-center justify-center rounded-xl">
                <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-indigo-600 mb-4"></div>
                <p className="text-indigo-800 font-medium animate-pulse">{processingStatus}</p>
             </div>
          )}

          <div className="flex justify-between items-center mb-6 pb-4 border-b border-slate-100 flex-wrap gap-4">
            <h2 className="text-2xl font-bold text-slate-800">
              {stage === PipelineStage.UPLOAD && "1. Ingestão Real & Refinamento AI"}
              {stage === PipelineStage.EMBEDDINGS && "2. Vetorização (Gemini-004)"}
              {stage === PipelineStage.CLUSTERING && "3. Clusterização Semântica"}
              {stage === PipelineStage.GRAPH && "4. Grafo de Conhecimento"}
              {stage === PipelineStage.QUERY && "5. Lab RAG (HyDE + CRAG + Graph)"}
            </h2>
            <div className="flex items-center space-x-3">
              
              {stage === PipelineStage.UPLOAD && chunks.length > 0 && (
                <>
                  {!aiEnhanced && (
                    <button onClick={handleEnhanceWithAI} disabled={isProcessing} className="bg-gradient-to-r from-purple-600 to-indigo-600 hover:from-purple-700 hover:to-indigo-700 text-white px-4 py-2 rounded-lg shadow-sm transition-all font-medium flex items-center">
                      <svg className="w-4 h-4 mr-2" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" /></svg> Enriquecer com Gemini
                    </button>
                  )}
                  <button onClick={handleProcessEmbeddings} disabled={isProcessing} className="bg-slate-800 hover:bg-slate-700 text-white px-4 py-2 rounded-lg shadow-sm transition-all font-medium flex items-center">
                    Gerar Embeddings Reais <span className="ml-2">→</span>
                  </button>
                </>
              )}
              {stage === PipelineStage.EMBEDDINGS && (
                <button onClick={handleRunClustering} disabled={isProcessing} className="bg-indigo-600 hover:bg-indigo-700 text-white px-4 py-2 rounded-lg shadow-sm transition-all font-medium flex items-center">
                  Executar Clusterização <span className="ml-2">→</span>
                </button>
              )}
              {stage === PipelineStage.CLUSTERING && (
                <button onClick={handleBuildGraph} disabled={isProcessing} className="bg-indigo-600 hover:bg-indigo-700 text-white px-4 py-2 rounded-lg shadow-sm transition-all font-medium flex items-center">
                  Construir Grafo <span className="ml-2">→</span>
                </button>
              )}
              {stage === PipelineStage.GRAPH && (
                <>
                <button onClick={() => setStage(PipelineStage.QUERY)} className="bg-rose-600 hover:bg-rose-700 text-white px-4 py-2 rounded-lg shadow-sm transition-all font-bold flex items-center">
                    🧪 Entrar no Lab RAG
                </button>
                <button onClick={handleUnifiedExport} className="bg-teal-600 hover:bg-teal-700 text-white px-4 py-2 rounded-lg shadow-sm transition-all font-medium flex items-center border-2 border-teal-500">
                    Dataset CSV
                </button>
                <button onClick={handleGenerateReport} className="bg-amber-600 hover:bg-amber-700 text-white px-4 py-2 rounded-lg shadow-sm transition-all font-medium flex items-center">
                    Relatório
                </button>
                </>
              )}
              {stage === PipelineStage.QUERY && (
                 <button onClick={() => setStage(PipelineStage.GRAPH)} className="bg-slate-200 hover:bg-slate-300 text-slate-800 px-4 py-2 rounded-lg shadow-sm transition-all font-medium">
                    Voltar ao Grafo
                 </button>
              )}
            </div>
          </div>

          {/* Views existentes (Upload, Embeddings, Clustering, Graph) mantidas... apenas adicionando o QUERY view abaixo */}
          
          {stage === PipelineStage.UPLOAD && (
            <div className="space-y-4">
              {uploadError && (<div className="bg-red-50 border border-red-200 text-red-700 px-4 py-3 rounded relative"><strong className="font-bold">Erro: </strong><span className="block sm:inline">{uploadError}</span></div>)}
              {chunks.length === 0 ? (
                <div className="flex flex-col items-center justify-center h-80 border-2 border-dashed border-slate-300 rounded-lg bg-slate-50 hover:bg-slate-100 hover:border-indigo-400">
                   <div className="text-center">
                      <svg className="mx-auto h-12 w-12 text-slate-400" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1} d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" /></svg>
                      <p className="mt-4 text-lg font-medium text-slate-700">Carregar Documentos PDF</p>
                      <label className="mt-6 inline-block cursor-pointer"><span className="bg-indigo-600 hover:bg-indigo-700 text-white px-6 py-3 rounded-lg shadow-md font-medium transition-colors">Selecionar Arquivos</span><input type="file" multiple accept=".pdf,application/pdf" className="hidden" onChange={handleFileUpload} /></label>
                   </div>
                </div>
              ) : (
                <div className="overflow-x-auto rounded-lg border border-slate-200">
                    <table className="min-w-full divide-y divide-slate-200">
                      <thead className="bg-slate-50"><tr><th className="px-6 py-3 text-left text-xs font-medium text-slate-500 uppercase">Rótulo</th><th className="px-6 py-3 text-left text-xs font-medium text-slate-500 uppercase">Conteúdo</th><th className="px-6 py-3 text-right text-xs font-medium text-slate-500 uppercase">Ação</th></tr></thead>
                      <tbody className="bg-white divide-y divide-slate-200">{chunks.slice(0, 10).map(c => (<tr key={c.id}><td className="px-6 py-4 text-sm font-semibold text-slate-800">{c.entityLabel}</td><td className="px-6 py-4 text-sm text-slate-500 truncate max-w-xs">{c.content}</td><td className="px-6 py-4 text-right"><button onClick={()=>openModal(c.entityLabel||'', c.content)} className="text-indigo-600 hover:underline">Ver</button></td></tr>))}</tbody>
                    </table>
                </div>
              )}
            </div>
          )}

          {stage === PipelineStage.EMBEDDINGS && (
            <div className="space-y-4">
              <div className="bg-slate-800 text-white p-4 rounded-lg shadow-lg mb-6">
                <div className="flex justify-between items-center border-b border-slate-700 pb-2 mb-3"><h3 className="font-bold">Refinamento CNN (Triplet Loss)</h3>{trainingMetrics && <div className="flex gap-3 text-xs"><span className="bg-green-600 px-2 py-1 rounded">Loss: {trainingMetrics.trainLoss.toFixed(4)}</span></div>}</div>
                <div className="flex justify-end"><button onClick={handleRunCNNTraining} disabled={isProcessing} className="bg-purple-600 hover:bg-purple-500 text-white px-4 py-2 rounded text-sm font-medium">Iniciar Treinamento</button></div>
              </div>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">{embeddings.slice(0, 6).map(emb => (<div key={emb.id} className="bg-white p-4 rounded-lg border border-slate-200"><span className="text-xs font-bold text-slate-600 block">{emb.entityLabel}</span><div className="font-mono text-xs text-green-600 break-all bg-slate-50 p-2 rounded">[{emb.vector.slice(0, 10).map(n => n.toFixed(3)).join(', ')}...]</div></div>))}</div>
            </div>
          )}

          {stage === PipelineStage.CLUSTERING && (
             <div className="h-[500px] w-full bg-slate-50 rounded-lg border border-slate-200 p-4">
                <ResponsiveContainer width="100%" height="100%"><ScatterChart><CartesianGrid /><XAxis type="number" dataKey="x" /><YAxis type="number" dataKey="y" /><Tooltip /><Scatter name="Docs" data={clusters} fill="#8884d8">{clusters.map((e, i) => <Cell key={`cell-${i}`} fill={['#3b82f6', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6'][e.clusterId % 5]} />)}</Scatter></ScatterChart></ResponsiveContainer>
             </div>
          )}

          {stage === PipelineStage.GRAPH && graphData && (
            <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
                <div className="lg:col-span-1"><ClusterAnalysisPanel graphData={graphData} onClusterSelect={(ids) => setHighlightedClusters(ids)} /></div>
                <div className="lg:col-span-3"><ForceGraph ref={graphRef} data={graphData} onNodeClick={(n) => openModal(n.label, n.fullContent)} highlightedClusterIds={highlightedClusters} /></div>
            </div>
          )}

          {/* --- STAGE 5: RAG LAB (NOVA VISUALIZAÇÃO) --- */}
          {stage === PipelineStage.QUERY && (
             <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 h-full">
                {/* Coluna Esquerda: Chat & Controls */}
                <div className="lg:col-span-2 flex flex-col space-y-4">
                   <div className="flex-1 bg-slate-50 border border-slate-200 rounded-lg p-4 overflow-y-auto max-h-[500px] min-h-[400px]">
                      {chatHistory.length === 0 && <div className="text-center text-slate-400 mt-20">Inicie uma conversa com seus documentos.</div>}
                      {chatHistory.map((msg, i) => (
                          <div key={i} className={`mb-4 flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}>
                              <div className={`max-w-[80%] rounded-lg p-3 ${msg.role === 'user' ? 'bg-indigo-600 text-white' : 'bg-white border border-slate-200 text-slate-800'}`}>
                                  <p className="text-sm whitespace-pre-wrap">{msg.content}</p>
                              </div>
                          </div>
                      ))}
                      {isRagThinking && <div className="text-xs text-slate-500 animate-pulse ml-2">Pensando (HyDE, Retrieval, Graph Walk)...</div>}
                   </div>
                   
                   <div className="flex gap-2">
                       <input 
                         type="text" 
                         className="flex-1 border border-slate-300 rounded-lg px-4 py-2 focus:ring-2 focus:ring-indigo-500 outline-none"
                         placeholder="Faça uma pergunta complexa..."
                         value={userQuery}
                         onChange={(e) => setUserQuery(e.target.value)}
                         onKeyPress={(e) => e.key === 'Enter' && handleRAGQuery()}
                         disabled={isRagThinking}
                       />
                       <button 
                         onClick={handleRAGQuery}
                         disabled={isRagThinking || !userQuery}
                         className="bg-indigo-600 hover:bg-indigo-700 text-white px-6 py-2 rounded-lg font-bold disabled:opacity-50"
                       >
                         Enviar
                       </button>
                   </div>
                </div>

                {/* Coluna Direita: Logs de Engenharia */}
                <div className="lg:col-span-1 bg-slate-900 rounded-lg p-4 text-slate-200 font-mono text-xs overflow-y-auto max-h-[600px] border border-slate-700">
                    <h3 className="font-bold text-emerald-400 mb-3 border-b border-slate-700 pb-2">RAG Execution Trace</h3>
                    {ragLogs.length === 0 && <p className="opacity-50 italic">Aguardando execução...</p>}
                    <div className="space-y-3">
                        {ragLogs.map((log, i) => (
                            <div key={i} className="border-l-2 border-slate-600 pl-3 py-1">
                                <div className="flex justify-between">
                                    <span className={`font-bold ${
                                        log.step === 'HYDE' ? 'text-purple-400' : 
                                        log.step === 'RETRIEVAL' ? 'text-blue-400' :
                                        log.step === 'CRAG' ? 'text-orange-400' :
                                        log.step === 'GRAPHRAG' ? 'text-pink-400' : 'text-white'
                                    }`}>{log.step}</span>
                                    <span className={log.status === 'success' ? 'text-green-500' : log.status === 'error' ? 'text-red-500' : 'text-yellow-500'}>
                                        {log.status === 'success' ? '✓' : log.status === 'error' ? '✗' : '⟳'}
                                    </span>
                                </div>
                                <p className="opacity-90 mt-1">{log.description}</p>
                                {log.data && (
                                    <pre className="mt-2 bg-black/30 p-2 rounded overflow-x-auto text-[10px] text-slate-400">
                                        {typeof log.data === 'string' ? log.data : JSON.stringify(log.data, null, 2)}
                                    </pre>
                                )}
                            </div>
                        ))}
                    </div>
                </div>
             </div>
          )}

        </div>
      </main>

      <footer className="bg-slate-900 text-slate-400 py-6 mt-12 border-t border-slate-800">
        <div className="max-w-7xl mx-auto px-4 text-center">
          <p className="text-sm">
            Desenvolvido por <strong className="text-slate-200">Prof. Marcelo Claro Laranjeira</strong>
          </p>
          <p className="text-xs mt-1 opacity-70">
            Ferramenta de Auditoria de Modelos RAG & Graph Theory | Qualis A1
          </p>
        </div>
      </footer>

      <FullContentModal 
        isOpen={modalOpen} 
        onClose={() => setModalOpen(false)} 
        title={modalContent.title}
        content={modalContent.text}
      />

      <FullContentModal 
        isOpen={reportOpen} 
        onClose={() => setReportOpen(false)} 
        title="Relatório Técnico Qualis A1"
        content={reportText}
      />
    </div>
  );
};

export default App;
