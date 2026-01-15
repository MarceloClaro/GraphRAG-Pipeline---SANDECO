
import React, { useState, useRef } from 'react';
import { PipelineStage, DocumentChunk, EmbeddingVector, ClusterPoint, GraphData, EmbeddingModelType } from './types';
import { 
  processRealPDFsToChunks, 
  generateEmbeddingsFromChunks, 
  generateClustersFromEmbeddings, 
  generateGraphFromClusters 
} from './services/mockDataService';
import { enhanceChunksWithAI, generateRealEmbeddingsWithGemini } from './services/geminiService';
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
  const [embeddingModel, setEmbeddingModel] = useState<EmbeddingModelType>('gemini-004');
  
  // Upload State
  const [isProcessing, setIsProcessing] = useState(false);
  const [processingStatus, setProcessingStatus] = useState("Processando...");
  const [uploadError, setUploadError] = useState<string | null>(null);
  const [aiEnhanced, setAiEnhanced] = useState(false);
  
  // Modal State
  const [modalOpen, setModalOpen] = useState(false);
  const [modalContent, setModalContent] = useState({ title: '', text: '' });

  // Report State
  const [reportOpen, setReportOpen] = useState(false);
  const [reportText, setReportText] = useState('');

  // Handle Real File Upload
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
    setProcessingStatus("Gemini AI: Limpando texto e identificando entidades (isso pode levar um momento)...");
    
    try {
      const enhanced = await enhanceChunksWithAI(chunks, (progress) => {
        setProcessingStatus(`Gemini AI: Processando chunks... ${progress}%`);
      });
      setChunks(enhanced);
      setAiEnhanced(true);
    } catch (err) {
      console.error("Erro na IA", err);
      setUploadError("Falha ao conectar com Gemini API. Verifique sua chave.");
    } finally {
      setIsProcessing(false);
    }
  };

  const handleProcessEmbeddings = async () => {
    setIsProcessing(true);
    setProcessingStatus("Inicializando modelo de embedding...");
    
    // Pequeno delay para UI atualizar
    await new Promise(r => setTimeout(r, 100));

    try {
        let embeds: EmbeddingVector[];
        
        if (embeddingModel === 'gemini-004') {
            setProcessingStatus("Gerando Embeddings Reais via Gemini API (High-Fidelity)...");
            embeds = await generateRealEmbeddingsWithGemini(chunks, (progress) => {
                setProcessingStatus(`Gerando vetores (Gemini-004)... ${progress}%`);
            });
        } else {
            setProcessingStatus(`Gerando vetores simulados (${embeddingModel === 'sentence-bert' ? 'Sentence-BERT' : 'USE'})...`);
            // Simulação estrutural local
            await new Promise(r => setTimeout(r, 1000));
            embeds = generateEmbeddingsFromChunks(chunks, embeddingModel);
        }

        setEmbeddings(embeds);
        setStage(PipelineStage.EMBEDDINGS);
    } catch (e) {
        console.error(e);
        setUploadError("Erro na geração de embeddings.");
    } finally {
        setIsProcessing(false);
    }
  };

  const handleRunClustering = () => {
    setIsProcessing(true);
    setProcessingStatus("Calculando clusters (K-Means & DBSCAN)...");
    setTimeout(() => {
        const clusterPoints = generateClustersFromEmbeddings(embeddings);
        setClusters(clusterPoints);
        setStage(PipelineStage.CLUSTERING);
        setIsProcessing(false);
    }, 800);
  };

  const handleBuildGraph = () => {
    setIsProcessing(true);
    setProcessingStatus("Construindo grafo de conhecimento e calculando métricas...");
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

  // CSV Export Handlers
  const exportChunks = () => {
    const dataToExport = chunks.map(c => ({
      ID: c.id,
      Arquivo: c.source,
      Tipo_Entidade: c.entityType,
      Rotulo_Entidade: c.entityLabel,
      Tokens: c.tokens,
      Palavras_Chave: c.keywords?.join('; '),
      Prazo: c.dueDate,
      Conteudo_Completo: c.content
    }));
    downloadCSV(dataToExport, 'etapa1_entidades_inteligentes.csv');
  };

  const exportEmbeddings = () => {
    const dataToExport = embeddings.map(e => ({
      ID: e.id,
      Tipo_Entidade: e.entityType,
      Modelo: e.modelUsed,
      Dimensao_Vetor: e.vector.length,
      Vetor_Preview: `[${e.vector.slice(0, 5).map(v => v.toFixed(4)).join('; ')}...]`,
      Conteudo_Completo: e.fullContent
    }));
    downloadCSV(dataToExport, 'etapa2_embeddings.csv');
  };

  const exportClusters = () => {
    const dataToExport = clusters.map(c => ({
      ID: c.id,
      Cluster_ID: c.clusterId,
      Rotulo: c.label,
      Tipo_Entidade: c.entityType,
      Conteudo_Completo: c.fullContent
    }));
    downloadCSV(dataToExport, 'etapa4_clusters.csv');
  };

  const exportGraph = () => {
    if (!graphData) return;
    const nodesExport = graphData.nodes.map(n => ({
      Node_ID: n.id,
      Label: n.label,
      Tipo_Entidade: n.entityType,
      Palavras_Chave: n.keywords?.join('; '),
      Centralidade: n.centrality.toFixed(4),
      Conteudo_Completo: n.fullContent
    }));
    downloadCSV(nodesExport, 'etapa6_grafo_nos.csv');
    
    const edgesExport = graphData.links.map(l => ({
      Origem: l.source,
      Destino: l.target,
      Peso: l.value.toFixed(4),
      Confianca: l.confidence.toFixed(4),
      Tipo: l.type
    }));
    downloadCSV(edgesExport, 'etapa6_grafo_arestas.csv');
  };

  const openModal = (title: string, content: string) => {
    setModalContent({ title, text: content });
    setModalOpen(true);
  };

  return (
    <div className="min-h-screen bg-slate-50 text-slate-900 font-sans">
      <header className="bg-slate-900 text-white shadow-lg sticky top-0 z-40">
        <div className="max-w-7xl mx-auto px-4 py-4 flex justify-between items-center">
          <div className="flex items-center space-x-3">
            <div className="bg-indigo-500 p-2 rounded-lg">
              <svg className="w-6 h-6 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
              </svg>
            </div>
            <div>
              <h1 className="text-xl font-bold tracking-tight">GraphRAG Pipeline</h1>
              <p className="text-xs text-slate-400">Powered by Gemini AI</p>
            </div>
          </div>
          <div className="text-right hidden sm:block">
             <span className="bg-gradient-to-r from-indigo-500 to-purple-600 text-white text-xs px-2 py-1 rounded shadow-sm">AI Enhanced Mode</span>
          </div>
        </div>
      </header>

      <main className="max-w-7xl mx-auto px-4 py-8">
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
              {stage === PipelineStage.UPLOAD && "1. Extração & Refinamento AI"}
              {stage === PipelineStage.EMBEDDINGS && "2. Vetores & Embeddings"}
              {stage === PipelineStage.CLUSTERING && "3. Clusterização Semântica"}
              {stage === PipelineStage.GRAPH && "4. Grafo de Conhecimento"}
            </h2>
            <div className="flex items-center space-x-3">
              
              {stage === PipelineStage.UPLOAD && chunks.length > 0 && (
                <>
                  <div className="flex items-center space-x-2 mr-2">
                     <label className="text-xs font-semibold text-slate-600">Modelo:</label>
                     <select 
                        value={embeddingModel}
                        onChange={(e) => setEmbeddingModel(e.target.value as EmbeddingModelType)}
                        className="text-sm border border-slate-300 rounded px-2 py-1 bg-white focus:ring-2 focus:ring-indigo-500"
                     >
                        <option value="gemini-004">Gemini Text-Embedding-004 (Real API)</option>
                        <option value="sentence-bert">Sentence-BERT (Simulado)</option>
                        <option value="use">Universal Sentence Encoder (Simulado)</option>
                     </select>
                  </div>

                  {!aiEnhanced && (
                    <button 
                      onClick={handleEnhanceWithAI} 
                      disabled={isProcessing} 
                      className="bg-gradient-to-r from-purple-600 to-indigo-600 hover:from-purple-700 hover:to-indigo-700 text-white px-4 py-2 rounded-lg shadow-sm transition-all font-medium flex items-center"
                    >
                      <svg className="w-4 h-4 mr-2" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
                      </svg>
                      Limpar & Classificar com Gemini
                    </button>
                  )}
                  <button onClick={handleProcessEmbeddings} disabled={isProcessing} className="bg-slate-800 hover:bg-slate-700 text-white px-4 py-2 rounded-lg shadow-sm transition-all font-medium flex items-center">
                    Gerar Embeddings <span className="ml-2">→</span>
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
                <button onClick={handleGenerateReport} className="bg-amber-600 hover:bg-amber-700 text-white px-4 py-2 rounded-lg shadow-sm transition-all font-medium flex items-center">
                    📄 Relatório Técnico
                </button>
                <button onClick={() => {
                    setStage(PipelineStage.UPLOAD);
                    setChunks([]);
                    setEmbeddings([]);
                    setClusters([]);
                    setGraphData(null);
                    setAiEnhanced(false);
                }} className="bg-slate-200 hover:bg-slate-300 text-slate-800 px-4 py-2 rounded-lg shadow-sm transition-all font-medium">
                  Novo Processamento
                </button>
                </>
              )}
            </div>
          </div>

          {/* Stage 1: Upload & Chunks View */}
          {stage === PipelineStage.UPLOAD && (
            <div className="space-y-4">
              {uploadError && (
                 <div className="bg-red-50 border border-red-200 text-red-700 px-4 py-3 rounded relative" role="alert">
                    <strong className="font-bold">Erro: </strong>
                    <span className="block sm:inline">{uploadError}</span>
                 </div>
              )}

              {chunks.length === 0 ? (
                <div className="flex flex-col items-center justify-center h-80 border-2 border-dashed border-slate-300 rounded-lg bg-slate-50 transition-colors hover:bg-slate-100 hover:border-indigo-400">
                   <div className="text-center">
                      <svg className="mx-auto h-12 w-12 text-slate-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                         <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1} d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
                      </svg>
                      <p className="mt-4 text-lg font-medium text-slate-700">Carregar Documentos PDF</p>
                      <p className="mt-1 text-sm text-slate-500">Pipeline RAG completa com limpeza de texto e classificação via Gemini AI.</p>
                      <label className="mt-6 inline-block cursor-pointer">
                        <span className="bg-indigo-600 hover:bg-indigo-700 text-white px-6 py-3 rounded-lg shadow-md font-medium transition-colors">
                           Selecionar Arquivos
                        </span>
                        <input 
                           type="file" 
                           multiple 
                           accept=".pdf,application/pdf" 
                           className="hidden" 
                           onChange={handleFileUpload}
                        />
                      </label>
                   </div>
                </div>
              ) : (
                <>
                  <div className={`flex justify-between items-center p-3 rounded-lg border ${aiEnhanced ? 'bg-purple-50 border-purple-100' : 'bg-green-50 border-green-100'}`}>
                     <span className={`${aiEnhanced ? 'text-purple-800' : 'text-green-800'} font-medium`}>
                        {aiEnhanced ? `✨ ${chunks.length} entidades enriquecidas e limpas pela IA.` : `✅ ${chunks.length} entidades extraídas (Bruto). Recomenda-se processar com Gemini.`}
                     </span>
                     <button onClick={exportChunks} className="flex items-center text-sm bg-emerald-600 text-white px-4 py-2 rounded hover:bg-emerald-700 transition">
                        <svg className="w-4 h-4 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4"/></svg>
                        CSV
                     </button>
                  </div>
                  <div className="overflow-x-auto rounded-lg border border-slate-200">
                    <table className="min-w-full divide-y divide-slate-200">
                      <thead className="bg-slate-50">
                        <tr>
                          <th className="px-6 py-3 text-left text-xs font-medium text-slate-500 uppercase tracking-wider">Tipo (IA)</th>
                          <th className="px-6 py-3 text-left text-xs font-medium text-slate-500 uppercase tracking-wider">Rótulo</th>
                          <th className="px-6 py-3 text-left text-xs font-medium text-slate-500 uppercase tracking-wider">Palavras-Chave (Entidades)</th>
                          <th className="px-6 py-3 text-left text-xs font-medium text-slate-500 uppercase tracking-wider">Conteúdo (Preview)</th>
                          <th className="px-6 py-3 text-right text-xs font-medium text-slate-500 uppercase tracking-wider">Ação</th>
                        </tr>
                      </thead>
                      <tbody className="bg-white divide-y divide-slate-200">
                        {chunks.map(chunk => (
                          <tr key={chunk.id} className="hover:bg-slate-50 transition-colors">
                            <td className="px-6 py-4 whitespace-nowrap text-sm">
                                <span className={`px-2 py-1 rounded text-xs font-bold ${
                                    chunk.entityType === 'ARTIGO' ? 'bg-blue-100 text-blue-800' :
                                    chunk.entityType === 'DEFINICAO' ? 'bg-teal-100 text-teal-800' :
                                    chunk.entityType === 'ESTRUTURA_MACRO' ? 'bg-purple-100 text-purple-800' :
                                    'bg-slate-100 text-slate-600'
                                }`}>
                                    {chunk.entityType}
                                </span>
                            </td>
                            <td className="px-6 py-4 whitespace-nowrap text-sm font-semibold text-slate-800">{chunk.entityLabel}</td>
                            <td className="px-6 py-4 text-sm text-slate-600 max-w-xs">
                              <div className="flex flex-wrap gap-1">
                                {chunk.keywords?.slice(0, 3).map((k, i) => (
                                    <span key={i} className="px-1.5 py-0.5 bg-yellow-50 text-yellow-700 border border-yellow-100 rounded text-[10px]">{k}</span>
                                ))}
                                {chunk.keywords && chunk.keywords.length > 3 && <span className="text-xs text-slate-400">+{chunk.keywords.length - 3}</span>}
                              </div>
                            </td>
                            <td className="px-6 py-4 text-sm text-slate-500 max-w-xs truncate">
                              {chunk.content.substring(0, 60)}...
                            </td>
                            <td className="px-6 py-4 whitespace-nowrap text-right text-sm font-medium">
                              <button 
                                onClick={() => openModal(`Entidade: ${chunk.entityLabel}`, chunk.content)}
                                className="text-indigo-600 hover:text-indigo-900 bg-indigo-50 px-3 py-1 rounded-full border border-indigo-100"
                              >
                                Ver
                              </button>
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </>
              )}
            </div>
          )}

          {/* Stage 2: Embeddings View */}
          {stage === PipelineStage.EMBEDDINGS && (
            <div className="space-y-4">
              <div className="flex justify-between items-center bg-indigo-50 p-4 rounded-lg border border-indigo-100 mb-4">
                 <div>
                    <h3 className="font-semibold text-indigo-900">Geração de Embeddings ({embeddingModel})</h3>
                    <p className="text-sm text-indigo-700">Vetores gerados com sucesso. Dimensões: {embeddings[0]?.vector.length || 0}.</p>
                 </div>
                 <button onClick={exportEmbeddings} className="flex items-center text-sm bg-emerald-600 text-white px-4 py-2 rounded hover:bg-emerald-700 transition">
                    CSV
                 </button>
              </div>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                {embeddings.map((emb, idx) => (
                  <div key={emb.id} className="bg-white p-4 rounded-lg border border-slate-200 hover:shadow-md transition-shadow">
                    <div className="flex justify-between items-start mb-2">
                        <div>
                            <span className="text-xs font-bold text-slate-600 block">{emb.entityLabel}</span>
                            <span className="text-[10px] uppercase text-slate-400">{emb.entityType}</span>
                        </div>
                        <div className="flex items-center space-x-2">
                           <span className="text-xs text-indigo-600 font-bold border border-indigo-200 px-1 rounded bg-indigo-50">Prazo: {emb.dueDate}</span>
                        </div>
                    </div>
                    <div className="font-mono text-xs text-green-600 break-all bg-slate-50 p-2 rounded mb-3">
                      [{emb.vector.slice(0, 10).map(n => n.toFixed(3)).join(', ')}, ...] ({emb.vector.length} dim)
                    </div>
                    <button 
                      onClick={() => openModal(`Texto Base: ${emb.entityLabel}`, emb.fullContent)}
                      className="text-xs font-semibold text-indigo-600 hover:underline"
                    >
                      Inspecionar Texto Limpo
                    </button>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Stage 3: Clustering View */}
          {stage === PipelineStage.CLUSTERING && (
            <div className="space-y-4">
               <div className="flex justify-between items-center mb-4">
                 <p className="text-slate-600">Clusters calculados com base nas propriedades semânticas.</p>
                 <button onClick={exportClusters} className="flex items-center text-sm bg-emerald-600 text-white px-4 py-2 rounded hover:bg-emerald-700 transition">
                    CSV
                 </button>
               </div>
               
               <div className="h-[500px] w-full bg-slate-50 rounded-lg border border-slate-200 p-4">
                  <ResponsiveContainer width="100%" height="100%">
                    <ScatterChart margin={{ top: 20, right: 20, bottom: 20, left: 20 }}>
                      <CartesianGrid />
                      <XAxis type="number" dataKey="x" name="UMAP X" unit="" stroke="#94a3b8" />
                      <YAxis type="number" dataKey="y" name="UMAP Y" unit="" stroke="#94a3b8" />
                      <Tooltip 
                        cursor={{ strokeDasharray: '3 3' }} 
                        content={({ active, payload }) => {
                            if (active && payload && payload.length) {
                                const data = payload[0].payload as ClusterPoint;
                                return (
                                    <div className="bg-white p-3 border border-slate-200 shadow-lg rounded text-sm max-w-xs">
                                        <p className="font-bold mb-1 text-indigo-700">{data.label}</p>
                                        <p className="text-xs font-semibold text-slate-600 mb-1">{data.entityType}</p>
                                        <div className="flex flex-wrap gap-1 mb-1">
                                            {data.keywords?.slice(0,3).map(k => <span className="text-[9px] bg-slate-100 px-1 rounded">{k}</span>)}
                                        </div>
                                        <p className="text-xs text-slate-500 mb-1">Cluster: {data.clusterId}</p>
                                        <p className="text-xs italic truncate">{data.fullContent.substring(0,50)}...</p>
                                    </div>
                                );
                            }
                            return null;
                        }}
                      />
                      <Scatter name="Documentos" data={clusters} fill="#8884d8">
                        {clusters.map((entry, index) => {
                           const colors = ['#3b82f6', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6', '#ec4899', '#6366f1'];
                           return <Cell key={`cell-${index}`} fill={colors[entry.clusterId % colors.length]} />;
                        })}
                      </Scatter>
                    </ScatterChart>
                  </ResponsiveContainer>
               </div>
            </div>
          )}

          {/* Stage 4: Graph View */}
          {stage === PipelineStage.GRAPH && graphData && (
            <div className="space-y-6">
               <GraphMetricsDashboard metrics={graphData.metrics} />

               <div className="flex justify-end items-center gap-4 mb-2">
                  <div className="flex space-x-2">
                    <button onClick={() => graphRef.current?.downloadGraphImage()} className="flex items-center text-sm bg-indigo-600 text-white px-4 py-2 rounded hover:bg-indigo-700 transition">
                        <svg className="w-4 h-4 mr-2" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" /></svg>
                        Baixar PNG
                    </button>
                    <button onClick={exportGraph} className="flex items-center text-sm bg-emerald-600 text-white px-4 py-2 rounded hover:bg-emerald-700 transition">
                        CSV (Nós)
                    </button>
                    <button onClick={exportGraph} className="flex items-center text-sm bg-emerald-700 text-white px-4 py-2 rounded hover:bg-emerald-800 transition">
                        CSV (Arestas)
                    </button>
                  </div>
               </div>
               
               <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
                  {/* Painel de Análise Lateral */}
                  <div className="lg:col-span-1">
                     <ClusterAnalysisPanel 
                        graphData={graphData} 
                        onClusterSelect={(ids) => setHighlightedClusters(ids)} 
                     />
                  </div>
                  
                  {/* Área do Grafo Principal */}
                  <div className="lg:col-span-3">
                     <ForceGraph 
                       ref={graphRef}
                       data={graphData} 
                       onNodeClick={(node) => openModal(`${node.label} (${node.entityType})`, node.fullContent)}
                       highlightedClusterIds={highlightedClusters}
                     />
                  </div>
               </div>
            </div>
          )}

        </div>
      </main>

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
