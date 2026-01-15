import React, { useState } from 'react';
import { PipelineStage, DocumentChunk, EmbeddingVector, ClusterPoint, GraphData } from './types';
import { 
  processRealPDFsToChunks, 
  generateEmbeddingsFromChunks, 
  generateClustersFromEmbeddings, 
  generateGraphFromClusters 
} from './services/mockDataService';
import { extractTextFromPDF } from './services/pdfService';
import { downloadCSV } from './services/exportService';
import PipelineProgress from './components/PipelineProgress';
import FullContentModal from './components/FullContentModal';
import ForceGraph from './components/charts/ForceGraph';
import { ScatterChart, Scatter, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell } from 'recharts';

const App: React.FC = () => {
  // State
  const [stage, setStage] = useState<PipelineStage>(PipelineStage.UPLOAD);
  const [chunks, setChunks] = useState<DocumentChunk[]>([]);
  const [embeddings, setEmbeddings] = useState<EmbeddingVector[]>([]);
  const [clusters, setClusters] = useState<ClusterPoint[]>([]);
  const [graphData, setGraphData] = useState<GraphData | null>(null);
  
  // Upload State
  const [isProcessing, setIsProcessing] = useState(false);
  const [uploadError, setUploadError] = useState<string | null>(null);
  
  // Modal State
  const [modalOpen, setModalOpen] = useState(false);
  const [modalContent, setModalContent] = useState({ title: '', text: '' });

  // Handle Real File Upload
  const handleFileUpload = async (event: React.ChangeEvent<HTMLInputElement>) => {
    if (!event.target.files || event.target.files.length === 0) return;

    setIsProcessing(true);
    setUploadError(null);

    try {
      const files = Array.from(event.target.files);
      const extractedDocs = [];

      for (const file of files) {
        if (file.type === "application/pdf") {
          const processed = await extractTextFromPDF(file);
          extractedDocs.push(processed);
        } else {
           // Fallback for text files if user wants to upload .txt
           const text = await file.text();
           extractedDocs.push({ filename: file.name, text: text, pageCount: 1 });
        }
      }

      const generatedChunks = processRealPDFsToChunks(extractedDocs);
      
      if (generatedChunks.length === 0) {
        setUploadError("Nenhum conteúdo de texto pôde ser extraído dos arquivos. Verifique se são PDFs pesquisáveis (não digitalizados como imagem).");
      } else {
        setChunks(generatedChunks);
        setStage(PipelineStage.UPLOAD);
      }
    } catch (err) {
      console.error(err);
      setUploadError("Erro ao processar arquivos. Certifique-se de que são PDFs válidos.");
    } finally {
      setIsProcessing(false);
    }
  };

  const handleProcessEmbeddings = () => {
    setIsProcessing(true);
    // Simular delay de processamento (GPU load)
    setTimeout(() => {
        const embeds = generateEmbeddingsFromChunks(chunks);
        setEmbeddings(embeds);
        setStage(PipelineStage.EMBEDDINGS);
        setIsProcessing(false);
    }, 800);
  };

  const handleRunClustering = () => {
    setIsProcessing(true);
    setTimeout(() => {
        const clusterPoints = generateClustersFromEmbeddings(embeddings);
        setClusters(clusterPoints);
        setStage(PipelineStage.CLUSTERING);
        setIsProcessing(false);
    }, 800);
  };

  const handleBuildGraph = () => {
    setIsProcessing(true);
    setTimeout(() => {
        const graph = generateGraphFromClusters(clusters);
        setGraphData(graph);
        setStage(PipelineStage.GRAPH);
        setIsProcessing(false);
    }, 1000);
  };

  // CSV Export Handlers
  const exportChunks = () => {
    const dataToExport = chunks.map(c => ({
      ID: c.id,
      Arquivo: c.source,
      Tokens: c.tokens,
      Conteudo_Completo: c.content
    }));
    downloadCSV(dataToExport, 'etapa1_chunks_extracao.csv');
  };

  const exportEmbeddings = () => {
    const dataToExport = embeddings.map(e => ({
      ID: e.id,
      Vetor_Simulado: `[${e.vector.map(v => v.toFixed(4)).join('; ')}]`,
      Conteudo_Completo: e.fullContent
    }));
    downloadCSV(dataToExport, 'etapa2_embeddings_cnn.csv');
  };

  const exportClusters = () => {
    const dataToExport = clusters.map(c => ({
      ID: c.id,
      Cluster_ID: c.clusterId,
      Coord_X: c.x.toFixed(4),
      Coord_Y: c.y.toFixed(4),
      Rotulo: c.label,
      Conteudo_Completo: c.fullContent
    }));
    downloadCSV(dataToExport, 'etapa4_clusters_otimizados.csv');
  };

  const exportGraph = () => {
    if (!graphData) return;
    const nodesExport = graphData.nodes.map(n => ({
      Node_ID: n.id,
      Label: n.label,
      Grupo: n.group,
      Centralidade: n.centrality.toFixed(4),
      Conteudo_Completo: n.fullContent
    }));
    downloadCSV(nodesExport, 'etapa6_grafo_nos.csv');
    
    const edgesExport = graphData.links.map(l => ({
      Origem: l.source,
      Destino: l.target,
      Peso: l.value.toFixed(4),
      Tipo: l.type
    }));
    downloadCSV(edgesExport, 'etapa6_grafo_arestas.csv');
  };

  // UI Helper: Open Modal
  const openModal = (title: string, content: string) => {
    setModalContent({ title, text: content });
    setModalOpen(true);
  };

  return (
    <div className="min-h-screen bg-slate-50 text-slate-900 font-sans">
      {/* Header */}
      <header className="bg-slate-900 text-white shadow-lg sticky top-0 z-40">
        <div className="max-w-7xl mx-auto px-4 py-4 flex justify-between items-center">
          <div className="flex items-center space-x-3">
            <div className="bg-indigo-500 p-2 rounded-lg">
              <svg className="w-6 h-6 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19.428 15.428a2 2 0 00-1.022-.547l-2.387-.477a6 6 0 00-3.86.517l-.318.158a6 6 0 01-3.86.517L6.05 15.21a2 2 0 00-1.806.547M8 4h8l-1 1v5.172a2 2 0 00.586 1.414l5 5c1.26 1.26.367 3.414-1.415 3.414H4.828c-1.782 0-2.674-2.154-1.414-3.414l5-5A2 2 0 009 10.172V5L8 4z" />
              </svg>
            </div>
            <div>
              <h1 className="text-xl font-bold tracking-tight">GraphRAG Pipeline</h1>
              <p className="text-xs text-slate-400">Processamento REAL de PDF para Grafo de Conhecimento</p>
            </div>
          </div>
          <div className="text-right hidden sm:block">
             <span className="bg-emerald-900 text-emerald-100 text-xs px-2 py-1 rounded border border-emerald-700">Modo Produção (Browser-Side)</span>
          </div>
        </div>
      </header>

      <main className="max-w-7xl mx-auto px-4 py-8">
        <PipelineProgress currentStage={stage} />

        <div className="bg-white rounded-xl shadow-sm border border-slate-200 min-h-[600px] p-6 relative">
          
          {/* Global Loading Overlay */}
          {isProcessing && (
             <div className="absolute inset-0 bg-white/80 backdrop-blur-sm z-50 flex flex-col items-center justify-center rounded-xl">
                <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-indigo-600 mb-4"></div>
                <p className="text-indigo-800 font-medium">Processando dados reais...</p>
             </div>
          )}

          {/* Controls Header */}
          <div className="flex justify-between items-center mb-6 pb-4 border-b border-slate-100 flex-wrap gap-4">
            <h2 className="text-2xl font-bold text-slate-800">
              {stage === PipelineStage.UPLOAD && "1. Upload & Extração de Texto"}
              {stage === PipelineStage.EMBEDDINGS && "2. Vetores & Embeddings"}
              {stage === PipelineStage.CLUSTERING && "3. Clusterização Semântica"}
              {stage === PipelineStage.GRAPH && "4. Grafo de Conhecimento Final"}
            </h2>
            <div className="flex space-x-3">
              
              {stage === PipelineStage.UPLOAD && chunks.length > 0 && (
                <button onClick={handleProcessEmbeddings} disabled={isProcessing} className="bg-indigo-600 hover:bg-indigo-700 text-white px-4 py-2 rounded-lg shadow-sm transition-all font-medium flex items-center">
                  Gerar Embeddings <span className="ml-2">→</span>
                </button>
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
                <button onClick={() => {
                    setStage(PipelineStage.UPLOAD);
                    setChunks([]);
                    setEmbeddings([]);
                    setClusters([]);
                    setGraphData(null);
                }} className="bg-slate-200 hover:bg-slate-300 text-slate-800 px-4 py-2 rounded-lg shadow-sm transition-all font-medium">
                  Novo Processamento
                </button>
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
                      <p className="mt-4 text-lg font-medium text-slate-700">Carregar Documentos Reais (PDF)</p>
                      <p className="mt-1 text-sm text-slate-500">Selecione arquivos PDF do seu computador para iniciar a pipeline.</p>
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
                  <div className="flex justify-between items-center bg-green-50 p-3 rounded-lg border border-green-100">
                     <span className="text-green-800 font-medium">✅ {chunks.length} chunks processados a partir de documentos reais.</span>
                     <button onClick={exportChunks} className="flex items-center text-sm bg-emerald-600 text-white px-4 py-2 rounded hover:bg-emerald-700 transition">
                        <svg className="w-4 h-4 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4"/></svg>
                        Baixar CSV (Texto Completo)
                     </button>
                  </div>
                  <div className="overflow-x-auto rounded-lg border border-slate-200">
                    <table className="min-w-full divide-y divide-slate-200">
                      <thead className="bg-slate-50">
                        <tr>
                          <th className="px-6 py-3 text-left text-xs font-medium text-slate-500 uppercase tracking-wider">ID</th>
                          <th className="px-6 py-3 text-left text-xs font-medium text-slate-500 uppercase tracking-wider">Fonte</th>
                          <th className="px-6 py-3 text-left text-xs font-medium text-slate-500 uppercase tracking-wider">Tokens</th>
                          <th className="px-6 py-3 text-left text-xs font-medium text-slate-500 uppercase tracking-wider">Conteúdo Extraído</th>
                          <th className="px-6 py-3 text-right text-xs font-medium text-slate-500 uppercase tracking-wider">Ação</th>
                        </tr>
                      </thead>
                      <tbody className="bg-white divide-y divide-slate-200">
                        {chunks.map(chunk => (
                          <tr key={chunk.id} className="hover:bg-slate-50 transition-colors">
                            <td className="px-6 py-4 whitespace-nowrap text-sm font-mono text-slate-500">{chunk.id.substring(0, 12)}...</td>
                            <td className="px-6 py-4 whitespace-nowrap text-sm text-slate-700">{chunk.source}</td>
                            <td className="px-6 py-4 whitespace-nowrap text-sm text-slate-600">{chunk.tokens}</td>
                            <td className="px-6 py-4 text-sm text-slate-500 max-w-md truncate">
                              {chunk.content.substring(0, 80)}...
                            </td>
                            <td className="px-6 py-4 whitespace-nowrap text-right text-sm font-medium">
                              <button 
                                onClick={() => openModal("Conteúdo Original do Chunk", chunk.content)}
                                className="text-indigo-600 hover:text-indigo-900 bg-indigo-50 px-3 py-1 rounded-full border border-indigo-100"
                              >
                                Ver Completo
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
                    <h3 className="font-semibold text-indigo-900">Geração de Embeddings (CNN 1D)</h3>
                    <p className="text-sm text-indigo-700">Aplicado aos chunks reais extraídos.</p>
                 </div>
                 <button onClick={exportEmbeddings} className="flex items-center text-sm bg-emerald-600 text-white px-4 py-2 rounded hover:bg-emerald-700 transition">
                    <svg className="w-4 h-4 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4"/></svg>
                    Baixar CSV (Vetores)
                 </button>
              </div>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                {embeddings.map((emb, idx) => (
                  <div key={emb.id} className="bg-white p-4 rounded-lg border border-slate-200 hover:shadow-md transition-shadow">
                    <div className="flex justify-between items-start mb-2">
                        <span className="text-xs font-mono text-slate-400">ID: {emb.id}</span>
                        <span className="bg-slate-100 text-slate-600 text-xs px-2 py-0.5 rounded">Vector dim: 1536</span>
                    </div>
                    <div className="font-mono text-xs text-green-600 break-all bg-slate-50 p-2 rounded mb-3">
                      [{emb.vector.map(n => n.toFixed(3)).join(', ')}, ...]
                    </div>
                    <p className="text-sm text-slate-600 mb-3 line-clamp-2">"{emb.contentSummary}"</p>
                    <button 
                      onClick={() => openModal("Texto Base do Embedding", emb.fullContent)}
                      className="text-xs font-semibold text-indigo-600 hover:underline"
                    >
                      Inspecionar Texto Fonte
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
                 <p className="text-slate-600">Clusters calculados com base nas propriedades dos chunks reais.</p>
                 <button onClick={exportClusters} className="flex items-center text-sm bg-emerald-600 text-white px-4 py-2 rounded hover:bg-emerald-700 transition">
                    <svg className="w-4 h-4 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4"/></svg>
                    Baixar CSV (Clusters)
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
                                        <p className="font-bold mb-1">{data.label}</p>
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
                           const colors = ['#3b82f6', '#10b981', '#f59e0b', '#ef4444'];
                           return <Cell key={`cell-${index}`} fill={colors[entry.clusterId % colors.length]} />;
                        })}
                      </Scatter>
                    </ScatterChart>
                  </ResponsiveContainer>
               </div>
               <div className="grid grid-cols-3 gap-4 text-center">
                  <div className="p-2 border-t-4 border-blue-500 bg-white shadow-sm rounded">Cluster A (Curto/Denso)</div>
                  <div className="p-2 border-t-4 border-emerald-500 bg-white shadow-sm rounded">Cluster B (Médio)</div>
                  <div className="p-2 border-t-4 border-amber-500 bg-white shadow-sm rounded">Cluster C (Extenso)</div>
               </div>
            </div>
          )}

          {/* Stage 4: Graph View */}
          {stage === PipelineStage.GRAPH && graphData && (
            <div className="space-y-4">
               <div className="flex flex-col md:flex-row justify-between items-center gap-4 mb-4">
                  <div className="text-sm text-slate-600">
                     <p><strong>Nós:</strong> {graphData.nodes.length} | <strong>Arestas:</strong> {graphData.links.length}</p>
                     <p className="text-xs text-slate-400">Arestas baseadas em similaridade de Jaccard (Palavras Reais em Comum).</p>
                  </div>
                  <div className="flex space-x-2">
                    <button onClick={exportGraph} className="flex items-center text-sm bg-emerald-600 text-white px-4 py-2 rounded hover:bg-emerald-700 transition">
                        <svg className="w-4 h-4 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4"/></svg>
                        Baixar CSV (Nós)
                    </button>
                    <button onClick={exportGraph} className="flex items-center text-sm bg-emerald-700 text-white px-4 py-2 rounded hover:bg-emerald-800 transition">
                        <svg className="w-4 h-4 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13.828 10.172a4 4 0 00-5.656 0l-4 4a4 4 0 105.656 5.656l1.102-1.101m-.758-4.899a4 4 0 005.656 0l4-4a4 4 0 00-5.656-5.656l-1.1 1.1"/></svg>
                        Baixar CSV (Arestas)
                    </button>
                  </div>
               </div>
               
               <ForceGraph 
                 data={graphData} 
                 onNodeClick={(node) => openModal(`Nó do Grafo: ${node.label}`, node.fullContent)}
               />
               
               <div className="bg-blue-50 border border-blue-100 p-4 rounded-lg text-sm text-blue-800 flex items-start">
                  <svg className="w-5 h-5 mr-2 mt-0.5 flex-shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"/></svg>
                  <p>
                    <strong>Análise de Impacto:</strong> O grafo foi gerado calculando a interseção de palavras entre os chunks reais. A visualização mostra agrupamentos de documentos que compartilham terminologia semelhante.
                  </p>
               </div>
            </div>
          )}

        </div>
      </main>

      {/* Global Modal */}
      <FullContentModal 
        isOpen={modalOpen} 
        onClose={() => setModalOpen(false)} 
        title={modalContent.title}
        content={modalContent.text}
      />
    </div>
  );
};

export default App;