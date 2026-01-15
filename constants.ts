
// Textos extraídos do livro: "RAG - RETRIEVAL-AUGMENTED GENERATION: Para Leigos"
// Autor: Sandeco Macedo (2025/2026)
// Fonte Teórica para o Framework

export const ACADEMIC_TEXTS = [
  `A Recuperação Aumentada por Geração (RAG) representa uma mudança paradigmática. O cenário ideal para o RAG é aquele onde a verdade é volátil. Pensem em qualquer base de conhecimento que não seja estática. A legislação de um país, por exemplo, está em constante fluxo. O RAG resolve isso com uma elegância impressionante: a 'inteligência' do modelo (sua capacidade de ler e interpretar) é separada do 'conhecimento' (os documentos).`,
  
  `O RAG Simples trata cada pergunta que você faz como se fosse a primeira vez que vocês conversam. Para construir chatbots e assistentes verdadeiramente úteis, precisamos superar essa amnésia. É aqui que entra o RAG com Memória. A ideia é dar ao nosso sistema a capacidade de lembrar do que foi dito antes. Em vez de descartar a conversa após cada resposta, o sistema passa a manter um histórico do diálogo, uma espécie de 'buffer de memória'.`,
  
  `Se o RAG com Memória deu ao nosso sistema a capacidade de lembrar, o Agentic RAG o eleva a um novo patamar: o da autonomia e da tomada de decisão. Em vez de seguir uma sequência fixa de passos (buscar e depois gerar), um Agent RAG se comporta como um pequeno 'cérebro' que decide dinamicamente qual a melhor ação a tomar. Pense no AgenticRAG como um guarda de trânsito cognitivo. Ele não dirige os carros, mas observa o fluxo e decide quem deve seguir.`,
  
  `O GraphRAG troca a nossa tradicional base de dados de vetores por uma estrutura muito mais rica: um Grafo de Conhecimento (Knowledge Graph). Pense em um grafo como um mapa de relacionamentos. Em vez de 'chunks' de texto isolados, temos Nós (Nodes) representando entidades e Arestas (Edges) representando relações. O GraphRAG convida vocês a fazer algo mais inteligente: antes de responder, organizar a biblioteca por temas e conexões.`,
  
  `O HyDE (Hypothetical Document Embedding) parte de uma premissa genial: em vez de usar a pergunta para encontrar uma resposta, que tal se a gente usasse uma resposta ideal (mesmo que falsa) para encontrar uma resposta real? O sistema gera uma resposta hipotética, transforma em vetor e usa isso para buscar no banco. É uma técnica poderosa para melhorar a precisão da recuperação em perguntas complexas.`,
  
  `O RAG Corretivo (CRAG) atua como um rigoroso editor de jornal. Após o Retriever buscar os documentos iniciais, um módulo de avaliador de qualidade entra em ação. Ele analisa os chunks recuperados e atribui uma pontuação de relevância. Se os documentos forem irrelevantes, o sistema pode descartá-los ou fazer uma nova busca na web, garantindo que o LLM não alucine com base em contexto ruim.`,
  
  `Na teoria dos grafos aplicada ao RAG, algoritmos de centralidade como PageRank ou Betweenness ajudam a identificar os nós mais influentes na rede de conhecimento. Um nó com alta centralidade geralmente representa um conceito fundamental (Hub) que conecta diversos tópicos distintos dentro do corpus, servindo como ponto de ancoragem para a navegação semântica multi-hop.`
];

export const COLORS = {
  primary: '#3b82f6', // blue-500
  secondary: '#6366f1', // indigo-500
  success: '#10b981', // emerald-500
  background: '#f8fafc', // slate-50
  surface: '#ffffff',
  text: '#1e293b', // slate-800
  border: '#e2e8f0', // slate-200
};
