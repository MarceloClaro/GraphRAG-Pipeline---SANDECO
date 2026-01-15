
import * as pdfjsLibProxy from 'pdfjs-dist';

// Fix for ESM/CJS interop issues with pdfjs-dist on some CDNs (esm.sh)
// We try to find the main library object whether it's on .default or the root proxy
const pdfjsLib = (pdfjsLibProxy as any).default || pdfjsLibProxy;

// Configurar o worker do PDF.js usando CDNJS (estável para workers)
// Isso resolve o erro 'NetworkError' e 'fake worker' causados por falha no carregamento do worker do esm.sh
const workerUrl = 'https://cdnjs.cloudflare.com/ajax/libs/pdf.js/3.11.174/pdf.worker.min.js';

try {
  // Tenta configurar no objeto resolvido
  if (pdfjsLib.GlobalWorkerOptions) {
    pdfjsLib.GlobalWorkerOptions.workerSrc = workerUrl;
  } 
  // Fallback: tenta configurar diretamente no proxy se o default falhou
  else if ((pdfjsLibProxy as any).GlobalWorkerOptions) {
    (pdfjsLibProxy as any).GlobalWorkerOptions.workerSrc = workerUrl;
  }
  // Fallback 2: tenta encontrar no window (caso raro de vazamento global)
  else if ((window as any).pdfjsLib?.GlobalWorkerOptions) {
    (window as any).pdfjsLib.GlobalWorkerOptions.workerSrc = workerUrl;
  }
} catch (e) {
  console.warn("Erro ao configurar GlobalWorkerOptions do PDF.js:", e);
}

export interface ProcessedPDF {
  filename: string;
  text: string;
  pageCount: number;
}

export const extractTextFromPDF = async (file: File): Promise<ProcessedPDF> => {
  try {
    const arrayBuffer = await file.arrayBuffer();
    
    // Carregar documento
    const loadingTask = pdfjsLib.getDocument({ data: arrayBuffer });
    const pdf = await loadingTask.promise;
    
    const pageTexts: string[] = [];
    
    // Iterar sobre todas as páginas
    for (let i = 1; i <= pdf.numPages; i++) {
      const page = await pdf.getPage(i);
      const textContent = await page.getTextContent();
      
      // Une os itens de texto com espaço.
      // Em PDFs, palavras frequentemente são itens separados.
      const pageText = textContent.items
        .map((item: any) => {
            // Alguns PDFs retornam strings vazias para espaçamento
            return item.str || " ";
        })
        .join(' ');
      
      pageTexts.push(pageText);
    }

    // Join pages with double newlines to mark page boundaries clearly
    let fullText = pageTexts.join('\n\n');

    // Limpeza Profunda para normalização, mas preservando conteúdo
    // Remove caracteres de controle estranhos, mas mantém pontuação e acentos
    fullText = fullText
      .replace(/[\x00-\x09\x0B-\x0C\x0E-\x1F]/g, '') // Remove non-printable control chars (exceto \n)
      .replace(/[ \t]+/g, ' ') // Normaliza espaços horizontais
      .trim();

    console.log(`[PDF Extraction] Extracted ${fullText.length} chars from ${file.name}`);

    return {
      filename: file.name,
      text: fullText,
      pageCount: pdf.numPages
    };
  } catch (error: any) {
    console.error(`Erro ao processar PDF ${file.name}:`, error);
    
    let errorMessage = `Falha ao ler o arquivo ${file.name}.`;
    
    if (error.name === 'PasswordException') {
      errorMessage += ' O arquivo está protegido por senha.';
    } else if (error.message && (error.message.includes('fake worker') || error.message.includes('worker') || error.message.includes('Setting up fake worker'))) {
      errorMessage += ' Erro no worker do PDF.js. Tente recarregar a página.';
    } else {
      errorMessage += ' Verifique se é um PDF válido.';
    }
    
    throw new Error(errorMessage);
  }
};
