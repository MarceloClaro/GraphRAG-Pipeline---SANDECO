import * as pdfjsLibProxy from 'pdfjs-dist';

// Fix for ESM/CJS interop issues with pdfjs-dist on some CDNs (esm.sh)
const pdfjsLib = (pdfjsLibProxy as any).default || pdfjsLibProxy;

// Configurar o worker do PDF.js
// Usando cdnjs que serve o arquivo estático diretamente, evitando problemas de redirect/MIME do esm.sh para workers
if (pdfjsLib.GlobalWorkerOptions) {
  pdfjsLib.GlobalWorkerOptions.workerSrc = 'https://cdnjs.cloudflare.com/ajax/libs/pdf.js/3.11.174/pdf.worker.min.js';
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
    
    let fullText = '';
    
    // Iterar sobre todas as páginas
    for (let i = 1; i <= pdf.numPages; i++) {
      const page = await pdf.getPage(i);
      const textContent = await page.getTextContent();
      
      const pageText = textContent.items
        .map((item: any) => item.str)
        .join(' ');
        
      fullText += pageText + '\n\n';
    }

    // Limpeza básica do texto
    fullText = fullText
      .replace(/\s+/g, ' ') // Remover espaços excessivos
      .replace(/(\r\n|\n|\r){2,}/g, '\n\n') // Normalizar quebras de parágrafo
      .trim();

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
    } else if (error.message && error.message.includes('fake worker')) {
      errorMessage += ' Erro interno do processador de PDF (Worker). Tente recarregar a página.';
    } else {
      errorMessage += ' Verifique se é um PDF válido.';
    }
    
    throw new Error(errorMessage);
  }
};