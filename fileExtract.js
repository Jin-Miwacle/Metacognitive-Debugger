/* FILEEXTRACT.JS -- pulls plain text out of an uploaded PDF, Word, or
   PowerPoint file, right in the browser, no backend needed. Comes back
   as plain text with blank lines between paragraphs, same shape as the
   "paste your own text" box.

   Needs pdf.js, mammoth.js, and JSZip loaded first (index.html /
   teacher.html). If one didn't load, that file type just won't work --
   the others still will.

   Only reads text already embedded in the file -- a scanned/photographed
   PDF has none to find, and this doesn't do OCR. */

if (window.pdfjsLib) {
  pdfjsLib.GlobalWorkerOptions.workerSrc =
    'https://cdn.jsdelivr.net/npm/pdfjs-dist@4.0.379/build/pdf.worker.min.js';
}

function fileKind(file) {
  const name = file.name.toLowerCase();
  if (name.endsWith('.pdf')) return 'pdf';
  if (name.endsWith('.docx')) return 'docx';
  if (name.endsWith('.pptx')) return 'pptx';
  if (name.endsWith('.txt')) return 'txt';
  return null;
}

async function extractTextFromFile(file) {
  const kind = fileKind(file);
  if (!kind) throw new Error("That file type isn't supported. Use a .pdf, .docx, .pptx, or .txt file.");
  if (kind === 'txt') return (await file.text()).trim();
  if (kind === 'pdf') {
    if (!window.pdfjsLib) throw new Error('The PDF reader did not load. Check your internet connection and try again.');
    return extractPdf(file);
  }
  if (kind === 'docx') {
    if (!window.mammoth) throw new Error('The Word-file reader did not load. Check your internet connection and try again.');
    return extractDocx(file);
  }
  if (kind === 'pptx') {
    if (!window.JSZip) throw new Error('The PowerPoint reader did not load. Check your internet connection and try again.');
    return extractPptx(file);
  }
}

async function extractPdf(file) {
  const buf = await file.arrayBuffer();
  const doc = await pdfjsLib.getDocument({ data: buf }).promise;
  const pages = [];
  for (let i = 1; i <= doc.numPages; i++) {
    const page = await doc.getPage(i);
    const content = await page.getTextContent();
    const text = content.items.map(it => it.str).join(' ').replace(/\s+/g, ' ').trim();
    if (text) pages.push(text);
  }
  if (!pages.length) throw new Error("Couldn't find any text in that PDF. If it's a scan or photo of a page, this won't work.");
  return pages.join('\n\n');
}

async function extractDocx(file) {
  const buf = await file.arrayBuffer();
  const result = await mammoth.extractRawText({ arrayBuffer: buf });
  const paragraphs = result.value.split(/\n+/).map(l => l.trim()).filter(Boolean);
  if (!paragraphs.length) throw new Error("Couldn't find any text in that Word file.");
  return paragraphs.join('\n\n');
}

async function extractPptx(file) {
  const zip = await JSZip.loadAsync(file);
  const slideFiles = Object.keys(zip.files)
    .filter(p => /^ppt\/slides\/slide\d+\.xml$/.test(p))
    .sort((a, b) => (+a.match(/slide(\d+)\.xml/)[1]) - (+b.match(/slide(\d+)\.xml/)[1]));
  const slides = [];
  for (const path of slideFiles) {
    const xml = await zip.files[path].async('text');
    const doc = new DOMParser().parseFromString(xml, 'application/xml');
    const text = [...doc.getElementsByTagName('a:t')].map(n => n.textContent).join(' ').replace(/\s+/g, ' ').trim();
    if (text) slides.push(text);
  }
  if (!slides.length) throw new Error("Couldn't find any text in that PowerPoint file.");
  return slides.join('\n\n');
}
