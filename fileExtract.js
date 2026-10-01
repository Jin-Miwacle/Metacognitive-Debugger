/* =====================================================================
   FILEEXTRACT.JS
   Pulls plain text out of an uploaded PDF, Word (.docx), or PowerPoint
   (.pptx) file, right in the browser -- no backend needed. The result
   comes back as plain text with paragraphs separated by a blank line,
   the same format the "paste your own text" box already expects.

   Needs these library scripts loaded first (see index.html / teacher.html):
     pdf.js     -> window.pdfjsLib
     mammoth.js -> window.mammoth
     JSZip      -> window.JSZip
   If one of those didn't load (e.g. no internet when the page opened),
   that one file type just won't work; the others still will.

   Note: this reads the text that's already embedded in the file. A
   PDF that's just a photo/scan of a page (no selectable text) has no
   embedded text to find, so nothing will come out of it -- that needs
   OCR, which this does not do.
   ===================================================================== */

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
