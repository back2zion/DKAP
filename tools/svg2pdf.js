#!/usr/bin/env node
/*
 * svg2pdf.js — convert the paper's figure SVGs to vector PDF.
 *
 * Why this exists: the build environment has no rsvg/inkscape/cairosvg, so the
 * figures are converted with pure-JS svg-to-pdfkit + PDFKit, embedding DejaVu
 * Sans so unicode glyphs (arrows, star, middot) render. Page size is taken from
 * each SVG's own width/height, so there is no hard-coded geometry.
 *
 * Setup (once):   cd tools && npm install        # installs pdfkit, svg-to-pdfkit
 * Convert all:    node svg2pdf.js                # ../paper/Fig1*.svg, Fig2*.svg
 * Convert one:    node svg2pdf.js in.svg [out.pdf]
 *
 * The paper's \includegraphics references are extension-less, so the generated
 * .pdf next to each .svg is picked up automatically by pdflatex over any .png.
 */
const fs = require('fs');
const path = require('path');
const PDFDocument = require('pdfkit');
const SVGtoPDF = require('svg-to-pdfkit');

// DejaVu carries the arrow/star/middot glyphs the standard PDF fonts lack.
const FONT_DIR = '/usr/share/fonts/truetype/dejavu';
const FONTS = {
  R:  path.join(FONT_DIR, 'DejaVuSans.ttf'),
  B:  path.join(FONT_DIR, 'DejaVuSans-Bold.ttf'),
  I:  path.join(FONT_DIR, 'DejaVuSans-Oblique.ttf'),
  BI: path.join(FONT_DIR, 'DejaVuSans-BoldOblique.ttf'),
};
const HAVE_DEJAVU = Object.values(FONTS).every((f) => fs.existsSync(f));

function svgSize(svg) {
  const w = svg.match(/<svg[^>]*\bwidth="([\d.]+)"/);
  const h = svg.match(/<svg[^>]*\bheight="([\d.]+)"/);
  if (w && h) return [parseFloat(w[1]), parseFloat(h[1])];
  const vb = svg.match(/viewBox="[\d.]+\s+[\d.]+\s+([\d.]+)\s+([\d.]+)"/);
  if (vb) return [parseFloat(vb[1]), parseFloat(vb[2])];
  throw new Error('cannot determine SVG size (no width/height or viewBox)');
}

function convert(svgPath, pdfPath) {
  return new Promise((resolve, reject) => {
    const svg = fs.readFileSync(svgPath, 'utf8');
    const [w, h] = svgSize(svg);
    const doc = new PDFDocument({ size: [w, h], margin: 0 });
    let fontCallback;
    if (HAVE_DEJAVU) {
      doc.registerFont('DV', FONTS.R);
      doc.registerFont('DV-B', FONTS.B);
      doc.registerFont('DV-I', FONTS.I);
      doc.registerFont('DV-BI', FONTS.BI);
      fontCallback = (f, bold, italic) =>
        bold && italic ? 'DV-BI' : bold ? 'DV-B' : italic ? 'DV-I' : 'DV';
    } else {
      // Fallback: built-in PDF fonts (no unicode arrows/star — glyphs may drop).
      fontCallback = (f, bold, italic) =>
        bold && italic ? 'Helvetica-BoldOblique' : bold ? 'Helvetica-Bold'
          : italic ? 'Helvetica-Oblique' : 'Helvetica';
    }
    const stream = fs.createWriteStream(pdfPath);
    doc.pipe(stream);
    SVGtoPDF(doc, svg, 0, 0, { width: w, height: h, assumePt: true, fontCallback });
    doc.end();
    stream.on('finish', () => resolve({ pdfPath, w, h }));
    stream.on('error', reject);
  });
}

(async () => {
  if (!HAVE_DEJAVU) {
    console.warn('WARN: DejaVu Sans not found at ' + FONT_DIR +
      ' — falling back to built-in fonts; unicode arrows/star may not render.');
  }
  const args = process.argv.slice(2);
  let jobs;
  if (args.length >= 1) {
    const inp = path.resolve(args[0]);
    const out = path.resolve(args[1] || inp.replace(/\.svg$/i, '.pdf'));
    jobs = [[inp, out]];
  } else {
    const paper = path.resolve(__dirname, '..', 'paper');
    jobs = [
      'Fig1_DKAP_Positioning_Matrix',
      'Fig2_DKAP_Architecture',
    ].map((n) => [path.join(paper, n + '.svg'), path.join(paper, n + '.pdf')]);
  }
  for (const [inp, out] of jobs) {
    const r = await convert(inp, out);
    console.log(`wrote ${path.basename(r.pdfPath)} (${r.w}x${r.h}pt)`);
  }
})().catch((e) => { console.error('ERR', e.message); process.exit(1); });
