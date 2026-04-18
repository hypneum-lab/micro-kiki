// Academic paper template v2 for micro-kiki — pandoc-compatible
// Improvements: cleaner typography, abstract styling, references, code

// Pandoc compatibility shims
#let horizontalrule = line(length: 100%, stroke: 0.5pt + luma(180))
#let blockquote(content) = block(
  inset: (left: 1em, top: 0.6em, bottom: 0.6em, right: 0.5em),
  stroke: (left: 2.5pt + rgb("#4a7bd9")),
  fill: luma(248),
  content
)

// Page + text
#set page(
  paper: "a4",
  margin: (x: 2.2cm, top: 2.5cm, bottom: 2.5cm),
  numbering: "— 1 —",
  number-align: center,
)
#let doc-lang = sys.inputs.at("lang", default: "en")
#set text(font: ("New Computer Modern", "Latin Modern Roman", "Times"), size: 10.5pt, lang: doc-lang)
#set par(justify: true, leading: 0.68em, first-line-indent: 0em)
#set heading(numbering: "1.1")

// Headings
#show heading.where(level: 1): it => {
  pagebreak(weak: true)
  block(above: 1.5em, below: 0.9em,
    text(size: 17pt, weight: "bold", fill: rgb("#1a1a2e"), it)
  )
}
#show heading.where(level: 2): it => block(
  above: 1.3em, below: 0.55em,
  text(size: 12.5pt, weight: "bold", fill: rgb("#2a2a4e"), it)
)
#show heading.where(level: 3): it => block(
  above: 1em, below: 0.35em,
  text(size: 11pt, weight: "bold", style: "italic", fill: rgb("#3a3a5e"), it)
)
#show heading.where(level: 4): it => block(
  above: 0.8em, below: 0.3em,
  text(size: 10.5pt, weight: "bold", style: "italic", it)
)

// Links + emphasis
#show link: set text(fill: rgb("#1e40af"), weight: "regular")
#show emph: it => text(style: "italic", fill: rgb("#2a2a4e"), it.body)
#show strong: it => text(weight: "bold", fill: rgb("#1a1a2e"), it.body)

// Code
#show raw.where(block: true): set block(fill: rgb("#f8f8fc"), inset: 0.85em, radius: 4pt, width: 100%)
#show raw.where(block: true): set text(size: 9pt, font: ("JetBrainsMono NF", "Menlo", "Courier"))
#show raw.where(block: false): set text(size: 9.5pt, font: ("JetBrainsMono NF", "Menlo", "Courier"), fill: rgb("#8a2a2a"))

// Tables
#show table: set text(size: 9.5pt)
#show table.cell.where(y: 0): strong
#set table(stroke: 0.5pt + luma(200), inset: 7pt)

// Lists
#set list(indent: 1em, marker: ([•], [‣], [–]))
#set enum(indent: 1em, numbering: "1.a.i")

// Figure captions
#show figure.caption: set text(size: 9.5pt, style: "italic", fill: luma(80))

