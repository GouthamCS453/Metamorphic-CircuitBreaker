# Diagrams - Metamorphic Circuit Breaker Framework

All diagrams are written in LaTeX using the **TikZ** package.

## Files

| File | Diagram Type | Description |
|------|-------------|-------------|
| `class_diagram.tex` | Class Diagram | UML class diagram with all system classes, attributes, methods, and relationships |
| `usecase_diagram.tex` | Use Case Diagram | Actors, use cases, <<include>>, and <<extend>> relationships |
| `dfd_diagram.tex` | DFD (Level 1) | Data Flow Diagram showing all processes, data stores, and external entities |
| `architecture_diagram.tex` | Architecture Diagram | 5-layer system architecture with component groupings and data flows |

## How to Compile

### Option 1: Local LaTeX (pdflatex)
Install a LaTeX distribution (MiKTeX or TeX Live), then run:

```powershell
cd diagrams
pdflatex class_diagram.tex
pdflatex usecase_diagram.tex
pdflatex dfd_diagram.tex
pdflatex architecture_diagram.tex
```

### Option 2: Overleaf (Online - Recommended)
1. Go to https://www.overleaf.com
2. Create a new project -> "Blank Project"
3. Upload / paste each `.tex` file
4. Click "Compile" (green button)

### Option 3: Use the compile script
Run `compile_all.bat` (Windows) in this folder.

## Required LaTeX Packages
- `tikz`
- `xcolor`
- TikZ libraries: `positioning`, `arrows.meta`, `shapes`, `shapes.multipart`,
  `shapes.geometric`, `fit`, `backgrounds`, `calc`, `decorations.pathreplacing`
