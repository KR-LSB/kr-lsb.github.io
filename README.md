# kr-lsb.github.io

Personal blog powered by [Hugo](https://gohugo.io/) + [PaperMod](https://github.com/adityatelange/hugo-PaperMod) theme, deployed on GitHub Pages.

## Quick Start

### Prerequisites
- [Hugo Extended](https://gohugo.io/installation/) (v0.147.0+)
- Git

### Local Development

```bash
# Clone with submodules (PaperMod theme)
git clone --recurse-submodules https://github.com/KR-LSB/kr-lsb.github.io.git
cd kr-lsb.github.io

# Run local server
hugo server -D

# Open http://localhost:1313
```

### Writing a New Post

```bash
hugo new posts/my-new-post/index.md
```

Edit the generated markdown file, then commit and push — GitHub Actions handles deployment automatically.

## Structure

```
├── hugo.yaml                  # Site configuration
├── content/
│   ├── posts/                 # Blog posts
│   ├── about.md               # About page
│   └── search.md              # Search page
├── static/images/             # Static assets
├── .github/workflows/hugo.yaml # Auto-deploy workflow
└── themes/PaperMod/           # Theme (git submodule)
```

## License

Blog content: CC BY 4.0 · Code snippets: MIT
