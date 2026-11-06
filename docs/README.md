# `docs/` - Documentation & Research Papers

## 🎯 Purpose
The `docs/` directory contains research specifications, mathematical formulations, task integration guidelines, development roadmaps, and generated experiment evaluation reports.

---

## 📁 Directory Content
- **[`arch.md`](arch.md)**: System architecture details, theoretical flow, and mathematical formulations for:
  - Multi-route task DAGs
  - Necessity score formulation $\mathcal{N}(t)$
  - Markov state transition modeling $\mathbf{v}_{k+1} = \mathbf{v}_k \mathbf{T}$
  - Stackelberg security game optimization $\max_\theta \mathbb{E}[U_D(\theta)]$
- **[`semeval_integration.md`](semeval_integration.md)**: Mapping psycholinguistic markers (urgency, deception, privilege, exfiltration) to numerical features, and full schema translation for `AgentRisk/agentrisk-data-2027`.
- **[`plan.md`](plan.md)**: Project roadmap formalizing handwritten design notes (Items 1–11), multi-agent execution flows (Researcher $\rightarrow$ Writer $\rightarrow$ Shield), and upcoming milestones.
- **`reports/`**: Destination for evaluation reports, benchmark metrics, and JSON run summaries (`stage_01_baseline_report.json`). Tracked via `.gitkeep`.

