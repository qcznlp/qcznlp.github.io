---
layout: post
title: "Comparing Muon, NorMuon and AdamW for Fine-tuning a Dense Retriever"
date: 2026-10-09 00:00:00-0500
description: A controlled comparison of three optimizers for fine-tuning DenseOn, with learning rates chosen by validation loss and three seeds per recipe.
tags: [blog]
categories: [Information Retrieval]
toc: true
related_posts: false
giscus_comments: false
_styles: |
  .qz-fig { --qz-adamw: #2E5E9E; --qz-muon: #C2661A; --qz-normuon: #13826F; margin: 1.6rem 0; }
  html[data-theme='dark'] .qz-fig { --qz-adamw: #86ACE6; --qz-muon: #F0A152; --qz-normuon: #4DC4AE; }
  .qz-c-adamw { --c: var(--qz-adamw); }
  .qz-c-muon { --c: var(--qz-muon); }
  .qz-c-normuon { --c: var(--qz-normuon); }
  .qz-fig svg { display: block; width: 100%; height: auto; margin: 0 auto; }
  .qz-fig svg.qz-narrow { max-width: 560px; }
  .qz-panels { display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: 12px; }
  .qz-fig svg text { font-size: 10.5px; fill: var(--global-text-color-light); }
  .qz-fig .qz-title, .qz-fig .qz-rowlabel { font-weight: 600; font-size: 12px; fill: var(--global-text-color); }
  .qz-fig .qz-rowvalue { font-size: 11px; fill: var(--global-text-color); }
  .qz-fig .qz-rowlabel, .qz-fig .qz-rowvalue { paint-order: stroke; stroke: var(--global-bg-color); stroke-width: 4px; stroke-linejoin: round; }
  .qz-fig .qz-dir { font-size: 10px; }
  .qz-fig .qz-grid { stroke: var(--global-divider-color); stroke-width: 1; }
  .qz-fig .qz-zero, .qz-fig .qz-axis { stroke: var(--global-text-color-light); stroke-width: 1.1; }
  .qz-fig .qz-ci { stroke: var(--c); stroke-width: 3; stroke-linecap: round; }
  .qz-fig .qz-seed { fill: var(--global-bg-color); stroke: var(--c); stroke-width: 1.6; }
  .qz-fig .qz-mean { fill: var(--c); stroke: var(--global-bg-color); stroke-width: 2; }
  .qz-fig .qz-line { fill: none; stroke: var(--c); stroke-width: 2; stroke-linejoin: round; }
  .qz-fig .qz-pt { fill: var(--c); stroke: var(--global-bg-color); stroke-width: 1.4; }
  .qz-fig .qz-sel { fill: none; stroke: var(--c); stroke-width: 1.4; }
  .qz-legend { display: flex; flex-wrap: wrap; justify-content: center; gap: 4px 18px; margin-top: 8px; font-size: 0.85rem; }
  .qz-legend span { display: inline-flex; align-items: center; gap: 7px; }
  .qz-legend i { display: inline-block; width: 18px; height: 0; border-top: 2.5px solid var(--c); }
  .qz-legend i.qz-dot { width: 9px; height: 9px; border: 0; border-radius: 50%; background: var(--c); }
  .qz-fig figcaption { font-size: 0.875rem; line-height: 1.5; color: var(--global-text-color-light); margin-top: 10px; }
---
In my [last post](/blog/2026/instruct-colbert/), I played with late interaction for instruction-following retrieval. This time I want to talk about the other half of every training recipe, the part I usually do not think about very much: the optimizer.

[Muon](https://kellerjordan.github.io/posts/muon/) is everywhere right now. Instead of applying the raw momentum, it orthogonalizes the momentum of each hidden weight matrix with a few Newton–Schulz iterations. In LLM pretraining it has been reported to match AdamW with about half the compute ([Liu et al., 2025](https://arxiv.org/abs/2502.16982)), and [NorMuon](https://arxiv.org/abs/2510.05491) adds neuron-wise normalization on top. It has also reached the retrieval world: in a small ablation on a NanoBEIR subset, the [mxbai-edge-colbert-v0 report](https://arxiv.org/abs/2510.14880) found Muon at its best learning rate slightly ahead of the best AdamW run (0.599 vs. 0.592 nDCG@10), though behind that run at every other learning rate, and concluded that Muon appears to be a strong optimizer for ColBERT training.

That made me curious. Fine-tuning a retriever is a pretty different regime from pretraining an LLM: the starting model is already contrastively trained, batches are small, training is short, and what we really care about is how well the model transfers to domains it has never seen. Whether Muon's advantages carry over to fine-tuning a model that was pretrained with Adam is itself [an open question](https://arxiv.org/abs/2605.10468). So I asked a narrow version of it: if I give AdamW and Muon the same tuning effort, does Muon give me a better retriever?

The short answer is no, at least not in my setup. Muon and NorMuon fit the training data better in every seed I ran, but the retrievers they produced were no better on BEIR (Figure 1). The details are below.

> **Quick links**<br>
> [Models on Hugging Face](https://huggingface.co/qcz/muon-retriever-finetuning) · [Code, configs and all result tables on GitHub](https://github.com/qcznlp/muon-retriever-finetuning)

<figure class="qz-fig">
<svg class="qz-chart qz-narrow" viewBox="0 0 400 176" role="img" aria-label="Seed-paired BEIR differences to AdamW with 95% intervals">
<line class="qz-grid" x1="67.1" x2="67.1" y1="12" y2="130"/>
<text x="67.1" y="145" text-anchor="middle">−0.4</text>
<line class="qz-grid" x1="120.3" x2="120.3" y1="12" y2="130"/>
<text x="120.3" y="145" text-anchor="middle">−0.2</text>
<line class="qz-zero" x1="173.4" x2="173.4" y1="12" y2="130"/>
<text x="173.4" y="145" text-anchor="middle">0</text>
<line class="qz-grid" x1="226.6" x2="226.6" y1="12" y2="130"/>
<text x="226.6" y="145" text-anchor="middle">+0.2</text>
<line class="qz-grid" x1="279.7" x2="279.7" y1="12" y2="130"/>
<text x="279.7" y="145" text-anchor="middle">+0.4</text>
<line class="qz-grid" x1="332.9" x2="332.9" y1="12" y2="130"/>
<text x="332.9" y="145" text-anchor="middle">+0.6</text>
<text class="qz-dir" x="167.4" y="161" text-anchor="end">← AdamW better</text>
<text class="qz-dir" x="179.4" y="161">Muon-class better →</text>
<g class="qz-c-muon">
<text class="qz-rowlabel" x="14" y="24">Muon (5e-4)</text>
<text class="qz-rowvalue" x="386" y="24" text-anchor="end">+0.09 [−0.51, +0.70]</text>
<line class="qz-ci" x1="38.1" x2="359.1" y1="46" y2="46"/>
<circle class="qz-seed" cx="145.8" cy="46" r="4"/>
<circle class="qz-seed" cx="179.5" cy="46" r="4"/>
<circle class="qz-seed" cx="270.6" cy="46" r="4"/>
<circle class="qz-mean" cx="198.6" cy="46" r="6"/>
</g>
<g class="qz-c-normuon">
<text class="qz-rowlabel" x="14" y="82">NorMuon (5e-4)</text>
<text class="qz-rowvalue" x="386" y="82" text-anchor="end">+0.04 [−0.32, +0.40]</text>
<line class="qz-ci" x1="89.5" x2="279.3" y1="104" y2="104"/>
<circle class="qz-seed" cx="140.7" cy="104" r="4"/>
<circle class="qz-seed" cx="201.3" cy="104" r="4"/>
<circle class="qz-seed" cx="211.3" cy="104" r="4"/>
<circle class="qz-mean" cx="184.4" cy="104" r="6"/>
</g>
</svg>
<figcaption><b>Figure 1.</b> 14-task BEIR nDCG@10 (×100) of the selected Muon and NorMuon recipes minus AdamW 5e-5, paired by training seed. Open circles are seeds 42, 2027 and 3407; the filled circle and bar are the mean and its 95% t-interval.</figcaption>
</figure>

# Setup
I fine-tuned [`lightonai/DenseOn-unsupervised`](https://huggingface.co/lightonai/DenseOn-unsupervised) ([Sourty et al., 2026](https://arxiv.org/abs/2607.27178)), a 22-layer ModernBERT-style encoder with 768-dimensional embeddings that has already gone through contrastive pretraining. The training data are 500K queries from [`lightonai/embeddings-fine-tuning`](https://huggingface.co/datasets/lightonai/embeddings-fine-tuning), each with one positive and seven mined hard negatives. I use InfoNCE with temperature 0.02 and no in-batch negatives, one epoch (3,907 steps) at batch size 128, 10% warmup followed by linear decay, and bf16. One run takes about 7.8 hours on four GPUs, whichever optimizer I use.

Muon and NorMuon only update the 88 hidden weight matrices (attention and MLP). As the Muon authors recommend, everything else, here the token-embedding matrix and 45 norm weights, is trained by a regular AdamW running alongside. To keep the comparison clean, this AdamW uses exactly the settings of the best AdamW run (learning rate 5e-5, same weight decay), so the only difference between the optimizers is how the hidden matrices are updated. For Muon and NorMuon I use momentum 0.95 and five Newton–Schulz steps, plus β₂ = 0.95 for NorMuon; weight decay is 0.01 everywhere except the norms.

Each optimizer gets six peak learning rates: 1e-6, 3e-6, 1e-5, 3e-5, 5e-5 and 1e-4 for AdamW, and 1e-4, 2e-4, 3e-4, 5e-4, 1e-3 and 3e-3 for Muon and NorMuon. Two rules decide the rest:
- Learning rates are selected by the contrastive loss on 4,096 held-out queries, never by BEIR.
- Evaluation is nDCG@10 on LightOn's decontaminated versions of 14 BEIR tasks (test queries and documents that also appear in the mGTE pretraining data are removed), on full corpora and with equal task weights. Five of these tasks are *in-domain*, because their source datasets are part of the training mix (FEVER, FiQA-2018, HotpotQA, MS MARCO and NQ); the other nine are *out-of-domain*.

Each selected recipe is then retrained with two more seeds (2027 and 3407), without any retuning.

# Results
Validation loss picks 5e-5 for AdamW and 5e-4 for both Muon and NorMuon (Figure 2). Both Muon-class optimizers reach a lower validation loss than the best AdamW run, but on BEIR their sweeps from 1e-4 to 1e-3 sit at AdamW's level rather than above it.

<figure class="qz-fig">
<div class="qz-panels">
<svg class="qz-chart" viewBox="0 0 330 250" role="img" aria-label="Validation loss against learning rate">
<text class="qz-title" x="46" y="16">Validation loss</text>
<line class="qz-grid" x1="46" x2="320" y1="180.3" y2="180.3"/>
<text x="40" y="183.8" text-anchor="end">0.18</text>
<line class="qz-grid" x1="46" x2="320" y1="140.8" y2="140.8"/>
<text x="40" y="144.3" text-anchor="end">0.19</text>
<line class="qz-grid" x1="46" x2="320" y1="101.2" y2="101.2"/>
<text x="40" y="104.7" text-anchor="end">0.20</text>
<line class="qz-grid" x1="46" x2="320" y1="61.7" y2="61.7"/>
<text x="40" y="65.2" text-anchor="end">0.21</text>
<line class="qz-grid" x1="71.1" x2="71.1" y1="30" y2="212"/>
<text x="71.1" y="226" text-anchor="middle">1e-5</text>
<line class="qz-grid" x1="184.0" x2="184.0" y1="30" y2="212"/>
<text x="184.0" y="226" text-anchor="middle">1e-4</text>
<line class="qz-grid" x1="296.9" x2="296.9" y1="30" y2="212"/>
<text x="296.9" y="226" text-anchor="middle">1e-3</text>
<line class="qz-axis" x1="46" x2="320" y1="212" y2="212"/>
<text x="183.0" y="244" text-anchor="middle">peak learning rate</text>
<g class="qz-c-adamw"><polyline class="qz-line" points="71.1,42.8 124.9,157.8 150.0,180.7 184.0,137.4"/>
<circle class="qz-pt" cx="71.1" cy="42.8" r="3.6"/>
<circle class="qz-pt" cx="124.9" cy="157.8" r="3.6"/>
<circle class="qz-sel" cx="150.0" cy="180.7" r="7.5"/>
<circle class="qz-pt" cx="150.0" cy="180.7" r="3.6"/>
<circle class="qz-pt" cx="184.0" cy="137.4" r="3.6"/>
</g>
<g class="qz-c-muon"><polyline class="qz-line" points="184.0,80.0 218.0,157.0 237.9,187.4 262.9,189.9 296.9,109.9"/>
<circle class="qz-pt" cx="184.0" cy="80.0" r="3.6"/>
<circle class="qz-pt" cx="218.0" cy="157.0" r="3.6"/>
<circle class="qz-pt" cx="237.9" cy="187.4" r="3.6"/>
<circle class="qz-sel" cx="262.9" cy="189.9" r="7.5"/>
<circle class="qz-pt" cx="262.9" cy="189.9" r="3.6"/>
<circle class="qz-pt" cx="296.9" cy="109.9" r="3.6"/>
</g>
<g class="qz-c-normuon"><polyline class="qz-line" points="184.0,72.1 218.0,154.7 237.9,184.3 262.9,194.9 296.9,132.9"/>
<circle class="qz-pt" cx="184.0" cy="72.1" r="3.6"/>
<circle class="qz-pt" cx="218.0" cy="154.7" r="3.6"/>
<circle class="qz-pt" cx="237.9" cy="184.3" r="3.6"/>
<circle class="qz-sel" cx="262.9" cy="194.9" r="7.5"/>
<circle class="qz-pt" cx="262.9" cy="194.9" r="3.6"/>
<circle class="qz-pt" cx="296.9" cy="132.9" r="3.6"/>
</g>
</svg>
<svg class="qz-chart" viewBox="0 0 330 250" role="img" aria-label="BEIR nDCG@10 against learning rate">
<text class="qz-title" x="46" y="16">BEIR nDCG@10</text>
<line class="qz-grid" x1="46" x2="320" y1="212.0" y2="212.0"/>
<text x="40" y="215.5" text-anchor="end">58.8</text>
<line class="qz-grid" x1="46" x2="320" y1="156.0" y2="156.0"/>
<text x="40" y="159.5" text-anchor="end">59.0</text>
<line class="qz-grid" x1="46" x2="320" y1="100.0" y2="100.0"/>
<text x="40" y="103.5" text-anchor="end">59.2</text>
<line class="qz-grid" x1="46" x2="320" y1="44.0" y2="44.0"/>
<text x="40" y="47.5" text-anchor="end">59.4</text>
<line class="qz-grid" x1="71.1" x2="71.1" y1="30" y2="212"/>
<text x="71.1" y="226" text-anchor="middle">1e-5</text>
<line class="qz-grid" x1="184.0" x2="184.0" y1="30" y2="212"/>
<text x="184.0" y="226" text-anchor="middle">1e-4</text>
<line class="qz-grid" x1="296.9" x2="296.9" y1="30" y2="212"/>
<text x="296.9" y="226" text-anchor="middle">1e-3</text>
<line class="qz-axis" x1="46" x2="320" y1="212" y2="212"/>
<text x="183.0" y="244" text-anchor="middle">peak learning rate</text>
<g class="qz-c-adamw"><polyline class="qz-line" points="71.1,194.4 124.9,176.6 150.0,105.0 184.0,174.5"/>
<circle class="qz-pt" cx="71.1" cy="194.4" r="3.6"/>
<circle class="qz-pt" cx="124.9" cy="176.6" r="3.6"/>
<circle class="qz-sel" cx="150.0" cy="105.0" r="7.5"/>
<circle class="qz-pt" cx="150.0" cy="105.0" r="3.6"/>
<circle class="qz-pt" cx="184.0" cy="174.5" r="3.6"/>
</g>
<g class="qz-c-muon"><polyline class="qz-line" points="184.0,145.6 218.0,112.4 237.9,98.4 262.9,134.2 296.9,177.6"/>
<circle class="qz-pt" cx="184.0" cy="145.6" r="3.6"/>
<circle class="qz-pt" cx="218.0" cy="112.4" r="3.6"/>
<circle class="qz-pt" cx="237.9" cy="98.4" r="3.6"/>
<circle class="qz-sel" cx="262.9" cy="134.2" r="7.5"/>
<circle class="qz-pt" cx="262.9" cy="134.2" r="3.6"/>
<circle class="qz-pt" cx="296.9" cy="177.6" r="3.6"/>
</g>
<g class="qz-c-normuon"><polyline class="qz-line" points="184.0,162.5 218.0,105.6 237.9,126.2 262.9,139.5 296.9,150.8"/>
<circle class="qz-pt" cx="184.0" cy="162.5" r="3.6"/>
<circle class="qz-pt" cx="218.0" cy="105.6" r="3.6"/>
<circle class="qz-pt" cx="237.9" cy="126.2" r="3.6"/>
<circle class="qz-sel" cx="262.9" cy="139.5" r="7.5"/>
<circle class="qz-pt" cx="262.9" cy="139.5" r="3.6"/>
<circle class="qz-pt" cx="296.9" cy="150.8" r="3.6"/>
</g>
</svg>
</div>
<div class="qz-legend"><span class="qz-c-adamw"><i></i>AdamW</span><span class="qz-c-muon"><i></i>Muon</span><span class="qz-c-normuon"><i></i>NorMuon</span></div>
<figcaption><b>Figure 2.</b> Learning-rate sweeps at seed 42, each optimizer on its own learning-rate scale. Left: contrastive loss on 4,096 held-out queries, which selects the learning rate (ringed). Right: 14-task BEIR nDCG@10 (×100). Off the plotted range: AdamW at 1e-6 and 3e-6 and every run at 3e-3 (BEIR 55.7 to 58.3).</figcaption>
</figure>

Then I retrained the selected recipes with two more seeds:

| Recipe | Seed 42 | 2027 | 3407 | Mean | vs. AdamW [95% CI] | p |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| AdamW 5e-5 | 59.18 | 59.13 | 59.00 | 59.10 | – | – |
| Muon 5e-4 | 59.08 | 59.15 | 59.37 | 59.20 | +0.09 [−0.51, +0.70] | 0.57 |
| NorMuon 5e-4 | 59.06 | 59.23 | 59.15 | 59.15 | +0.04 [−0.32, +0.40] | 0.67 |

Neither Muon nor NorMuon beats AdamW. Three seeds cannot show that the recipes are equivalent, since the intervals are 0.7 to 1.2 points wide, but they do bound the effect: for NorMuon, the 95% interval excludes an advantage above 0.40, while for Muon, one favorable seed (3407, +0.37) keeps the upper end at 0.70.

# So what does Muon do differently?
That does not mean the two optimizers behave the same. In every seed, the Muon and NorMuon recipes
- end with a lower training loss (0.234 to 0.237, compared with 0.240 for AdamW, averaged over the last 200 steps),
- reach a lower loss on the held-out queries, by 0.002 to 0.005,
- and score higher on average over the five in-domain BEIR tasks, by 0.04 to 0.63.

| Recipe | Train loss (42 / 2027 / 3407) | Validation loss (42 / 2027 / 3407) |
| --- | ---: | ---: |
| AdamW 5e-5 | 0.2395 / 0.2402 / 0.2400 | 0.1799 / 0.1844 / 0.1805 |
| Muon 5e-4 | 0.2349 / 0.2342 / 0.2367 | 0.1776 / 0.1812 / 0.1757 |
| NorMuon 5e-4 | 0.2346 / 0.2341 / 0.2359 | 0.1763 / 0.1802 / 0.1752 |

They also move the hidden weights further away from the pretrained model: a 3.0% relative change, compared with 1.7% for AdamW at seed 42. The nine out-of-domain tasks, however, do not follow (Figure 3). There, the differences range from −0.30 to +0.32 and average −0.06 for Muon and −0.10 for NorMuon.

<figure class="qz-fig">
<svg class="qz-chart qz-narrow" viewBox="0 0 400 290" role="img" aria-label="In-domain versus out-of-domain difference to AdamW, per seed">
<line class="qz-zero" x1="92.0" x2="92.0" y1="14" y2="246"/>
<text x="92.0" y="260" text-anchor="middle">0</text>
<line class="qz-grid" x1="176.0" x2="176.0" y1="14" y2="246"/>
<text x="176.0" y="260" text-anchor="middle">+0.2</text>
<line class="qz-grid" x1="260.0" x2="260.0" y1="14" y2="246"/>
<text x="260.0" y="260" text-anchor="middle">+0.4</text>
<line class="qz-grid" x1="344.0" x2="344.0" y1="14" y2="246"/>
<text x="344.0" y="260" text-anchor="middle">+0.6</text>
<line class="qz-grid" x1="50" x2="386" y1="246.0" y2="246.0"/>
<text x="44" y="249.5" text-anchor="end">−0.4</text>
<line class="qz-grid" x1="50" x2="386" y1="199.6" y2="199.6"/>
<text x="44" y="203.1" text-anchor="end">−0.2</text>
<line class="qz-zero" x1="50" x2="386" y1="153.2" y2="153.2"/>
<text x="44" y="156.7" text-anchor="end">0</text>
<line class="qz-grid" x1="50" x2="386" y1="106.8" y2="106.8"/>
<text x="44" y="110.3" text-anchor="end">+0.2</text>
<line class="qz-grid" x1="50" x2="386" y1="60.4" y2="60.4"/>
<text x="44" y="63.9" text-anchor="end">+0.4</text>
<line class="qz-grid" x1="50" x2="386" y1="14.0" y2="14.0"/>
<text x="44" y="17.5" text-anchor="end">+0.6</text>
<text x="218.0" y="282" text-anchor="middle">in-domain tasks (5): difference to AdamW</text>
<text transform="translate(12 130.0) rotate(-90)" text-anchor="middle">out-of-domain tasks (9)</text>
<g class="qz-c-muon">
<circle class="qz-pt" cx="128.0" cy="201.8" r="5"/>
<circle class="qz-pt" cx="345.9" cy="222.9" r="5"/>
<circle class="qz-pt" cx="283.6" cy="80.0" r="5"/>
</g>
<g class="qz-c-normuon">
<circle class="qz-pt" cx="109.9" cy="203.2" r="5"/>
<circle class="qz-pt" cx="355.6" cy="196.3" r="5"/>
<circle class="qz-pt" cx="176.9" cy="127.8" r="5"/>
</g>
</svg>
<div class="qz-legend"><span class="qz-c-muon"><i class="qz-dot"></i>Muon 5e-4</span><span class="qz-c-normuon"><i class="qz-dot"></i>NorMuon 5e-4</span></div>
<figcaption><b>Figure 3.</b> Difference to AdamW 5e-5 in the same seed on the five in-domain tasks (x) and the nine out-of-domain tasks (y), nDCG@10 ×100, one point per seed. All six points sit right of zero; four of the six sit below zero.</figcaption>
</figure>

# What I take from this
- **No learning rate makes Muon better here.** Even if I pick the learning rate by BEIR itself, which my selection rules forbid, the best seed-42 Muon run (3e-4, 59.21) and the best NorMuon run (2e-4, 59.18) are level with AdamW (59.18). The learning rate matters far more than the optimizer: across the six rates, AdamW's BEIR score at seed 42 ranges from 56.5 to 59.2 and Muon's from 55.7 to 59.2, while switching optimizers at the selected rates moves the three-seed mean by only +0.09 (Muon) or +0.04 (NorMuon).
- **Muon did worse on scientific and biomedical retrieval.** On all four scientific and biomedical tasks, Muon scores below AdamW in every seed: SciFact (−0.4 on average), NFCorpus (−0.4), SCIDOCS (−0.5) and TREC-COVID (−0.5). NorMuon is below AdamW in 10 of these 12 comparisons. Both gain in every seed on ClimateFEVER (+1.4 for Muon, which shares FEVER's Wikipedia corpus), FiQA (+1.1) and NQ (+0.6). Other tasks swing a lot between seeds (Touché-2020 is −1.83, −1.34 and +0.05 for Muon), so I read these numbers as directions rather than precise effect sizes. Whether Muon helps therefore depends on the target: close to the training mix it can, on specialized scientific text it did worse.
- **Held-out loss tunes Muon for fit, not transfer.** At seed 42, raising Muon's learning rate from 1e-4 to 1e-3 lifts its in-domain score from 72.5 to 73.5 but lowers its out-of-domain score at every step, from 51.6 to 50.8; NorMuon shows the same trade-off. The held-out queries come from the training distribution, so the held-out loss rewards in-distribution fit and selects 5e-4, about 0.35 points below the out-of-domain peak at 1e-4 (Muon) or 2e-4 (NorMuon). For AdamW, the selected 5e-5 is also the top of its out-of-domain curve. These are single-seed sweeps and 0.35 points is close to the seed-to-seed spread, but the direction holds across rates for both optimizers. For a general-purpose retriever, I would try Muon at smaller learning rates and check them on out-of-domain data.

So for fine-tuning an already contrastively pretrained retriever at batch size 128, Muon is not a free upgrade. I would not read this as contradicting the mxbai report, which trained late-interaction models in a different pipeline; it just means that the advantage is not automatic.

## My Remaining Questions
- Does Muon help more when the model has more to learn, for example when fine-tuning from a plain MLM checkpoint such as ModernBERT-base instead of an already contrastively trained one?
- Is the setup tilted toward AdamW from the start? Everything before my fine-tuning used Adam-type optimizers: ModernBERT was pretrained with StableAdamW ([Warner et al., 2024](https://arxiv.org/abs/2412.13663)), and DenseOn's contrastive pretraining used AdamW, the default in its [released training script](https://github.com/lightonai/mdenseon-mlateon/blob/b0db47a48f969d825446668b5b17bfc27a359fc1/scripts/pretrain/english_dense.py). Earlier work suggests this matters: [Qu et al. (2026)](https://arxiv.org/abs/2605.10468) find that switching an Adam-pretrained model to Muon for fine-tuning degrades performance because the two optimizers favor structurally different weights, and [Liu, Wang and Zhang (2026)](https://arxiv.org/abs/2605.06654) find that full fine-tuning with the same optimizer as pretraining forgets less. A fairer test would start from an encoder pretrained with Muon.
- Does the picture change with much larger batches?
- Would late-interaction models, like the ones in the mxbai report, behave differently?
- Weight decay is applied in proportion to the learning rate, so at the selected rates Muon's decay on the hidden matrices is ten times AdamW's. I did not separate that effect, and I did not tune momentum, the number of Newton–Schulz steps or NorMuon's β₂ either.

That is it for this one. As always, I would really appreciate feedback, corrections, or pointers to related results, especially if you have seen Muon help (or not help) in your own retrieval training.
