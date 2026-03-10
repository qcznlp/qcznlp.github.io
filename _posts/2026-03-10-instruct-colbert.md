---
layout: post
title: Playing with PyLate in Instruction Information Retrieval
date: 2026-03-10 10:00:00-0500
description: Does ColBERT show better results in instruction following information retrieval?
tags: [blog]
categories: [Information Retrieval]
toc: true
related_posts: false
giscus_comments: false
---
Hello world, and welcome to my first blog post. I decided to start blogging because some results, whether exciting, disappointing, or just mildly confusing, are worth sharing in a format that is less intimidating than even a short paper. I will probably make mistakes along the way, but that is part of the point: I want to share what I find, open-source the code when I can, and learn from other people's feedback before everything turns into a polished PDF.

For this first post, I want to talk about instruction-following information retrieval, a very fun research direction introduced by [Orion Weller](https://orionweller.github.io/) and colleagues in [FollowIR](https://arxiv.org/abs/2403.15246). I have been fairly obsessed with IR since early 2025, but most of my hands-on experience has been with bi-encoder retrieval models like [E5](https://arxiv.org/abs/2212.03533) and [Qwen Embedding](https://arxiv.org/abs/2506.05176). Meanwhile, I kept hearing the same thing from [mixedbread](https://huggingface.co/mixedbread-ai/mxbai-edge-colbert-v0-32m), [Antoine Chaffin](https://x.com/antoine_chaffin), and [Omar Khattab](https://omarkhattab.com/): late interaction is really good for IR. Smaller models, stronger retrieval, and surprisingly solid reasoning behavior? That is exactly the kind of claim that makes me want to open a notebook and start poking at ColBERT. So this post is basically my attempt to see how well it works for instruction-following IR, which, as far as I know, still has not been explored very much.

> **Quick links**  
> [Model on Hugging Face](https://huggingface.co/collections/qcz/instruct-ir-exploration) · [Code and configs on GitHub](https://github.com/qcznlp/ins_ir_exploration)

# Some concepts of instruction-following IR
Before getting into the actual training runs, it is worth pausing for a quick explanation of what instruction-following IR actually is. In the usual IR setup, the goal is simple: given a query, retrieve relevant documents. In principle, that query could be arbitrarily complex, but in practice many classic datasets such as [MSMARCO](https://huggingface.co/datasets/microsoft/ms_marco) mostly contain short, fact-seeking queries like `was ronald reagan a democrat`.

Instruction-following IR adds another layer on top of that. The query is no longer just "find documents about this topic"; it also includes extra constraints about what should count as relevant. Those constraints are the *instruction*. In the benchmark [FollowIR](https://arxiv.org/abs/2403.15246), for example, a query might ask about the most and least crashworthy passenger vehicles, while also specifying that relevant documents should compare vehicles rather than talk about drivers. So the retriever has to do more than simple topical matching. It has to understand the user's intent, keep track of the constraints, and retrieve documents that satisfy the instruction rather than just vaguely match the subject. That is what makes this setting interesting and, frankly, a lot harder.

The evaluation setup is a little different too. In addition to standard IR metrics such as nDCG and MRR, FollowIR introduces a metric called *p-MRR*, which is meant to capture how sensitive a retriever is to changes in the instruction. Concretely, each query in FollowIR comes with two instruction variants that can lead to different relevance judgments, and *p-MRR* checks whether the model actually responds to those changes in the right way. There is already [some work](https://arxiv.org/abs/2410.23841) arguing that *p-MRR* may not be the best metric, but for this post I am going to stick with it so we can stay aligned with the original benchmark.

With that out of the way, we can finally get to the fun part: training.

# Question and Training Details
The main question I want to answer is pretty simple: in instruction-following IR, how much do we actually gain by moving from a standard bi-encoder to a late-interaction model? To make the comparison as fair as I could, I kept the backbone fixed and changed only the retrieval paradigm. The main backbone is [`Alibaba-NLP/gte-modernbert-base`](https://huggingface.co/Alibaba-NLP/gte-modernbert-base). I trained the late-interaction version with [PyLate](https://github.com/lightonai/pylate) and the bi-encoder version with [Tevatron](https://github.com/texttron/tevatron/tree/main). I also ran an extra experiment starting from [`lightonai/GTE-ModernColBERT-v1`](https://huggingface.co/lightonai/GTE-ModernColBERT-v1) to see whether prior general-IR training helps once you fine-tune for instruction-following IR.

For training data, I used the dataset from [Promptriever](https://proceedings.iclr.cc/paper_files/paper/2025/file/2cefdb2c4c3274b78cd450bac35228df-Paper-Conference.pdf), which contains roughly 980K examples. I am sparing this post the full hyperparameter dump because, frankly, that is better suited for a config file than a blog paragraph. The complete setup is in the [GitHub repo](https://github.com/qcznlp/ins_ir_exploration) if you want every detail. For evaluation, I follow FollowIR and report the same metrics.

# Results
The main results are shown below. Here, `Score` is the average retrieval score used for the overall comparison, while `p-MRR` reflects how well the model responds to changes in the instruction.

| Model | Initialization | Framework | Score | p-MRR |
| --- | --- | --- | ---: | ---: |
| Bi-encoder | gte-modernbert-base | No training | 34.37 | -1.70 |
| Late interaction | GTE-ModernColBERT-v1 | No training | 32.47 | 1.21 |
| Bi-encoder | gte-modernbert-base | Tevatron | 33.17 | 4.80 |
| Late interaction | gte-modernbert-base | PyLate | 33.90 | 1.50 |
| Late interaction | GTE-ModernColBERT-v1 | PyLate | 41.80 | 3.79 |

At least to me, the most immediate observation is that late interaction does not look like an automatic win. When both models start from `gte-modernbert-base`, the PyLate model is only slightly better than the Tevatron bi-encoder on the overall `Score` (33.90 vs. 33.17), and it is actually worse on `p-MRR` (1.50 vs. 4.80). I do not want to over-read that gap, but it does make me hesitant to tell a simple story like "late interaction is just better."

What feels more interesting is the behavior of `GTE-ModernColBERT-v1`. Even before any additional training, it already has a positive `p-MRR`, while the untuned bi-encoder baseline is negative. After fine-tuning with PyLate, it reaches the best overall `Score` in the table by a large margin, while still keeping reasonably strong instruction sensitivity. If I had to guess, I would say the main story here may be less about late interaction alone and more about the combination of late interaction with a strong ColBERT-style initialization.

One possible explanation is that late-interaction models are simply a bit more data-demanding, or at least more initialization-sensitive, than bi-encoders. That is only a hypothesis for now, but it seems plausible given these results. A generic encoder checkpoint plus late-interaction fine-tuning does not seem to unlock much, whereas a model that already has a stronger late-interaction prior looks much more promising. My impression, for now, is that the extra modeling power may need either better starting points or more task-relevant data before it really shows up.

I should also be careful here: I did not do an especially exhaustive hyperparameter search. These runs are useful as an exploration, but I would not claim they are fully robust. It is very possible that some of the weaker settings, especially the `gte-modernbert-base` late-interaction run, could improve with more careful tuning. So I think these numbers are better read as a first signal than as a definitive comparison.

## My Remaining Questions
I definitely do not think this small experiment settles much. If anything, it mostly left me with a longer list of questions than I had at the beginning. The ones below are the questions I currently find most interesting, and they are also the ones I would want to look at more carefully before making any stronger claim.

- Why is `p-MRR` slightly worse for most of the late-interaction runs? One possibility is that late interaction improves general matching more easily than it improves instruction sensitivity, but I am not confident that this is the right explanation.
- Are late-interaction models more hyperparameter-sensitive in this setting? Since I did not do a very careful sweep, I cannot rule out the possibility that some of these runs are simply under-tuned.
- Are late-interaction models more data-demanding for instruction-following IR? My current guess is maybe yes, but that could easily be confounded with initialization quality.
- Would the gap look different if I trained longer, used a different negative sampling strategy, or changed the distillation setup? I suspect the answer could be yes.

That is all for this first post. I am still figuring this out, so I would genuinely appreciate any feedback, corrections, or suggestions. If you have thoughts on these results, or if you have tried something similar, I would be very happy to hear from you.
