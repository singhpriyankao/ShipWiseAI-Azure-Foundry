# The components

ShipWise AI is an Orchestrator and three analysts. Two of the analysts are Azure AI Foundry agents, and one is a code pipeline. In every case code does the checking, and the model reads and writes.

## Orchestrator

**Job:** take a PM's plain-English message and decide what to run.

1. A small model turns the message into structured fields: the feature, the targets that were actually stated (latency, budget, accuracy, volume) and the kind of request.
2. Code validates those fields and decides which analysts run. Routing rules live in code, not in a prompt. A vague request gets a clarifying question, and a narrow request runs only the analyst it needs.
3. The analysts run in parallel, with progress streamed to the page from real events.
4. Code applies precedence rules. For example, a feature classified as prohibited never receives a model recommendation.
5. A model writes the reply from the structured results. Code then checks it: the legal finding must be present, dates the model restated are corrected, and invented follow-up questions are removed.

## Regulations analyst (Foundry agent)

**Job:** say whether the feature falls under the EU AI Act and what that means, citing the Act's own text.

- **Model and tools:** `gpt-4o` for the final reasoning, with no tools. Retrieval happens in code before the agent is called. A smaller model (`gpt-4.1-nano`) breaks the feature into search queries.
- **Retrieval:** hybrid keyword and vector search over the Act in Azure AI Search, merged by reciprocal rank fusion. Chunks keep their article, paragraph, annex and recital structure. Article 3 (definitions) and Article 113 (application dates) are always retrieved.
- **Output:** a classification (`prohibited`, `high_risk`, `transparency_obligated` or `no_specific_obligation_identified`), whether the Article 6(3) derogation applies, the date from which the obligation applies, structured citations, and a plain-language explanation.
- **Guardrails, all in code:**
  - Every citation must match a passage that was actually retrieved.
  - A conclusion cannot rest on a recital alone.
  - An answer must be grounded in retrieved text, otherwise it becomes `insufficient_information` instead of a soft "no obligation".
  - Applicability dates are overwritten from the retrieved Article 113 text. A run that cannot pass the guardrails returns `analysis_failed`, which downstream steps treat as a stop, never as a clean result.
  - The "not legal advice" notice is appended by code.

## Competitive analyst (Foundry agent)

**Job:** find comparable products and what they charge and claim, with sources.

- **Model and tools:** `gpt-4o` with Grounding with Bing Search. It must not answer from memory.
- **Output:** a fixed plain-text layout, one block per product, with pricing, accuracy and speed lines, each cited with the search tool's own citations.
- **Guardrails, all in code:** a parser attaches each claim to the page it cited. Code then loads that page and checks that it names the vendor and supports the figure. Claims a page doesn't support are removed, and the result states what was and wasn't verified. A reply that ignores the layout is retried once.

## Model Stack (a code pipeline, not an agent)

**Job:** shortlist models for the feature, with yearly cost, latency and a quality proxy compared to the PM's targets.

It stopped being an agent after earlier versions produced shifting prices, invented accuracy figures and wrong arithmetic. Once every step was fixed code, a fixed workflow was more reliable and cheaper than an agent.

1. **Interpret the feature:** a small model estimates token sizes, whether the reply streams or arrives whole, and any requirement the options can't be checked against. A request volume is only used if the PM actually stated one, and code confirms the quoted words appear in the message.
2. **Prices:** read from each vendor's own pricing pages by a small model, with every price verified against the page by code.
3. **Latency and quality:** from the Artificial Analysis API. Each model and reasoning-effort setting is reported on its own.
4. **Options:** code builds one option per configuration, calculates yearly cost, checks latency and budget against the targets, picks three, and writes the comparisons between them.

Accuracy is never compared to the PM's target, because no public benchmark measures accuracy on their task. Each option carries a general quality percentile, labelled as a proxy.

## Learning loop

When the PM approves an option, the recommendation and its forecast are saved. Later the PM reports what happened in their own words. A model extracts the figures, code verifies each against the exact quote, and nothing is saved until the PM confirms a read-back. Reported results adjust latency and cost forecasts for that company, and a Decision Quality figure counts how many shipped features met the PM's own targets.
