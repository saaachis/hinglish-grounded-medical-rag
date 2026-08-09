# Saachi — presentation script (≤ 3:30)

**Your sections:** Abstract → Research gap → System architecture (with example)  
**Pace:** ~130 words/min ≈ **under 460 spoken words** total. Practice with a phone timer.

---

## Flow (one glance)

1. **Intro** — who you are, what you’ll cover, why it matters  
2. **Abstract** — problem + what you built + how you evaluate  
3. **Research gap** — why LLMs alone aren’t enough + why Hinglish needs this stack  
4. **Architecture + example** — walk the poster pipeline once with one query  
5. **Outro** — clean handoff to Devika  

---

## Intro — how you start (~0:20)

**[Optional opening line — say while facing the audience]**  
“Good morning / afternoon — I’m Saachi, from Group 5. Our project is the **Grounded RAG Pipeline for Code Switching in Clinical System**. I’ll frame the problem and walk you through that pipeline once, using the example on the poster.”

**Main talking points**

- Three speakers; you’re **first**: context + **system story**.  
- Full project name: **Grounded RAG Pipeline for Code Switching in Clinical System** (Hinglish clinical queries).  
- Promise: **short** — abstract and gap first, then **one** end-to-end walkthrough.

**[~0:20 → transition]**  
“First, what we’re trying to solve, in one slide.”

### Say this — intro (full script)

Good morning — I’m Saachi, from Group 5. Our project is the **Grounded RAG Pipeline for Code Switching in Clinical System** — we focus on Hinglish clinical queries and evidence-grounded answers. I’m going to start by framing our problem, then I’ll walk you once through that pipeline using the example on the poster. 

---

## Abstract (~0:50–1:00)

**Main talking points (order matters: plain → technical)**

- **Title in plain words:** **Code-switching** = people mix languages (here, **Hinglish**). **Grounded** = the system should answer using **real clinical text it looked up**, not only whatever the model “remembers.”  
- **Context:** **Indian AI healthcare** has to work for how patients and frontline staff **actually** talk — not only formal English.  
- **Mismatch:** Everyday questions in **Hinglish**; trustworthy write-ups often in **English** — so we need a bridge.  
- **What we built (simple):** **Look up** similar cases first → **then** generate the answer; also compare to **skipping** that lookup (same model).  
- **Evaluation (simple):** Is the answer **supported**? Does it **drift** or **hallucinate**? Does it still work when people **mix languages** more or less? — Devika gives the **numbers and tests**.  
- **End with:** “So the gap we’re filling is this.” (~**1:10** cumulative after abstract — adjust if you’re over time.)

### Say this — abstract (full script)

Very briefly, what our **title** means in simple terms. **Code-switching** is when someone blends languages — in our setting, **Hinglish**, Hindi and English together. **Grounded** means we don’t want the system to only improvise from its training; we want it to **tie its answer to real clinical text** — passages it **looked up** from a case library — so there is something concrete behind what it says.

That matters especially for **Indian AI healthcare**, because in real life people don’t always speak textbook English. They describe symptoms the way they talk. Meanwhile, the evidence we trust — case notes, descriptions — is often still in **English**. So our problem is: how do you help someone in **Hinglish** while still **anchoring** the reply to that English evidence?

What we built, in one line: **search first, then answer**. We take the Hinglish question, find the closest relevant English passages in our index, put those in front of the language model, and generate from there. We also run the **same** model on the **same** question **without** that lookup, so we can compare “with evidence” versus “without.”

We then check whether answers are **factually supported**, whether they **drift or hallucinate** away from what we retrieved, and whether things stay stable when people **code-mix** more heavily or more lightly — Devika will walk you through the **exact metrics and statistical tests**. So the gap we’re filling is this.

---

## Research gap (~0:45)

**Main talking points**

- **LLMs alone:** Strong language, but **not inherently tied** to a specific case or citation-like evidence → **hallucination** and **overconfidence** risk in medicine.  
- **Monolingual / English-only tools:** Miss **code-mixed** input; users won’t always switch to formal English.  
- **What’s needed:** **Retrieve** relevant evidence, **condition** generation on it, and **measure** whether the answer actually **uses** that evidence — that’s the **grounding** idea.  
- **Your angle:** **LaBSE**-style cross-lingual alignment + **FAISS** retrieval + **grounded prompting** — the poster’s **center column** is that story.

**[~2:05 cumulative]**  
“I’ll show how that flows on the poster, with one example.”

### Say this — research gap (full script)

Large language models are strong at fluent text, but they are not automatically anchored to a specific patient case or a concrete piece of evidence. 

In medicine, that gap matters: you can get confident-sounding answers that aren’t really supported by anything you can point to. At the same time, English-only tools often fail when the user stays in Hinglish — and we shouldn’t force people to switch languages just to get help. 

What we need is to retrieve the right evidence first, build the prompt around it, and then check whether the answer actually follows from that evidence — that’s what we mean by grounding. 

On the poster, the middle column is that story: cross-lingual encoding with LaBSE, fast similarity search with FAISS plus conditioning so retrieval stays on-domain, and then grounded prompting. I’ll show how that flows in one example.

---

## System architecture + example (~1:20–1:35)

**Flow to follow (point at poster / slide as you go)**

1. **Hinglish query** — patient symptom description (use **one** short line from the poster; don’t read the whole box).  
2. **Encoding + retrieval** — query → **vector**; **similarity search** in the index; mention **conditioning / filtering** in **one phrase** (domain alignment).  
3. **Retrieved evidence** — **English** snippet (one sentence: e.g. rash / eruption wording).  
4. **Grounded prompt** — query + evidence → **Llama** (e.g. via **Groq**); **zero-shot** is same model **without** that evidence.  
5. **Outputs** — **Grounded:** uses evidence-consistent terms; **Zero-shot:** more generic — **scoring** checks factual alignment (Devika will detail tests).

**Main talking points**

- **One path, left to right** — no second example.  
- Emphasize **evidence-first**, not “the model guessed.”  
- **Hand wave** scoring: “We automatically score how well the answer is supported — details next.”

**[If over time — cut to ~0:55]**  
Skip repeating “zero-shot” in full; say: “Same model without evidence — we compare.”

### Say this — architecture + example (full script)

If you look at the center of the poster, we start from a Hinglish query — the patient describes what’s going on, for example a rash that’s spreading. 

We don’t jump straight to the LLM. We first encode that query into a vector using LaBSE, so we’re in a shared semantic space with the English cases. Then we search our FAISS index for the closest clinical passages, with conditioning so we stay aligned with the clinical domain we care about.

What comes back is English evidence — for example a description that matches the kind of eruption we’re seeing. We fold that evidence into the prompt together with the user’s question, and we call Llama through the Groq API. 

For comparison, we also run the same model with the same question but without that retrieved block — that’s our zero-shot path.

You can see the difference on the poster: the grounded answer picks up language that’s consistent with the evidence, while the zero-shot answer stays more generic. 

We then score how well each path is supported by the evidence we actually retrieved. I won’t go into the metrics here — Devika will cover the statistical side next.

---

## Outro — how you pass to Devika (~0:15)

**Closing line (memorize this)**  
“That’s the pipeline and the running example on the poster. Next, Devika will walk through our **methodology**, our **two hypotheses**, and the **statistical results**.”

**[Stop. Step back / gesture to Devika.]**

---

## Time checkpoints (practice)

| Cumulative time | You should be at…        |
|-----------------|---------------------------|
| ~0:20           | End intro                 |
| ~1:10           | End abstract              |
| ~2:05           | End research gap          |
| ~3:20–3:30      | End architecture + example|
| ≤3:30           | Handoff done              |

---

## If you’re running long (save ~30–40 s)

- Shorten **abstract**: keep **one** plain sentence on the title (code-switching + grounded), then **one** sentence on search-then-answer vs without lookup, skip the middle paragraph if needed.  
- **Gap:** Only two bullets — LLM limits + need for cross-lingual retrieval + grounding.  
- **Architecture:** Name **four** stops only: query → retrieve evidence → grounded prompt → grounded vs zero-shot output.

---

*Rehearse out loud 3× with a timer; adjust wording to sound natural in your voice.*

---

## Note on length

The **“Say this”** paragraphs are the full spoken script in order. If you run over **3:30**, shorten the **architecture** block first (say LaBSE → FAISS → evidence → Llama → grounded vs zero-shot in one breath, skip one comparison sentence), then trim one sentence from the **gap** block.
