"""
Day 14 - LLM & Transformers
============================
Scenarios:
  1.  AI Email Assistant
  2.  Game Storyline Generator
  3.  Research Assistant (Paper Summarizer)
  4.  Healthcare Chatbot (BERT-based QA)
  5.  AI Creativity - Poem Generator
  6.  Scriptwriting Assistant
  7.  News Headline Generator
  8.  Student Brainstorming Tool
  9.  GPT-2 vs DistilGPT-2 Evaluation
  10. Educational Tool - Concept Explainer

Note: GPT-2 / BERT model downloads are not required.
      All LLM outputs are simulated to demonstrate the concept.
"""

import textwrap

def separator(title):
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


# ─────────────────────────────────────────────────────────────
# SCENARIO 1: AI Email Assistant
# ─────────────────────────────────────────────────────────────
# A company wants to build an AI assistant that reads incoming
# emails and automatically drafts professional replies.
# The model is fine-tuned on business email datasets.
# Given the subject and body of an email, the LLM generates
# a polite, context-aware reply — saving employees time on
# routine correspondence like meeting requests and status updates.
separator("SCENARIO 1: AI Email Assistant")
print("Task: Auto-generate professional email replies using LLM\n")

emails = [
    {
        "subject": "Meeting Request",
        "body":    "Hi, can we schedule a meeting tomorrow at 3 PM?",
        "reply":   (
            "Hi,\n\n"
            "Thank you for reaching out. I'd be happy to meet tomorrow at 3 PM. "
            "Please send me a calendar invite with the meeting details.\n\n"
            "Best regards"
        )
    },
    {
        "subject": "Project Update",
        "body":    "Please provide an update on the current project status.",
        "reply":   (
            "Hi,\n\n"
            "Thank you for your inquiry. The project is currently on track. "
            "We have completed 70% of the deliverables and expect to finish by the deadline. "
            "I will send a detailed report by end of day.\n\n"
            "Best regards"
        )
    }
]

for i, email in enumerate(emails, 1):
    print(f"Email {i}:")
    print(f"  Subject : {email['subject']}")
    print(f"  Body    : {email['body']}")
    print(f"  AI Reply:")
    for line in email['reply'].split('\n'):
        print(f"    {line}")
    print()


# ─────────────────────────────────────────────────────────────
# SCENARIO 2: Game Storyline Generator
# ─────────────────────────────────────────────────────────────
# A game development studio wants to use GPT-2 to generate
# unique storylines for their RPG game. Writers give the model
# a starting prompt and the LLM continues the story.
# This speeds up the creative process — writers can generate
# 10 story variations in seconds and pick the best one.
# The model is fine-tuned on fantasy and sci-fi story datasets.
separator("SCENARIO 2: Game Storyline Generator")
print("Task: Generate creative game storylines from a starting prompt\n")

prompts = [
    "In a world where magic and technology collide,",
    "The last dragon awakened from a thousand-year sleep,"
]

storylines = [
    (
        "In a world where magic and technology collide, a young engineer discovers "
        "an ancient spell embedded in a circuit board. As she deciphers the code, "
        "she realizes it holds the key to unlimited power — but at a terrible cost. "
        "Now she must choose between saving her city or preserving the balance of the universe."
    ),
    (
        "The last dragon awakened from a thousand-year sleep, only to find the world "
        "had forgotten the old ways. Cities of steel and glass replaced the ancient forests. "
        "Determined to restore harmony, the dragon forms an unlikely alliance with a "
        "street-smart hacker who can speak the language of both worlds."
    )
]

for prompt, story in zip(prompts, storylines):
    print(f"Prompt: \"{prompt}\"")
    print("Generated Story:")
    print(textwrap.fill(story, 65, initial_indent="  ", subsequent_indent="  "))
    print()


# ─────────────────────────────────────────────────────────────
# SCENARIO 3: Research Assistant (Paper Summarizer)
# ─────────────────────────────────────────────────────────────
# Researchers spend hours reading papers. An AI research assistant
# can read the abstract of a paper and produce a short summary
# plus the key contribution — in seconds.
# This is built using a summarization LLM (like BART or T5)
# fine-tuned on academic paper datasets.
# Useful for literature reviews, grant writing, and staying
# up to date with the latest research in a field.
separator("SCENARIO 3: AI Research Assistant (Paper Summarizer)")
print("Task: Summarize research papers and extract key contributions\n")

papers = [
    {
        "title": "Attention Is All You Need",
        "abstract": (
            "We propose a new simple network architecture, the Transformer, based solely "
            "on attention mechanisms, dispensing with recurrence and convolutions entirely. "
            "Experiments on two machine translation tasks show these models to be superior "
            "in quality while being more parallelizable and requiring significantly less time to train."
        ),
        "summary": (
            "Introduces the Transformer architecture using self-attention instead of RNNs/CNNs. "
            "Achieves state-of-the-art on translation tasks while being faster to train."
        ),
        "key_contribution": "Self-attention mechanism replacing recurrent networks"
    },
    {
        "title": "BERT: Pre-training of Deep Bidirectional Transformers",
        "abstract": (
            "We introduce BERT, which stands for Bidirectional Encoder Representations from "
            "Transformers. BERT is designed to pre-train deep bidirectional representations "
            "from unlabeled text by jointly conditioning on both left and right context."
        ),
        "summary": (
            "BERT uses masked language modeling and next sentence prediction for pre-training. "
            "Fine-tuning achieves state-of-the-art on 11 NLP benchmarks."
        ),
        "key_contribution": "Bidirectional pre-training for language understanding"
    }
]

for paper in papers:
    print(f"Paper: {paper['title']}")
    print(f"Abstract:")
    print(textwrap.fill(paper['abstract'], 65, initial_indent="  ", subsequent_indent="  "))
    print(f"AI Summary: {paper['summary']}")
    print(f"Key Contribution: {paper['key_contribution']}")
    print()


# ─────────────────────────────────────────────────────────────
# SCENARIO 4: Healthcare Chatbot (BERT-based QA)
# ─────────────────────────────────────────────────────────────
# A hospital wants to deploy a chatbot that answers patient
# health queries based on medical documents.
# BERT's extractive QA capability is used here:
#   - Given a context paragraph (from a medical document)
#   - And a patient's question
#   - BERT finds and extracts the exact answer span from the context
# This is different from generative models — BERT doesn't make up
# answers, it only extracts from the provided document.
# Useful for symptom checking, medication info, and FAQs.
separator("SCENARIO 4: Healthcare Chatbot (BERT-based QA)")
print("Task: Answer patient health queries using BERT extractive QA\n")

qa_pairs = [
    {
        "context":  (
            "Diabetes is a chronic disease that occurs when the pancreas does not produce "
            "enough insulin or when the body cannot effectively use the insulin it produces. "
            "Common symptoms include frequent urination, excessive thirst, and blurred vision."
        ),
        "question": "What are the symptoms of diabetes?",
        "answer":   "Frequent urination, excessive thirst, and blurred vision."
    },
    {
        "context":  (
            "Hypertension, also known as high blood pressure, is a condition where the force "
            "of blood against artery walls is too high. It can be managed through lifestyle "
            "changes such as reducing salt intake, regular exercise, and medication if needed."
        ),
        "question": "How can hypertension be managed?",
        "answer":   "Reducing salt intake, regular exercise, and medication if needed."
    },
    {
        "context":  (
            "COVID-19 is caused by the SARS-CoV-2 virus. Vaccines have been developed to "
            "provide immunity. Common symptoms include fever, cough, and loss of taste or smell."
        ),
        "question": "What causes COVID-19?",
        "answer":   "COVID-19 is caused by the SARS-CoV-2 virus."
    }
]

for i, qa in enumerate(qa_pairs, 1):
    print(f"Query {i}: {qa['question']}")
    print(f"Context:")
    print(textwrap.fill(qa['context'], 65, initial_indent="  ", subsequent_indent="  "))
    print(f"BERT Answer: {qa['answer']}")
    print()


# ─────────────────────────────────────────────────────────────
# SCENARIO 5: AI Creativity - Poem Generator
# ─────────────────────────────────────────────────────────────
# A creative writing platform wants to offer an AI poem generator.
# Users enter a theme (nature, technology, hope) and the LLM
# generates a short poem in that style.
# The model is fine-tuned on a large poetry corpus.
# This shows that LLMs are not just for factual tasks —
# they can also produce creative, artistic content.
separator("SCENARIO 5: AI Creativity - Poem Generator")
print("Task: Generate short poems based on a given theme\n")

themes = ["nature", "technology", "hope"]
poems  = [
    (
        "The river flows through ancient stone,\n"
        "Where silver fish and green moss grow,\n"
        "The wind whispers secrets of its own,\n"
        "In valleys where the wild winds blow."
    ),
    (
        "In circuits deep and silicon dreams,\n"
        "A mind awakens, learns to see,\n"
        "Through data streams and laser beams,\n"
        "A new intelligence runs free."
    ),
    (
        "When darkness falls and shadows creep,\n"
        "A single light begins to glow,\n"
        "Through storms and trials, hearts still keep,\n"
        "The seeds of hope that we must sow."
    )
]

for theme, poem in zip(themes, poems):
    print(f"Theme: \"{theme}\"")
    print("Generated Poem:")
    for line in poem.split('\n'):
        print(f"  {line}")
    print()


# ─────────────────────────────────────────────────────────────
# SCENARIO 6: Scriptwriting Assistant
# ─────────────────────────────────────────────────────────────
# A film production company wants to use an LLM to help writers
# draft dialogue for movie and TV scripts.
# The writer provides the scene setting and characters,
# and the model generates realistic, engaging dialogue.
# This is useful for brainstorming, writer's block, or quickly
# generating first drafts that writers can then refine.
separator("SCENARIO 6: Scriptwriting Assistant")
print("Task: Generate movie/TV script dialogue from scene descriptions\n")

scenes = [
    {
        "setting": "A detective's office, late night",
        "script": [
            ("Detective", "You said you saw him at midnight. Are you sure?"),
            ("Witness",   "Positive. The clock on the wall — it was exactly twelve."),
            ("Detective", "And you didn't think to call the police?"),
            ("Witness",   "I was scared. I didn't know what I'd seen... not yet."),
            ("Detective", "Well, you know now. And that changes everything.")
        ]
    },
    {
        "setting": "A spaceship bridge, emergency alert",
        "script": [
            ("Captain",  "Report! What's our status?"),
            ("Engineer", "Hull breach in sector 7, Captain. We have maybe 20 minutes."),
            ("Captain",  "Can we make it to the nearest station?"),
            ("Engineer", "Not at current speed. But if we reroute power from life support..."),
            ("Captain",  "Do it. We'll deal with the cold. Get us home.")
        ]
    }
]

for scene in scenes:
    print(f"Scene: {scene['setting']}")
    print("Script:")
    for character, line in scene['script']:
        print(f"  {character}: \"{line}\"")
    print()


# ─────────────────────────────────────────────────────────────
# SCENARIO 7: News Headline Generator
# ─────────────────────────────────────────────────────────────
# A news agency wants to auto-generate catchy headlines from
# article summaries. Journalists write the article body,
# and the LLM suggests 3 headline options.
# The model is fine-tuned on news headline datasets.
# This saves time and helps editors pick the most engaging title
# without having to brainstorm from scratch every time.
separator("SCENARIO 7: News Headline Generator")
print("Task: Generate multiple headline options from an article summary\n")

articles = [
    {
        "summary":   "Scientists discover a new species of deep-sea fish with bioluminescent properties near the Mariana Trench.",
        "headlines": [
            "Scientists Discover Glowing Deep-Sea Fish Near Mariana Trench",
            "New Bioluminescent Species Found in Ocean's Deepest Point",
            "Mystery of the Deep: New Fish Species Lights Up the Mariana Trench"
        ]
    },
    {
        "summary":   "A tech company announces breakthrough in quantum computing, achieving 1000-qubit processor.",
        "headlines": [
            "Tech Giant Unveils 1000-Qubit Quantum Processor in Historic Breakthrough",
            "Quantum Computing Milestone: 1000 Qubits Achieved",
            "The Future is Here: New Quantum Chip Could Revolutionize Computing"
        ]
    }
]

for article in articles:
    print(f"Article Summary: {article['summary']}")
    print("Generated Headlines:")
    for i, headline in enumerate(article['headlines'], 1):
        print(f"  {i}. {headline}")
    print()


# ─────────────────────────────────────────────────────────────
# SCENARIO 8: Student Brainstorming Tool
# ─────────────────────────────────────────────────────────────
# Students often struggle to come up with essay topics or research
# angles. An LLM-powered brainstorming tool takes a broad subject
# and generates 5 specific, interesting essay ideas.
# This helps students move past the blank-page problem and gives
# them concrete directions to explore.
# The tool is especially useful for competitive exams, college
# applications, and academic research projects.
separator("SCENARIO 8: Student Brainstorming Tool")
print("Task: Generate essay topic ideas for a given subject area\n")

topics = [
    {
        "subject": "Climate Change",
        "ideas": [
            "The economic impact of transitioning to renewable energy",
            "How youth activism is reshaping climate policy",
            "Carbon capture technology: promise vs. reality",
            "Climate refugees: the hidden human cost",
            "Comparing climate policies across G20 nations"
        ]
    },
    {
        "subject": "Artificial Intelligence in Education",
        "ideas": [
            "Personalized learning through AI tutors",
            "Ethical concerns of AI grading systems",
            "AI vs. human teachers: complementary or competitive?",
            "Detecting plagiarism in the age of ChatGPT",
            "Preparing students for an AI-driven job market"
        ]
    }
]

for topic in topics:
    print(f"Subject: {topic['subject']}")
    print("Brainstormed Essay Ideas:")
    for i, idea in enumerate(topic['ideas'], 1):
        print(f"  {i}. {idea}")
    print()


# ─────────────────────────────────────────────────────────────
# SCENARIO 9: GPT-2 vs DistilGPT-2 Evaluation
# ─────────────────────────────────────────────────────────────
# A startup wants to deploy a text generation feature in their app.
# They need to choose between GPT-2 (full model, 117M parameters)
# and DistilGPT-2 (distilled version, 82M parameters).
# DistilGPT-2 is smaller and faster but slightly less accurate.
# This evaluation compares both models on:
#   - Output quality (text coherence)
#   - Speed (tokens per second)
#   - Model size (MB)
#   - Perplexity (lower = better language model)
# The goal is to find the right balance for production deployment.
separator("SCENARIO 9: GPT-2 vs DistilGPT-2 Evaluation")
print("Task: Compare GPT-2 and DistilGPT-2 on quality, speed, and size\n")

prompt = "The future of artificial intelligence is"
print(f"Prompt: \"{prompt}\"\n")

print("--- GPT-2 Output ---")
gpt2_output = (
    "The future of artificial intelligence is both exciting and uncertain. "
    "As models grow larger and more capable, they begin to tackle problems "
    "once thought impossible — from drug discovery to climate modeling. "
    "Yet questions of safety, bias, and control remain at the forefront of research."
)
print(textwrap.fill(gpt2_output, 65, initial_indent="  ", subsequent_indent="  "))

print("\n--- DistilGPT-2 Output ---")
distilgpt2_output = (
    "The future of artificial intelligence is promising. "
    "New breakthroughs in deep learning continue to push boundaries. "
    "Industries from healthcare to finance are being transformed."
)
print(textwrap.fill(distilgpt2_output, 65, initial_indent="  ", subsequent_indent="  "))

print("\n--- Model Comparison ---")
print(f"\n{'Metric':<32} {'GPT-2':<20} {'DistilGPT-2':<20}")
print("-" * 72)
rows = [
    ("Parameters",              "117M",    "82M"),
    ("Model Size",              "~500MB",  "~350MB"),
    ("Speed (tokens/sec)",      "~45",     "~70"),
    ("Perplexity (lower=better)","29.41",  "32.65"),
    ("Output Quality",          "Higher",  "Slightly Lower"),
    ("Best Use Case",           "Quality", "Speed/Production"),
]
for metric, gpt2_val, distil_val in rows:
    print(f"{metric:<32} {gpt2_val:<20} {distil_val:<20}")

print("\nConclusion: DistilGPT-2 is 40% smaller and ~55% faster than GPT-2,")
print("with only a slight drop in text quality. Ideal for production deployments")
print("where response speed matters more than perfect fluency.")


# ─────────────────────────────────────────────────────────────
# SCENARIO 10: Educational Tool - Concept Explainer
# ─────────────────────────────────────────────────────────────
# An ed-tech platform wants to explain complex AI/ML concepts
# at three different levels: Beginner, Intermediate, Advanced.
# The LLM takes a concept name and a target audience level,
# then generates an explanation tailored to that level.
# Beginners get simple analogies, intermediates get technical
# descriptions, and advanced users get mathematical formulations.
# This makes learning accessible to everyone — from school
# students to PhD researchers.
separator("SCENARIO 10: Educational Tool - Concept Explainer")
print("Task: Explain AI/ML concepts at Beginner, Intermediate, and Advanced levels\n")

concepts = [
    {
        "concept": "Neural Networks",
        "beginner": (
            "A neural network is like a brain made of math. "
            "It learns from examples, just like how you learned to recognize cats "
            "by seeing many pictures of cats. It has layers that process information "
            "step by step until it makes a decision."
        ),
        "intermediate": (
            "Neural networks consist of layers of interconnected nodes (neurons). "
            "Each connection has a weight adjusted during training via backpropagation. "
            "The network learns to map inputs to outputs by minimizing a loss function "
            "using gradient descent optimization."
        ),
        "advanced": (
            "Neural networks are universal function approximators composed of parameterized "
            "linear transformations followed by non-linear activations. Training involves "
            "computing gradients of the loss w.r.t. parameters via automatic differentiation "
            "and updating weights using first-order optimizers like Adam or SGD with momentum."
        )
    },
    {
        "concept": "Attention Mechanism",
        "beginner": (
            "Attention is like a spotlight. When reading a sentence, you focus more on "
            "important words. AI attention works the same way — it learns which parts "
            "of the input to focus on when making predictions."
        ),
        "intermediate": (
            "The attention mechanism computes a weighted sum of values based on the "
            "similarity between queries and keys. Self-attention allows each token to "
            "attend to all other tokens, capturing long-range dependencies in sequences."
        ),
        "advanced": (
            "Multi-head self-attention computes scaled dot-product attention in parallel "
            "across h heads: Attention(Q,K,V) = softmax(QK^T / sqrt(d_k)) * V. "
            "This enables the model to jointly attend to information from different "
            "representation subspaces at different positions in the sequence."
        )
    }
]

for item in concepts:
    print(f"Concept: {item['concept']}")
    print(f"\n  [Beginner Level]")
    print(textwrap.fill(item['beginner'], 65, initial_indent="  ", subsequent_indent="  "))
    print(f"\n  [Intermediate Level]")
    print(textwrap.fill(item['intermediate'], 65, initial_indent="  ", subsequent_indent="  "))
    print(f"\n  [Advanced Level]")
    print(textwrap.fill(item['advanced'], 65, initial_indent="  ", subsequent_indent="  "))
    print()

print("=" * 60)
print("DAY 14 - ALL SCENARIOS COMPLETE")
print("=" * 60)
