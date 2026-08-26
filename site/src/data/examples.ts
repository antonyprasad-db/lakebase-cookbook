export interface ExampleCard {
  category: string;
  tag: string;
  title: string;
  bracket: string;
  description: string;
  href: string;
  status: 'ready' | 'soon';
}

// Order categories appear on the landing page. Categories not listed fall to
// the end. Keep in sync with CATEGORY_POSITIONS in src/lib/docs.ts.
export const CATEGORY_ORDER = [
  'Agents',
  'Developer Experience',
  'Apps',
  'Data',
  'Ingestion',
];

export const EXAMPLES: ExampleCard[] = [
  {
    category: 'Agents',
    tag: 'GraphRAG',
    title: 'Knowledge-graph RAG on Lakebase',
    bracket: 'pgvector seed, recursive-CTE graph traversal',
    description:
      'Augment RAG with a knowledge graph stored in Lakebase. A pgvector semantic seed finds entry nodes, a recursive CTE traverses relationships out to k hops, and a blended score ranks the expanded context — surfacing connections flat retrieval misses.',
    href: '/docs/examples/graphrag/',
    status: 'ready',
  },
  {
    category: 'Agents',
    tag: 'AI Memory',
    title: 'Durable memory for agents',
    bracket: 'Chainlit history + pgvector semantic recall',
    description:
      'Give an agent persistent memory on one Lakebase store: short-term conversation history via Chainlit’s data layer, long-term facts, and semantic recall over them with pgvector — the app owns its schema, so access survives redeploys.',
    href: '/docs/examples/ai-memory/',
    status: 'ready',
  },
  {
    category: 'Developer Experience',
    tag: 'Branching CI/CD',
    title: 'Ship schema changes with confidence',
    bracket: 'branch-per-PR, AI impact review, exactly-once deploy',
    description:
      'A GitHub CI/CD workflow that forks a Lakebase branch off production for every PR touching migrations, runs the changed SQL there, and posts an AI impact review before merge — then applies migrations to production exactly once.',
    href: '/docs/examples/branching-cicd/',
    status: 'ready',
  },
  {
    category: 'Developer Experience',
    tag: 'Agentic Development',
    title: 'Consort — enforced spec-first TDD on Lakebase branches',
    bracket: 'role agents, deterministic gates, honest-green',
    description:
      'An agentic development framework: a Scrum team rendered as role agents runs spec-first, test-driven development where every "green" is a real test on a live Lakebase branch, driven by a deterministic state machine with human-approval gates and immutable tests.',
    href: '/docs/examples/consort/',
    status: 'ready',
  },
  {
    category: 'Developer Experience',
    tag: 'IDE Extension',
    title: 'Paired Git + Lakebase branching in your IDE',
    bracket: 'VS Code / Cursor, branch pairing, credential rotation',
    description:
      'A VS Code / Cursor extension that replaces built-in Git source control with a unified Git + Lakebase provider, pairing every code branch with a Lakebase database branch so code and schema travel together through review, CI/CD, and merge.',
    href: '/docs/examples/lakebase-scm-extension/',
    status: 'ready',
  },
  {
    category: 'Developer Experience',
    tag: 'SCM Engine',
    title: 'The portable Lakebase branching + SCM engine',
    bracket: 'library API + CLIs, paired-branch state machine',
    description:
      'The portable engine behind the SCM extension and Consort: database branching, the paired-branch SCM workflow state machine, credential minting, and schema migration — consumable as a TypeScript library and as CLIs.',
    href: '/docs/examples/lakebase-scm-utils/',
    status: 'ready',
  },
  {
    category: 'Apps',
    tag: 'FastAPI App',
    title: 'FastAPI backend on Lakebase',
    bracket: 'Databricks App, OAuth rotation, scale-to-zero pooling',
    description:
      'A Databricks App that serves a Lakebase synced table through a FastAPI REST API — with automatic OAuth token rotation, a scale-to-zero-aware connection pool, and fully bundle-driven provisioning of the project, endpoint, catalog, and synced table.',
    href: '/docs/examples/lakebase-fastapi/',
    status: 'ready',
  },
  {
    category: 'Apps',
    tag: 'Genie Caching',
    title: 'Semantic caching for Genie Spaces',
    bracket: 'pgvector cache, multi-space routing, feedback loop',
    description:
      'A Databricks App that puts a pgvector semantic cache in front of Genie Spaces, so a reworded question re-executes the SQL that already worked instead of regenerating it — with per-user memory, supervisor routing across Spaces, and a feedback loop that promotes thumbs-up answers into the cache.',
    href: '/docs/examples/lakebase-genie-caching/',
    status: 'ready',
  },
  {
    category: 'Data',
    tag: 'Feature Store',
    title: 'Online feature serving',
    bracket: 'synced tables → model serving',
    description:
      'Serve features to models with low latency by syncing lakehouse tables into Lakebase — no bespoke online store to build or operate.',
    href: '/docs/examples/feature-store/',
    status: 'soon',
  },
  {
    category: 'Data',
    tag: 'Reverse ETL',
    title: 'Lakehouse → operational store',
    bracket: 'managed sync, no custom pipelines',
    description:
      'Push analytical results back into Lakebase for low-latency serving to applications, without building and maintaining reverse-ETL pipelines.',
    href: '/docs/examples/reverse-etl/',
    status: 'soon',
  },
  {
    category: 'Ingestion',
    tag: 'Source from Kafka',
    title: 'Ingest from Kafka w/Real-Time Mode',
    bracket: 'sub-second, pipelines, real-time mode',
    description: 'Ingest data from Kafka and land it in Lakebase with ultra low latency',
    href: '/docs/examples/kafka-rt-to-lakebase/',
    status: 'ready',
  },
  {
    category: 'Ingestion',
    tag: 'Source from Zerobus',
    title: 'Ingest with Zerobus and Synced Tables',
    bracket: 'zerobus, synced tables, near-real-time',
    description:
      'Land data in a Delta table using Zerobus, then automatically sync it to Lakebase with synced tables',
    href: '/docs/examples/zerobus-to-lakebase/',
    status: 'ready',
  }
];
