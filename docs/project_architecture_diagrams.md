# CausalOMOP Architecture Diagrams

These diagrams reflect the maintained architecture. Green components are
implemented, yellow components require human review, blue components are
external, and dashed components are planned.

## End-to-end architecture

```mermaid
flowchart TB
    Q[Clinical question] --> QG{Causal-question gate}
    QG -->|descriptive| FAST[FastOMOP descriptive workflow]
    QG -->|clarify| USER[Return a clarification request]
    QG -->|causal| DRAFT[Target-trial draft]

    LIT[PubMed + OpenAlex] --> DAG[Evidence-grounded candidate DAG]
    KG[OptimusKG optional] --> DAG
    DRAFT --> DAG
    DAG --> CRIT[Bounded critique and revision]
    CRIT --> ID[Deterministic DAG and backdoor checks]
    ID --> PG{Protocol / DAG human review}

    PG --> MAP[Concept sets + covariate review]
    OMCP[OMCP vocabulary and governed SQL] --> MAP
    MAP --> REG[OMOP variable registry]
    REG --> MG{Clinical mapping gate}
    MG --> FEAS[Feasibility, mapping, cohort, follow-up audits]
    FEAS --> PLAN[Analysis plan + execution readiness]
    PLAN --> APG{Analysis-plan gate}
    APG --> SQL[Person-level analytic-cohort SQL plan]

    SQL -. planned .-> EST[Observed-data estimand + estimators]
    EST -. planned .-> DIAG[Estimator diagnostics + sensitivity]
    DIAG -. planned .-> REPORT[Auditable causal report]

    MIMIC[(MIMIC-IV OMOP\nBigQuery or local DuckDB)] --> OMCP
    FAST --> OMCP
    FEAS --> OMCP

    classDef done fill:#d9ead3,stroke:#38761d,color:#1b1b1b;
    classDef gate fill:#fff2cc,stroke:#bf9000,color:#1b1b1b;
    classDef external fill:#cfe2f3,stroke:#3d85c6,color:#1b1b1b;
    classDef planned fill:#eeeeee,stroke:#777,color:#333,stroke-dasharray:5 5;
    class Q,QG,USER,DRAFT,DAG,CRIT,ID,MAP,REG,FEAS,PLAN,SQL done;
    class PG,MG,APG gate;
    class FAST,LIT,KG,OMCP,MIMIC external;
    class EST,DIAG,REPORT planned;
```

The descriptive branch is currently a redirect: the CLI and UI tell the user
to use FastOMOP but do not automatically transfer the request into the
FastOMOP process.

## Persistent workflow and gates

```mermaid
flowchart LR
    PR[protocol_review] --> P{protocol gate}
    P --> CM[concept_mapping]
    CM --> C{clinical_mapping gate}
    C --> F[feasibility]
    F --> AP[analysis_plan]
    AP --> A{analysis_plan gate}
    A --> E[execution]
    E --> D[diagnostics]
    D --> RR[results_review]
    RR --> R{results gate}
    R --> RP[report]

    C -. provisional exploratory .-> AGG[Aggregate audits only]
    A -. blocked .-> NOPERSON[No authorized person-level execution]

    classDef phase fill:#d9ead3,stroke:#38761d,color:#1b1b1b;
    classDef gate fill:#fff2cc,stroke:#bf9000,color:#1b1b1b;
    classDef block fill:#f4cccc,stroke:#cc0000,color:#1b1b1b;
    class PR,CM,F,AP,E,D,RR,RP,AGG phase;
    class P,C,A,R gate;
    class NOPERSON block;
```

The four current gates are implementation facts, not the final scientific
design. The roadmap calls for separate review of the target trial, DAG,
identification/estimand, and operationalization.

## Trust and data boundary

```mermaid
flowchart LR
    LLM[LLM providers] -->|draft JSON only| ART[Versioned run artifacts]
    RET[PubMed / OpenAlex / OptimusKG] -->|retrieved evidence| ART
    HUMAN[Human reviewer] -->|gate and mapping decisions| ART
    ART --> DET[Deterministic validation and SQL builders]
    DET --> MCP[CausalOMOP OMCP client]
    MCP -->|context + read-only SQL| SERVER[OMCP server]
    SERVER --> DB[(OMOP database)]
    DB -->|aggregates by default| ART

    PII[Patient-level analytic data] -. never sent .-> LLM
```

Raw LLM responses and run artifacts may contain the submitted clinical
question. Patient-level data must never be included in prompts. A generated
person-level cohort query is not authorization to export or share its result.

## Current boundary

CausalOMOP currently reaches governed study design, mapping, feasibility,
aggregate diagnostics, readiness assessment, and analytic-cohort SQL planning.
It does not yet implement treatment-effect estimation or a final causal report.
