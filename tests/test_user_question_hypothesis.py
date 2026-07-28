import json
import tempfile
import unittest
from pathlib import Path

from causal.agents.user_question import (
    UserQuestionHypothesisService,
    _covariate_search_term,
    _kg_reinforcement,
)
from causal.knowledge_graph.optimuskg_client import KGEdge
from causal.literature.orchestrator import LiteratureOrchestrator
from causal.literature.pubmed import PubMedClient, PubMedConfig
from causal.orchestrator import WorkflowManager


QUESTION = "Does treatment A reduce 28-day mortality compared with treatment B?"

GROUNDED_ESEARCH = b'{"esearchresult": {"idlist": ["111"]}}'
GROUNDED_EFETCH = b"""<?xml version="1.0"?>
<PubmedArticleSet>
  <PubmedArticle>
    <MedlineCitation>
      <PMID>111</PMID>
      <Article>
        <ArticleTitle>Confounding in mortality trials</ArticleTitle>
        <Abstract><AbstractText>Baseline severity is associated with both treatment choice and mortality.</AbstractText></Abstract>
      </Article>
    </MedlineCitation>
  </PubmedArticle>
</PubmedArticleSet>
"""
EMPTY_ESEARCH = b'{"esearchresult": {"idlist": []}}'

AGE_ESEARCH = b'{"esearchresult": {"idlist": ["222"]}}'
AGE_EFETCH = b"""<?xml version="1.0"?>
<PubmedArticleSet>
  <PubmedArticle>
    <MedlineCitation>
      <PMID>222</PMID>
      <Article>
        <ArticleTitle>Age and treatment selection in critical illness</ArticleTitle>
        <Abstract><AbstractText>Older age independently predicts both treatment choice and mortality.</AbstractText></Abstract>
      </Article>
    </MedlineCitation>
  </PubmedArticle>
</PubmedArticleSet>
"""


def fake_pubmed(esearch_json: bytes, efetch_xml: bytes) -> PubMedClient:
    def fetcher(url: str) -> bytes:
        return esearch_json if "esearch.fcgi" in url else efetch_xml

    return PubMedClient(PubMedConfig(), fetcher=fetcher)


def fake_literature(esearch_json: bytes, efetch_xml: bytes) -> LiteratureOrchestrator:
    return LiteratureOrchestrator(pubmed=fake_pubmed(esearch_json, efetch_xml))


class FakeKnowledgeGraph:
    """Duck-typed stand-in for OptimusKGClient: returns canned edges by unordered term pair."""

    def __init__(self, edges_by_pair: dict[frozenset, tuple[KGEdge, ...]]):
        self._edges_by_pair = edges_by_pair

    def find_relations(self, term_a: str, term_b: str) -> tuple[KGEdge, ...]:
        return self._edges_by_pair.get(frozenset({term_a, term_b}), ())


BASELINE_SEVERITY_MORTALITY_EDGE = KGEdge(
    source_id="X1",
    source_label="Baseline severity",
    predicate="associated_with",
    target_id="X2",
    target_label="All-cause mortality",
)

AGE_MORTALITY_EDGE = KGEdge(
    source_id="A1",
    source_label="Age",
    predicate="associated_with",
    target_id="A2",
    target_label="All-cause mortality",
)


def valid_response():
    return {
        "study_id": "treatment_a_mortality",
        "source_question": QUESTION,
        "target_trial": {
            "population": "Eligible adults",
            "treatment": "Treatment A",
            "comparator": "Treatment B",
            "time_zero": "Treatment initiation",
            "grace_period_hours": 0,
            "outcome": "All-cause mortality",
            "follow_up_days": 28,
            "primary_estimand": "ATE",
            "secondary_estimands": [],
        },
        "literature_search": {
            "population_terms": ["critical illness"],
            "treatment_terms": ["treatment A"],
            "comparator_terms": ["treatment B"],
            "outcome_terms": ["mortality", "death"],
        },
        "first_pass_covariates": ["age"],
        "design_constraints": [],
        "assumptions_requiring_review": ["Eligibility is not fully specified"],
    }


class UserQuestionHypothesisTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.prompt = Path(self.temp.name) / "prompt.md"
        self.prompt.write_text("SYSTEM\n{{USER_QUESTION}}", encoding="utf-8")
        self.dag_prompt = Path(self.temp.name) / "dag.md"
        self.dag_prompt.write_text(
            "POPULATION={{POPULATION}}\nTREATMENT={{TREATMENT}}\n"
            "COMPARATOR={{COMPARATOR}}\nOUTCOME={{OUTCOME}}\nARTICLES={{ARTICLES}}\n"
            "KG_RELATIONS={{KG_RELATIONS}}",
            encoding="utf-8",
        )
        self.dag_critique_prompt = Path(self.temp.name) / "dag_critique.md"
        self.dag_critique_prompt.write_text(
            "POPULATION={{POPULATION}}\nTREATMENT={{TREATMENT}}\n"
            "COMPARATOR={{COMPARATOR}}\nOUTCOME={{OUTCOME}}\nARTICLES={{ARTICLES}}\n"
            "KG_RELATIONS={{KG_RELATIONS}}\nPROPOSED_DAG={{PROPOSED_DAG}}\n"
            "STRUCTURAL_FINDINGS={{STRUCTURAL_FINDINGS}}",
            encoding="utf-8",
        )
        self.service = UserQuestionHypothesisService(self.prompt, self.dag_prompt, self.dag_critique_prompt)

    def tearDown(self):
        self.temp.cleanup()

    def test_validated_llm_draft_is_never_executable(self):
        result = self.service.generate(
            QUESTION,
            lambda prompt: json.dumps(valid_response()),
            data_source="project.dataset",
        )
        self.assertFalse(result.hypothesis.executable)
        self.assertEqual(result.protocol["data_source"], "project.dataset")
        self.assertEqual(result.protocol["concept_sets"]["treatment"]["status"], "pending_omop_mapping")
        self.assertIn(QUESTION, self.service.build_prompt(QUESTION))
        self.assertEqual(len(result.llm_outputs), 1)
        self.assertEqual(result.llm_outputs[0].stage, "target_trial_draft")

        state = WorkflowManager(Path(self.temp.name) / "runs").start_generated(
            result, model="openrouter:openai/gpt-oss-120b"
        )
        raw_artifact = json.loads(
            (Path(self.temp.name) / "runs" / state.run_id / "llm_outputs.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertIn("llm_outputs.json", state.artifacts)
        self.assertEqual(raw_artifact["outputs"][0]["raw_text"], json.dumps(valid_response()))
        self.assertTrue(raw_artifact["contains_raw_llm_text"])

    def test_rejects_changed_question(self):
        response = valid_response()
        response["source_question"] = "A different question"
        with self.assertRaises(ValueError):
            self.service.parse_response(QUESTION, json.dumps(response))

    def test_normalizes_mixed_case_study_id_instead_of_rejecting(self):
        response = valid_response()
        response["study_id"] = "Sepsis_ICU_Albumin_vs_Crystalloid_Mortality"
        protocol = self.service.parse_response(QUESTION, json.dumps(response))
        self.assertEqual(protocol["study_id"], "sepsis_icu_albumin_vs_crystalloid_mortality")

    def test_still_rejects_study_id_that_cannot_be_normalized(self):
        response = valid_response()
        response["study_id"] = "123_starts_with_digit"
        with self.assertRaises(ValueError):
            self.service.parse_response(QUESTION, json.dumps(response))

    def test_emits_raw_output_before_validation_failure(self):
        seen = []
        with self.assertRaises(ValueError):
            self.service.generate(
                QUESTION,
                lambda prompt: "not valid JSON",
                on_llm_output=seen.append,
            )
        self.assertEqual(seen[0].stage, "target_trial_draft")
        self.assertEqual(seen[0].raw_text, "not valid JSON")

    def test_rejects_markdown_wrapped_json(self):
        with self.assertRaises(ValueError):
            self.service.parse_response(QUESTION, "```json\n{}\n```")

    def test_builds_short_fielded_pubmed_query(self):
        protocol = self.service.parse_response(QUESTION, json.dumps(valid_response()))

        query = self.service._literature_search_term(protocol)

        self.assertEqual(
            query,
            '("critical illness"[Title/Abstract]) AND '
            '("treatment A"[Title/Abstract]) AND '
            '("treatment B"[Title/Abstract] OR "mortality"[Title/Abstract] OR '
            '"death"[Title/Abstract])',
        )
        self.assertNotIn("Eligible adults", query)

    def test_rejects_pubmed_field_injection(self):
        response = valid_response()
        response["literature_search"]["treatment_terms"] = ['albumin"[All Fields]']
        with self.assertRaisesRegex(ValueError, "Invalid PubMed term"):
            self.service.parse_response(QUESTION, json.dumps(response))

    def test_no_literature_source_skips_dag_stage(self):
        result = self.service.generate(QUESTION, lambda prompt: json.dumps(valid_response()))
        self.assertEqual(result.hypothesis.candidate_dag, ())
        self.assertNotIn("candidate_dag", result.protocol)

    def test_unconfigured_orchestrator_skips_dag_stage(self):
        result = self.service.generate(
            QUESTION, lambda prompt: json.dumps(valid_response()), literature=LiteratureOrchestrator()
        )
        self.assertEqual(result.hypothesis.candidate_dag, ())
        self.assertNotIn("candidate_dag", result.protocol)

    def test_attaches_literature_grounded_dag(self):
        calls = []

        def llm(prompt: str) -> str:
            calls.append(prompt)
            if len(calls) == 1:
                return json.dumps(valid_response())
            if len(calls) == 2:
                return json.dumps({
                    "candidate_dag": [
                        {
                            "label": "Baseline severity",
                            "role": "confounder",
                            "pmid": "111",
                            "temporal_classification": "pre_treatment",
                            "rationale": "Baseline severity is linked to treatment choice and to mortality.",
                        }
                    ],
                    "covariate_relations": [],
                })
            return json.dumps({"verdict": "approve", "concerns": []})

        result = self.service.generate(
            QUESTION, llm, literature=fake_literature(GROUNDED_ESEARCH, GROUNDED_EFETCH)
        )
        self.assertEqual(len(calls), 3)
        self.assertEqual(len(result.protocol["candidate_dag"]), 1)
        self.assertEqual(result.hypothesis.candidate_dag[0].pmid, "111")
        self.assertEqual(result.hypothesis.candidate_dag[0].label, "Baseline severity")
        self.assertEqual([item.stage for item in result.llm_outputs], [
            "target_trial_draft", "literature_dag_draft", "literature_dag_critique"
        ])
        self.assertEqual(
            result.protocol["literature_evidence"]["retrieved_citations"],
            [{"source": "pubmed", "id": "111"}],
        )
        self.assertIn("Title/Abstract", result.protocol["literature_evidence"]["search_query"])

    def test_kg_only_evidence_is_accepted_by_the_gate(self):
        calls = []

        def llm(prompt: str) -> str:
            calls.append(prompt)
            if len(calls) == 1:
                return json.dumps(valid_response())
            if len(calls) == 2:
                self.assertIn(BASELINE_SEVERITY_MORTALITY_EDGE.citation_id, prompt)
                return json.dumps({
                    "candidate_dag": [
                        {
                            "label": "Baseline severity",
                            "role": "confounder",
                            "pmid": BASELINE_SEVERITY_MORTALITY_EDGE.citation_id,
                            "temporal_classification": "pre_treatment",
                            "rationale": "The knowledge graph links baseline severity to mortality.",
                        }
                    ],
                    "covariate_relations": [],
                })
            return json.dumps({"verdict": "approve", "concerns": []})

        kg = FakeKnowledgeGraph({
            frozenset({"Treatment A", "All-cause mortality"}): (BASELINE_SEVERITY_MORTALITY_EDGE,),
        })
        result = self.service.generate(QUESTION, llm, knowledge_graph=kg)
        self.assertEqual(len(calls), 3)
        self.assertEqual(len(result.protocol["candidate_dag"]), 1)
        self.assertEqual(result.hypothesis.candidate_dag[0].pmid, BASELINE_SEVERITY_MORTALITY_EDGE.citation_id)
        self.assertEqual(result.hypothesis.candidate_dag[0].kg_evidence_id, "")
        self.assertEqual(
            result.protocol["literature_evidence"]["retrieved_citations"],
            [{"source": "optimuskg", "id": BASELINE_SEVERITY_MORTALITY_EDGE.citation_id}],
        )

    def test_kg_confirmation_of_a_literature_edge_is_recorded_as_reinforcement(self):
        calls = []

        def llm(prompt: str) -> str:
            calls.append(prompt)
            if len(calls) == 1:
                return json.dumps(valid_response())
            if len(calls) == 2:
                return json.dumps({
                    "candidate_dag": [
                        {
                            "label": "Baseline severity",
                            "role": "confounder",
                            "pmid": "111",
                            "temporal_classification": "pre_treatment",
                            "rationale": "Baseline severity is linked to treatment choice and to mortality.",
                        }
                    ],
                    "covariate_relations": [],
                })
            return json.dumps({"verdict": "approve", "concerns": []})

        kg = FakeKnowledgeGraph({
            frozenset({"Treatment A", "All-cause mortality"}): (BASELINE_SEVERITY_MORTALITY_EDGE,),
        })
        result = self.service.generate(
            QUESTION, llm,
            literature=fake_literature(GROUNDED_ESEARCH, GROUNDED_EFETCH),
            knowledge_graph=kg,
        )
        self.assertEqual(result.hypothesis.candidate_dag[0].pmid, "111")
        self.assertEqual(
            result.hypothesis.candidate_dag[0].kg_evidence_id, BASELINE_SEVERITY_MORTALITY_EDGE.citation_id
        )
        self.assertTrue(
            any("corroborated by both literature and the knowledge graph" in note
                for note in result.protocol["design_constraints"])
        )

    def test_kg_covariate_lookup_grounds_a_confounder_the_anchor_only_search_misses(self):
        """OptimusKG has no free-text search: a relation is only found by
        querying two labels directly. The four anchor terms
        (population/treatment/comparator/outcome) never mention "age", so an
        anchor-only pairwise sweep can never surface a KG edge naming a
        first-pass covariate. This edge only exists under the ("age",
        "All-cause mortality") pair -- proving it is the covariate-targeted
        lookup, not the anchor sweep, that finds it.
        """
        calls = []

        def llm(prompt: str) -> str:
            calls.append(prompt)
            if len(calls) == 1:
                return json.dumps(valid_response())
            if len(calls) == 2:
                self.assertIn(AGE_MORTALITY_EDGE.citation_id, prompt)
                return json.dumps({
                    "candidate_dag": [
                        {
                            "label": "age",
                            "role": "confounder",
                            "pmid": AGE_MORTALITY_EDGE.citation_id,
                            "temporal_classification": "pre_treatment",
                            "rationale": "The knowledge graph links age to mortality.",
                        }
                    ],
                    "covariate_relations": [],
                })
            return json.dumps({"verdict": "approve", "concerns": []})

        kg = FakeKnowledgeGraph({
            frozenset({"age", "All-cause mortality"}): (AGE_MORTALITY_EDGE,),
        })
        result = self.service.generate(QUESTION, llm, knowledge_graph=kg)
        self.assertEqual(len(result.protocol["candidate_dag"]), 1)
        self.assertEqual(result.hypothesis.candidate_dag[0].pmid, AGE_MORTALITY_EDGE.citation_id)
        self.assertIn(
            {"source": "optimuskg", "id": AGE_MORTALITY_EDGE.citation_id},
            result.protocol["literature_evidence"]["retrieved_citations"],
        )

    def test_contested_certainty_is_accepted_and_propagated(self):
        calls = []

        def llm(prompt: str) -> str:
            calls.append(prompt)
            if len(calls) == 1:
                return json.dumps(valid_response())
            if len(calls) == 2:
                return json.dumps({
                    "candidate_dag": [
                        {
                            "label": "Baseline severity",
                            "role": "confounder",
                            "pmid": "111",
                            "temporal_classification": "pre_treatment",
                            "rationale": "Different cited studies disagree on this association's direction.",
                            "certainty": "contested",
                        }
                    ],
                    "covariate_relations": [],
                })
            return json.dumps({"verdict": "approve", "concerns": []})

        result = self.service.generate(
            QUESTION, llm, literature=fake_literature(GROUNDED_ESEARCH, GROUNDED_EFETCH)
        )
        self.assertEqual(result.hypothesis.candidate_dag[0].certainty, "contested")

    def test_invalid_certainty_falls_back_to_established(self):
        calls = []

        def llm(prompt: str) -> str:
            calls.append(prompt)
            if len(calls) == 1:
                return json.dumps(valid_response())
            if len(calls) == 2:
                return json.dumps({
                    "candidate_dag": [
                        {
                            "label": "Baseline severity",
                            "role": "confounder",
                            "pmid": "111",
                            "temporal_classification": "pre_treatment",
                            "rationale": "Baseline severity is linked to treatment choice and to mortality.",
                            "certainty": "very confident",
                        }
                    ],
                    "covariate_relations": [],
                })
            return json.dumps({"verdict": "approve", "concerns": []})

        result = self.service.generate(
            QUESTION, llm, literature=fake_literature(GROUNDED_ESEARCH, GROUNDED_EFETCH)
        )
        self.assertEqual(result.hypothesis.candidate_dag[0].certainty, "established")

    def test_rejects_hallucinated_pmid(self):
        calls = []

        def llm(prompt: str) -> str:
            calls.append(prompt)
            if len(calls) == 1:
                return json.dumps(valid_response())
            return json.dumps({
                "candidate_dag": [
                    {
                        "label": "Baseline severity",
                        "role": "confounder",
                        "pmid": "999999",
                        "rationale": "This PMID was never retrieved.",
                    }
                ],
                "covariate_relations": [],
            })

        result = self.service.generate(
            QUESTION, llm, literature=fake_literature(GROUNDED_ESEARCH, GROUNDED_EFETCH)
        )
        self.assertEqual(result.protocol["candidate_dag"], [])
        self.assertTrue(
            any("rejected" in note for note in result.protocol["design_constraints"])
        )

    def test_rejects_dag_variable_missing_temporal_classification(self):
        calls = []

        def llm(prompt: str) -> str:
            calls.append(prompt)
            if len(calls) == 1:
                return json.dumps(valid_response())
            return json.dumps({
                "candidate_dag": [
                    {
                        "label": "Baseline severity",
                        "role": "confounder",
                        "pmid": "111",
                        "rationale": "Baseline severity is linked to treatment choice and to mortality.",
                    }
                ],
                "covariate_relations": [],
            })

        result = self.service.generate(
            QUESTION, llm, literature=fake_literature(GROUNDED_ESEARCH, GROUNDED_EFETCH)
        )
        self.assertEqual(result.protocol["candidate_dag"], [])

    def test_rejects_dag_variable_with_role_temporal_mismatch(self):
        calls = []

        def llm(prompt: str) -> str:
            calls.append(prompt)
            if len(calls) == 1:
                return json.dumps(valid_response())
            return json.dumps({
                "candidate_dag": [
                    {
                        "label": "new resistant organism",
                        "role": "confounder",
                        "pmid": "111",
                        "temporal_classification": "post_treatment",
                        "rationale": "Cultured only after treatment began.",
                    }
                ],
                "covariate_relations": [],
            })

        result = self.service.generate(
            QUESTION, llm, literature=fake_literature(GROUNDED_ESEARCH, GROUNDED_EFETCH)
        )
        self.assertEqual(result.protocol["candidate_dag"], [])

    def test_no_literature_found_skips_second_llm_call(self):
        calls = []

        def llm(prompt: str) -> str:
            calls.append(prompt)
            return json.dumps(valid_response())

        result = self.service.generate(
            QUESTION,
            llm,
            literature=fake_literature(EMPTY_ESEARCH, b"<PubmedArticleSet></PubmedArticleSet>"),
        )
        self.assertEqual(len(calls), 1)
        self.assertEqual(result.protocol["candidate_dag"], [])
        self.assertTrue(
            any("No literature could be retrieved" in note for note in result.protocol["design_constraints"])
        )

    def test_critique_approves_candidate_dag_without_revision(self):
        calls = []

        def llm(prompt: str) -> str:
            calls.append(prompt)
            if len(calls) == 1:
                return json.dumps(valid_response())
            if len(calls) == 2:
                return json.dumps({
                    "candidate_dag": [
                        {
                            "label": "Baseline severity",
                            "role": "confounder",
                            "pmid": "111",
                            "temporal_classification": "pre_treatment",
                            "rationale": "Baseline severity is linked to treatment choice and to mortality.",
                        }
                    ],
                    "covariate_relations": [],
                })
            return json.dumps({"verdict": "approve", "concerns": []})

        result = self.service.generate(
            QUESTION, llm, literature=fake_literature(GROUNDED_ESEARCH, GROUNDED_EFETCH)
        )
        self.assertEqual(len(calls), 3)
        self.assertEqual([item.stage for item in result.llm_outputs], [
            "target_trial_draft", "literature_dag_draft", "literature_dag_critique"
        ])
        self.assertEqual(len(result.protocol["candidate_dag"]), 1)
        self.assertEqual(result.protocol["candidate_dag"][0]["label"], "Baseline severity")
        self.assertTrue(
            any("approved without changes" in note for note in result.protocol["design_constraints"])
        )

    def test_critique_requests_revision_and_generator_revises(self):
        calls = []

        def llm(prompt: str) -> str:
            calls.append(prompt)
            if len(calls) == 1:
                return json.dumps(valid_response())
            if len(calls) == 2:
                return json.dumps({
                    "candidate_dag": [
                        {
                            "label": "Baseline severity",
                            "role": "confounder",
                            "pmid": "111",
                            "temporal_classification": "pre_treatment",
                            "rationale": "Baseline severity is linked to treatment choice and to mortality.",
                        }
                    ],
                    "covariate_relations": [],
                })
            if len(calls) == 3:
                return json.dumps({
                    "verdict": "revise",
                    "concerns": [
                        {"target": "Baseline severity", "issue": "the cited abstract does not support this direction"}
                    ],
                })
            self.assertIn("Baseline severity", prompt)
            self.assertIn("the cited abstract does not support this direction", prompt)
            return json.dumps({
                "candidate_dag": [
                    {
                        "label": "Baseline severity",
                        "role": "outcome_cause_only",
                        "pmid": "111",
                        "temporal_classification": "pre_treatment",
                        "rationale": "Revised: the abstract only supports an outcome effect, not treatment choice.",
                    }
                ],
                "covariate_relations": [],
            })

        result = self.service.generate(
            QUESTION, llm, literature=fake_literature(GROUNDED_ESEARCH, GROUNDED_EFETCH)
        )
        self.assertEqual(len(calls), 4)
        self.assertEqual([item.stage for item in result.llm_outputs], [
            "target_trial_draft", "literature_dag_draft", "literature_dag_critique", "literature_dag_revision"
        ])
        self.assertEqual(result.protocol["candidate_dag"][0]["role"], "outcome_cause_only")
        self.assertTrue(
            any("revised after critique" in note for note in result.protocol["design_constraints"])
        )

    def test_malformed_critique_response_degrades_to_pre_critique_dag(self):
        calls = []

        def llm(prompt: str) -> str:
            calls.append(prompt)
            if len(calls) == 1:
                return json.dumps(valid_response())
            if len(calls) == 2:
                return json.dumps({
                    "candidate_dag": [
                        {
                            "label": "Baseline severity",
                            "role": "confounder",
                            "pmid": "111",
                            "temporal_classification": "pre_treatment",
                            "rationale": "Baseline severity is linked to treatment choice and to mortality.",
                        }
                    ],
                    "covariate_relations": [],
                })
            return "not valid JSON"

        result = self.service.generate(
            QUESTION, llm, literature=fake_literature(GROUNDED_ESEARCH, GROUNDED_EFETCH)
        )
        self.assertEqual(len(calls), 3)
        self.assertEqual(len(result.protocol["candidate_dag"]), 1)
        self.assertEqual(result.protocol["candidate_dag"][0]["label"], "Baseline severity")
        self.assertTrue(
            any("critique step failed" in note for note in result.protocol["design_constraints"])
        )

    def test_malformed_revision_response_degrades_to_pre_revision_dag(self):
        calls = []

        def llm(prompt: str) -> str:
            calls.append(prompt)
            if len(calls) == 1:
                return json.dumps(valid_response())
            if len(calls) == 2:
                return json.dumps({
                    "candidate_dag": [
                        {
                            "label": "Baseline severity",
                            "role": "confounder",
                            "pmid": "111",
                            "temporal_classification": "pre_treatment",
                            "rationale": "Baseline severity is linked to treatment choice and to mortality.",
                        }
                    ],
                    "covariate_relations": [],
                })
            if len(calls) == 3:
                return json.dumps({
                    "verdict": "revise",
                    "concerns": [{"target": "Baseline severity", "issue": "check the timing"}],
                })
            return "not valid JSON"

        result = self.service.generate(
            QUESTION, llm, literature=fake_literature(GROUNDED_ESEARCH, GROUNDED_EFETCH)
        )
        self.assertEqual(len(calls), 4)
        self.assertEqual(len(result.protocol["candidate_dag"]), 1)
        self.assertEqual(result.protocol["candidate_dag"][0]["label"], "Baseline severity")
        self.assertTrue(
            any("revision step failed" in note for note in result.protocol["design_constraints"])
        )

    def test_revision_hallucinated_pmid_is_still_rejected(self):
        calls = []

        def llm(prompt: str) -> str:
            calls.append(prompt)
            if len(calls) == 1:
                return json.dumps(valid_response())
            if len(calls) == 2:
                return json.dumps({
                    "candidate_dag": [
                        {
                            "label": "Baseline severity",
                            "role": "confounder",
                            "pmid": "111",
                            "temporal_classification": "pre_treatment",
                            "rationale": "Baseline severity is linked to treatment choice and to mortality.",
                        }
                    ],
                    "covariate_relations": [],
                })
            if len(calls) == 3:
                return json.dumps({
                    "verdict": "revise",
                    "concerns": [{"target": "general", "issue": "consider an additional confounder"}],
                })
            return json.dumps({
                "candidate_dag": [
                    {
                        "label": "Unretrieved variable",
                        "role": "confounder",
                        "pmid": "999999",
                        "temporal_classification": "pre_treatment",
                        "rationale": "This PMID was never retrieved.",
                    }
                ],
                "covariate_relations": [],
            })

        result = self.service.generate(
            QUESTION, llm, literature=fake_literature(GROUNDED_ESEARCH, GROUNDED_EFETCH)
        )
        self.assertEqual(len(calls), 4)
        self.assertEqual(result.protocol["candidate_dag"], [])

    def test_enable_critique_false_short_circuits_to_two_calls(self):
        calls = []

        def llm(prompt: str) -> str:
            calls.append(prompt)
            if len(calls) == 1:
                return json.dumps(valid_response())
            return json.dumps({
                "candidate_dag": [
                    {
                        "label": "Baseline severity",
                        "role": "confounder",
                        "pmid": "111",
                        "temporal_classification": "pre_treatment",
                        "rationale": "Baseline severity is linked to treatment choice and to mortality.",
                    }
                ],
                "covariate_relations": [],
            })

        result = self.service.generate(
            QUESTION, llm, literature=fake_literature(GROUNDED_ESEARCH, GROUNDED_EFETCH),
            enable_critique=False,
        )
        self.assertEqual(len(calls), 2)
        self.assertEqual([item.stage for item in result.llm_outputs], [
            "target_trial_draft", "literature_dag_draft"
        ])
        self.assertEqual(len(result.protocol["candidate_dag"]), 1)

    def test_critique_skipped_when_candidate_dag_is_empty(self):
        calls = []

        def llm(prompt: str) -> str:
            calls.append(prompt)
            if len(calls) == 1:
                return json.dumps(valid_response())
            return json.dumps({
                "candidate_dag": [
                    {
                        "label": "Baseline severity",
                        "role": "confounder",
                        "pmid": "999999",
                        "temporal_classification": "pre_treatment",
                        "rationale": "This PMID was never retrieved.",
                    }
                ],
                "covariate_relations": [],
            })

        result = self.service.generate(
            QUESTION, llm, literature=fake_literature(GROUNDED_ESEARCH, GROUNDED_EFETCH)
        )
        self.assertEqual(len(calls), 2)
        self.assertEqual(result.protocol["candidate_dag"], [])

    def test_covariate_targeted_search_grounds_a_confounder_the_generic_search_misses(self):
        """The generic P/T/C/O search finds nothing at all; only the
        per-covariate targeted search (for the "age" first-pass covariate)
        turns up an article, and that article must still be citable and
        accepted -- proving a classic confounder is no longer solely at the
        mercy of what the generic search happens to retrieve.
        """
        calls = []

        def fetcher(url: str) -> bytes:
            if "esearch.fcgi" in url:
                return AGE_ESEARCH if "age" in url else EMPTY_ESEARCH
            return AGE_EFETCH

        literature = LiteratureOrchestrator(pubmed=PubMedClient(PubMedConfig(), fetcher=fetcher))

        def llm(prompt: str) -> str:
            calls.append(prompt)
            if len(calls) == 1:
                return json.dumps(valid_response())
            if len(calls) == 2:
                self.assertIn("222", prompt)
                return json.dumps({
                    "candidate_dag": [
                        {
                            "label": "age",
                            "role": "confounder",
                            "pmid": "222",
                            "temporal_classification": "pre_treatment",
                            "rationale": "Older age independently predicts both treatment choice and mortality.",
                        }
                    ],
                    "covariate_relations": [],
                })
            return json.dumps({"verdict": "approve", "concerns": []})

        result = self.service.generate(QUESTION, llm, literature=literature)

        self.assertEqual(len(result.protocol["candidate_dag"]), 1)
        self.assertEqual(result.hypothesis.candidate_dag[0].label, "age")
        self.assertEqual(result.hypothesis.candidate_dag[0].pmid, "222")
        self.assertEqual(result.hypothesis.candidate_dag[0].role, "confounder")
        self.assertIn(
            {"source": "pubmed", "id": "222"},
            result.protocol["literature_evidence"]["retrieved_citations"],
        )
        self.assertTrue(
            any(
                "1 first-pass covariate(s) had dedicated targeted evidence" in note
                for note in result.protocol["design_constraints"]
            )
        )

    def test_unresolved_dag_covariates_are_surfaced(self):
        calls = []

        def llm(prompt: str) -> str:
            calls.append(prompt)
            if len(calls) == 1:
                return json.dumps(valid_response())
            return json.dumps({
                "candidate_dag": [],
                "covariate_relations": [],
                "unresolved": [
                    {"label": "renal function", "reason": "no retrieved abstract discusses it"}
                ],
            })

        result = self.service.generate(
            QUESTION, llm, literature=fake_literature(GROUNDED_ESEARCH, GROUNDED_EFETCH),
            enable_critique=False,
        )
        self.assertEqual(
            result.protocol["candidate_dag_unresolved"],
            [{"label": "renal function", "reason": "no retrieved abstract discusses it"}],
        )
        self.assertTrue(
            any("flagged unresolved" in note for note in result.protocol["design_constraints"])
        )


class CovariateSearchTermTests(unittest.TestCase):
    """Population terms are mandatory, not just another OR option alongside
    treatment/outcome -- otherwise a covariate paired only with a generic
    outcome term like "mortality" is satisfied by almost any critical-care
    study regardless of population (see `_covariate_search_term`'s docstring
    for a real example this caused)."""

    def test_requires_population_terms(self):
        self.assertEqual(_covariate_search_term("age", (), ("mortality",)), "")

    def test_requires_anchor_terms(self):
        self.assertEqual(_covariate_search_term("age", ("critical illness",), ()), "")

    def test_builds_population_and_anchor_group(self):
        term = _covariate_search_term("age", ("critical illness",), ("treatment A", "mortality"))
        self.assertEqual(
            term,
            '"age"[Title/Abstract] AND ("critical illness"[Title/Abstract]) AND '
            '("treatment A"[Title/Abstract] OR "mortality"[Title/Abstract])',
        )


class KGReinforcementTests(unittest.TestCase):
    """`_kg_reinforcement` must require the variable label and the anchor
    term to match *opposite* endpoints of the same KG edge -- matching both
    on the same endpoint is a coincidental substring collision, not evidence
    that the edge relates the variable to the anchor at all.
    """

    def test_requires_opposite_endpoints_not_just_any_endpoint(self):
        # Both "age" and "treatment a" happen to be substrings of the same
        # (source) label; the target is unrelated to either. A same-side
        # match here must not be read as the KG confirming a link between
        # "age" and "Treatment A".
        edge = KGEdge(
            source_id="A1",
            source_label="Age at Treatment A initiation",
            predicate="related_to",
            target_id="A2",
            target_label="Unrelated concept",
        )
        result = _kg_reinforcement("age", ("Treatment A", "All-cause mortality"), (edge,), "111")
        self.assertEqual(result, "")

    def test_matches_when_label_and_anchor_are_on_opposite_endpoints(self):
        edge = KGEdge(
            source_id="A1",
            source_label="Age",
            predicate="associated_with",
            target_id="A2",
            target_label="All-cause mortality",
        )
        result = _kg_reinforcement("age", ("Treatment A", "All-cause mortality"), (edge,), "111")
        self.assertEqual(result, edge.citation_id)

    def test_kg_only_pmid_is_never_reinforced_by_itself(self):
        edge = KGEdge(
            source_id="A1", source_label="Age", predicate="associated_with",
            target_id="A2", target_label="All-cause mortality",
        )
        result = _kg_reinforcement("age", ("Treatment A", "All-cause mortality"), (edge,), edge.citation_id)
        self.assertEqual(result, "")


if __name__ == "__main__":
    unittest.main()
