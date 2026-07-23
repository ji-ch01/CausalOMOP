import json
import tempfile
import unittest
from pathlib import Path

from causal.agents.user_question import UserQuestionHypothesisService
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


def fake_pubmed(esearch_json: bytes, efetch_xml: bytes) -> PubMedClient:
    def fetcher(url: str) -> bytes:
        return esearch_json if "esearch.fcgi" in url else efetch_xml

    return PubMedClient(PubMedConfig(), fetcher=fetcher)


def fake_literature(esearch_json: bytes, efetch_xml: bytes) -> LiteratureOrchestrator:
    return LiteratureOrchestrator(pubmed=fake_pubmed(esearch_json, efetch_xml))


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
            "COMPARATOR={{COMPARATOR}}\nOUTCOME={{OUTCOME}}\nARTICLES={{ARTICLES}}",
            encoding="utf-8",
        )
        self.service = UserQuestionHypothesisService(self.prompt, self.dag_prompt)

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
            QUESTION, llm, literature=fake_literature(GROUNDED_ESEARCH, GROUNDED_EFETCH)
        )
        self.assertEqual(len(calls), 2)
        self.assertEqual(len(result.protocol["candidate_dag"]), 1)
        self.assertEqual(result.hypothesis.candidate_dag[0].pmid, "111")
        self.assertEqual(result.hypothesis.candidate_dag[0].label, "Baseline severity")
        self.assertEqual([item.stage for item in result.llm_outputs], [
            "target_trial_draft", "literature_dag_draft"
        ])
        self.assertEqual(
            result.protocol["literature_evidence"]["retrieved_citations"],
            [{"source": "pubmed", "id": "111"}],
        )
        self.assertIn("Title/Abstract", result.protocol["literature_evidence"]["search_query"])

    def test_contested_certainty_is_accepted_and_propagated(self):
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
                        "rationale": "Different cited studies disagree on this association's direction.",
                        "certainty": "contested",
                    }
                ],
                "covariate_relations": [],
            })

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


if __name__ == "__main__":
    unittest.main()
