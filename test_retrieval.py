import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from retrieval import FaqSearcher, expand_query, load_faqs

# (user wording, expected FAQ question or None for "no good match")
EVAL_SET = [
    ("How can I prepare for exams?", "How can I prepare for exams?"),
    ("How do I manage my time?", "How can I manage university and personal tasks?"),
    ("How can I make a good study timetable?", "How do I make a study timetable?"),
    ("tips for studying for finals", "How can I prepare for exams?"),
    ("What is AI?", "What is artificial intelligence?"),
    ("explain ML", "What is machine learning?"),
    ("What is NLP?", "What is natural language processing?"),
    ("difference between training and test data",
     "Why do we split data into training and testing sets?"),
    ("what does a python dict do", "What is a dictionary in Python?"),
    ("what is a list in python", "What is a Python list?"),
    ("how to write a readme", "What should I include in a project README?"),
    ("how do I give a good presentation", "How can I prepare for a presentation?"),
    ("how to take notes in class", "How should I take lecture notes?"),
    ("what is the weather today", None),
    ("who won the cricket match", None),
    ("the of and", None),
]


class TestData(unittest.TestCase):
    def test_bundled_faqs_load(self):
        faqs = load_faqs()
        self.assertGreaterEqual(len(faqs), 20)
        self.assertEqual(faqs["question"].duplicated().sum(), 0)

    def test_missing_column_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "bad.csv"
            p.write_text("question,reply\nhi,there\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                load_faqs(p)

    def test_missing_file_rejected(self):
        with self.assertRaises(ValueError):
            load_faqs(Path("/nonexistent/faqs.csv"))


class TestSearch(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.searcher = FaqSearcher(load_faqs())

    def test_alias_expansion(self):
        self.assertIn("machine learning", expand_query("what is ML"))
        self.assertEqual(expand_query("hello world"), "hello world")

    def test_exact_question_is_top_match(self):
        for question in self.searcher.faqs["question"]:
            row, score = self.searcher.best_match(question)
            self.assertIsNotNone(row, question)
            self.assertEqual(row["question"], question)

    def test_retrieval_accuracy_on_eval_set(self):
        hits = 0
        for query, expected in EVAL_SET:
            row, _ = self.searcher.best_match(query)
            hits += (None if row is None else row["question"]) == expected
        accuracy = hits / len(EVAL_SET)
        print(f"\nEval-set accuracy: {hits}/{len(EVAL_SET)} = {accuracy:.0%}")
        self.assertGreaterEqual(accuracy, 0.85)

    def test_gibberish_has_no_match(self):
        row, _ = self.searcher.best_match("zzzz qqqq xxxx")
        self.assertIsNone(row)


class TestApp(unittest.TestCase):
    def test_app_runs_and_answers(self):
        try:
            from streamlit.testing.v1 import AppTest
        except ImportError:
            self.skipTest("streamlit not installed")
        at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=30).run()
        self.assertFalse(at.exception)
        at.text_input(key="question_input").set_value("What is machine learning?").run()
        at.button[-1].click().run()  # "Find my answer" is the last button
        self.assertFalse(at.exception)
        self.assertTrue(any("answer-card" in m.value for m in at.markdown))


if __name__ == "__main__":
    unittest.main()
