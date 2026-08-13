"""Unit tests for the E12 placeholder-template module."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "04_query"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "04_query" / "soft_prefix"))

from template_placeholder import (
    replace_attrs_with_placeholders,
    replace_placeholders_with_attrs,
    parse_template,
    build_attr_mapping_prompt,
)


class TemplatePlaceholderTest(unittest.TestCase):

    def test_roundtrip_basic(self):
        attrs = {"A1": "Ties", "A2": "KK BETO", "A3": "8.99", "A4": "Small", "A5": "Baby"}
        q = "I'm searching for KK BETO Ties in a Small appearance for Baby at 8.99."
        tpl, err = replace_attrs_with_placeholders(q, attrs)
        self.assertIsNone(err, err)
        final, err2 = replace_placeholders_with_attrs(tpl, attrs)
        self.assertIsNone(err2, err2)
        self.assertEqual(final, q)

    def test_numeric_fidelity(self):
        # 30.77 and repeated-digit prices must survive verbatim
        attrs = {"A1": "Car Seats", "A2": "Baby Trend", "A3": "30.77", "A4": "Black", "A5": "Storage"}
        q = "I would like a Baby Trend Car Seats option in Black that supports Storage and is priced around 30.77."
        tpl, err = replace_attrs_with_placeholders(q, attrs)
        self.assertIsNone(err, err)
        final, _ = replace_placeholders_with_attrs(tpl, attrs)
        self.assertEqual(final, q)
        self.assertIn("30.77", final)

    def test_longest_match_first(self):
        # "Baby" is contained in "Baby Trend": longer value must win
        attrs = {"A1": "Gift Sets", "A2": "BabySquad", "A3": "21.95", "A4": "White", "A5": "Baby"}
        q = "I'm looking for a BabySquad Gift Sets option for Baby use, in White color at 21.95."
        tpl, err = replace_attrs_with_placeholders(q, attrs)
        self.assertIsNone(err, err)
        for ph in ["<A1>", "<A2>", "<A3>", "<A4>", "<A5>"]:
            self.assertEqual(tpl.count(ph), 1, f"{ph} count != 1 in {tpl}")
        final, _ = replace_placeholders_with_attrs(tpl, attrs)
        self.assertIn("BabySquad", final)
        self.assertIn("Baby use", final)

    def test_variant_inflection(self):
        # plural in query, singular value -> still templatable; final uses
        # the ORIGINAL value verbatim (by design)
        attrs = {"A1": "Maternity Pillow", "A2": "Bed Buddy", "A3": "29.99", "A4": "Gray", "A5": "Sleeping"}
        q = "I want a Bed Buddy Maternity Pillows that is Gray for Sleeping at 29.99."
        tpl, err = replace_attrs_with_placeholders(q, attrs)
        self.assertIsNone(err, err)
        final, _ = replace_placeholders_with_attrs(tpl, attrs)
        self.assertIn("Maternity Pillow", final)  # original value restored

    def test_punctuation_adjacent_placeholder_ok(self):
        attrs = {"A1": "Ties", "A2": "KK BETO", "A3": "8.99", "A4": "Small", "A5": "Baby"}
        tpl = "I want <A2> <A1> for <A5> at <A3>. and <A4> ones."
        parsed = parse_template(tpl)
        self.assertTrue(parsed["valid"], parsed)
        final, _ = replace_placeholders_with_attrs(tpl, attrs)
        self.assertIn("8.99.", final)

    def test_glued_placeholder_rejected(self):
        tpl = "<A3>x <A1> <A2> <A4> <A5>"
        parsed = parse_template(tpl)
        self.assertFalse(parsed["valid"])

    def test_duplicate_placeholder_rejected(self):
        tpl = "<A1> <A1> <A2> <A4> <A5> <A3>"
        parsed = parse_template(tpl)
        self.assertFalse(parsed["valid"])

    def test_missing_placeholder_rejected(self):
        tpl = "<A1> <A2> <A4> <A5>"
        parsed = parse_template(tpl)
        self.assertFalse(parsed["valid"])
        self.assertIn("<A3>", parsed["missing"])

    def test_extra_placeholder_rejected(self):
        tpl = "<A1> <A2> <A3> <A4> <A5> <A9>"
        parsed = parse_template(tpl)
        self.assertFalse(parsed["valid"])
        self.assertIn("<A9>", parsed["extra"])

    def test_prompt_contains_mapping(self):
        attrs = {"A1": "Ties", "A2": "KK BETO", "A3": "8.99", "A4": "Small", "A5": "Baby"}
        prompt = build_attr_mapping_prompt(attrs)
        self.assertIn("<A3> = 8.99", prompt)
        self.assertIn("<A1> = Ties", prompt)

    def test_multi_digit_placeholder(self):
        # attrs with key A10 (two-digit key) still templatable
        attrs = {"A1": "Bottles", "A2": "Nalgene", "A3": "28.95", "A5": "Kids", "A6": "Plastic"}
        q = "I'm looking for Nalgene Bottles priced at 28.95 for Kids, made from Plastic."
        tpl, err = replace_attrs_with_placeholders(q, attrs)
        self.assertIsNone(err, err)
        self.assertIn("<A6>", tpl)
        final, _ = replace_placeholders_with_attrs(tpl, attrs)
        self.assertEqual(final, q)


if __name__ == "__main__":
    unittest.main(verbosity=2)
