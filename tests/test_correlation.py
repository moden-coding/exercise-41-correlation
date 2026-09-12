#!/usr/bin/env python3
"""Tests for the Correlation assignment."""

import unittest
from unittest.mock import patch

from src.correlation import correlations, lengths


class TestLengths(unittest.TestCase):
    """lengths() -> Pearson correlation between two of the iris length columns."""

    def test_lengths(self):
        result = lengths()
        self.assertAlmostEqual(
            result,
            0.8717537758865832,
            places=4,
            msg="lengths() should be approximately 0.8717537758865832, got "
            "%r." % (result,),
        )

    def test_lengths_calls(self):
        with patch("scipy.stats.pearsonr") as pcorr:
            lengths()
            self.assertTrue(
                pcorr.called,
                msg="lengths() should compute the correlation with "
                "scipy.stats.pearsonr.",
            )


class TestCorrelations(unittest.TestCase):
    """correlations() -> full correlation matrix of the iris measurement columns."""

    def test_correlations(self):
        result = correlations()
        n, m = result.shape
        for r in range(n):
            for c in range(r):
                self.assertAlmostEqual(
                    result[r, c],
                    result[c, r],
                    places=4,
                    msg="The correlation matrix is not symmetric: "
                    "result[%d,%d]=%r != result[%d,%d]=%r."
                    % (r, c, result[r, c], c, r, result[c, r]),
                )
            self.assertAlmostEqual(
                result[r, r],
                1,
                places=4,
                msg="Values on the diagonal should be one, got "
                "result[%d,%d]=%r." % (r, r, result[r, r]),
            )

        self.assertAlmostEqual(
            result[0, 1],
            -0.11756978,
            places=4,
            msg="Incorrect value in position [0,1]: expected -0.11756978, "
            "got %r." % (result[0, 1],),
        )
        self.assertAlmostEqual(
            result[0, 2],
            0.87175378,
            places=4,
            msg="Incorrect value in position [0,2]: expected 0.87175378, "
            "got %r." % (result[0, 2],),
        )
        self.assertAlmostEqual(
            result[0, 3],
            0.81794113,
            places=4,
            msg="Incorrect value in position [0,3]: expected 0.81794113, "
            "got %r." % (result[0, 3],),
        )
        self.assertAlmostEqual(
            result[1, 2],
            -0.4284401,
            places=4,
            msg="Incorrect value in position [1,2]: expected -0.4284401, "
            "got %r." % (result[1, 2],),
        )
        self.assertAlmostEqual(
            result[1, 3],
            -0.36612593,
            places=4,
            msg="Incorrect value in position [1,3]: expected -0.36612593, "
            "got %r." % (result[1, 3],),
        )
        self.assertAlmostEqual(
            result[2, 3],
            0.96286543,
            places=4,
            msg="Incorrect value in position [2,3]: expected 0.96286543, "
            "got %r." % (result[2, 3],),
        )

    def test_correlations_calls(self):
        with patch("numpy.corrcoef") as pcorr:
            correlations()
            self.assertTrue(
                pcorr.called,
                msg="correlations() should compute the correlation matrix "
                "with np.corrcoef.",
            )


if __name__ == "__main__":
    unittest.main()
