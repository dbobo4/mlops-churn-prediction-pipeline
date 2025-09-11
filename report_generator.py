"""
report_generator.py
─────────────────────────────────────────────────────────
Creates three HTML reports in ./reports:

  • drift_report.html        – legacy DataDriftTable visual report
  • tests_report.html        – column/row quality & structure checks
  • suite_report.html        – stability + no-target performance suite

Plus two JSON previews:

  • train_preview.json       – first 20 rows of reference data
  • evaluate_preview.json    – first 20 rows of current data

Call:
    paths = create_report(df_current, df_reference)

Returns:
    {
        "drift": "<path>/drift_report.html",
        "tests": "<path>/tests_report.html",
        "suite": "<path>/suite_report.html",
        "train_preview": "<path>/train_preview.json",
        "evaluate_preview": "<path>/evaluate_preview.json",
    }
─────────────────────────────────────────────────────────
"""

import os
import pandas as pd

# ───────────────────────────────
# Legacy drift report
# ───────────────────────────────
from evidently.legacy.report import Report
from evidently.legacy.metrics.data_drift.data_drift_table import DataDriftTable

# ───────────────────────────────
# Legacy TestSuite + Tests
# ───────────────────────────────
# TestSuite – try package-level import, fallback to direct module
try:
    from evidently.legacy.test_suite import TestSuite
except ImportError:
    from evidently.legacy.test_suite.test_suite import TestSuite

# Individual tests – package-level import, fallback to base_test
try:
    from evidently.legacy.tests import (
        TestNumberOfColumnsWithMissingValues,
        TestNumberOfRowsWithMissingValues,
        TestNumberOfConstantColumns,
        TestNumberOfDuplicatedRows,
        TestNumberOfDuplicatedColumns,
        TestColumnsType,
        TestNumberOfDriftedColumns,
    )
except ImportError:
    from evidently.legacy.tests.base_test import (
        TestNumberOfColumnsWithMissingValues,
        TestNumberOfRowsWithMissingValues,
        TestNumberOfConstantColumns,
        TestNumberOfDuplicatedRows,
        TestNumberOfDuplicatedColumns,
        TestColumnsType,
        TestNumberOfDriftedColumns,
    )

# Presets for stability & no-target performance
from evidently.legacy.test_preset import (
    DataStabilityTestPreset,
    NoTargetPerformanceTestPreset,
)

# ───────────────────────────────
# Paths (module-level exports)
# ───────────────────────────────
REPORTS_DIR            = "reports"
REPORT_PATH            = os.path.join(REPORTS_DIR, "drift_report.html")
TESTS_PATH             = os.path.join(REPORTS_DIR, "tests_report.html")
SUITE_PATH             = os.path.join(REPORTS_DIR, "suite_report.html")
TRAIN_PREVIEW_PATH     = os.path.join(REPORTS_DIR, "train_preview.json")
EVALUATE_PREVIEW_PATH  = os.path.join(REPORTS_DIR, "evaluate_preview.json")

# Ensure the output directory exists
os.makedirs(REPORTS_DIR, exist_ok=True)

# ───────────────────────────────
# Main generator
# ───────────────────────────────
def create_report(
    df_current: pd.DataFrame,
    df_reference: pd.DataFrame,
) -> dict:
    """Generate three legacy Evidently HTML reports plus JSON previews."""

    # Data-drift visual report
    drift = Report(metrics=[DataDriftTable()])
    drift.run(reference_data=df_reference, current_data=df_current)
    drift.save_html(REPORT_PATH)

    # Column/row quality & structure tests
    tests_suite = TestSuite(tests=[
        TestNumberOfColumnsWithMissingValues(),
        TestNumberOfRowsWithMissingValues(),
        TestNumberOfConstantColumns(),
        TestNumberOfDuplicatedRows(),
        TestNumberOfDuplicatedColumns(),
        TestColumnsType(),
        TestNumberOfDriftedColumns(),
    ])
    tests_suite.run(reference_data=df_reference, current_data=df_current)
    tests_suite.save_html(TESTS_PATH)

    # Stability & no-target performance suite
    preset_suite = TestSuite(tests=[
        DataStabilityTestPreset(),
        NoTargetPerformanceTestPreset(),
    ])
    preset_suite.run(reference_data=df_reference, current_data=df_current)
    preset_suite.save_html(SUITE_PATH)

    # Save first 20 rows of each DataFrame as JSON for the dashboard
    df_reference.head(20).to_json(TRAIN_PREVIEW_PATH, orient="records", date_format="iso")
    df_current.head(20).to_json(EVALUATE_PREVIEW_PATH, orient="records", date_format="iso")

    return {
        "drift": REPORT_PATH,
        "tests": TESTS_PATH,
        "suite": SUITE_PATH,
        "train_preview": TRAIN_PREVIEW_PATH,
        "evaluate_preview": EVALUATE_PREVIEW_PATH,
    }
