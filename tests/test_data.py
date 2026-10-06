from breast_cancer_ml.data import FEATURE_NAMES, TARGET, make_synthetic_cohort


def test_synthetic_cohort_is_identifier_free_and_reproducible():
    first = make_synthetic_cohort(n_samples=120, random_state=7)
    second = make_synthetic_cohort(n_samples=120, random_state=7)

    assert first.equals(second)
    assert list(first.columns) == FEATURE_NAMES + [TARGET]
    assert first.shape == (120, 21)
    assert not first.isna().any().any()
    forbidden_identifier_fields = {"id", "patient_id", "medical_record_number", "mrn"}
    assert not forbidden_identifier_fields.intersection(column.lower() for column in first.columns)
    assert not any("date" in column.lower() for column in first.columns)
