from earshift_bakeoff.same_take_followup import audit_prior_processing_path


if __name__ == "__main__":
    result = audit_prior_processing_path()
    print(result["receipt_sha256"])

