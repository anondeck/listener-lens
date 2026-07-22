from earshift_bakeoff.same_take_corrective import run_corrective_grid


if __name__ == "__main__":
    result = run_corrective_grid()
    print(result["status"])
    print(result["selected_candidate"])
    print(result["selected_shift"])
    print(result["receipt_sha256"])

