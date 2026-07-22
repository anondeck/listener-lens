from earshift_bakeoff.same_take_world import run_world_pass


if __name__ == "__main__":
    result = run_world_pass()
    print(result["status"])
    print(result["selected_shift"])
    print(result["receipt_sha256"])

