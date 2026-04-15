"""Example: Forgetting check after training a new stack."""
from src.eval.forgetting import check_forgetting, save_forgetting_report

results = []
for prior in ["stack-01-chat-fr", "stack-02-reasoning"]:
    r = check_forgetting(
        win_rate_delta=0.01, angle_deg=55.0,
        prior_stack=prior, new_stack="stack-03-python",
    )
    results.append(r)
    print(f"{prior}: passed={r.passed}, angle={r.gradient_subspace_angle_deg}°")

report = save_forgetting_report(results, "stack-03-python", "results")
print(f"Report saved to {report}")
