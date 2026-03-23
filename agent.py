from skills import root_cause_analysis, validate_failure, create_work_order


class SkillAgent:

    def plan(self, task: str):
        # simple fixed plan (baseline)
        return [
            "root_cause_analysis",
            "validate_failure",
            "create_work_order"
        ]

    def run(self, task: str):
        asset_id = "Chiller 6"

        plan = self.plan(task)
        print(f"Plan: {plan}")

        context = {}

        for step in plan:
            print(f"\nExecuting: {step}")

            if step == "root_cause_analysis":
                context.update(root_cause_analysis(asset_id))

            elif step == "validate_failure":
                result = validate_failure(asset_id, context["failure"])
                context.update(result)

                if not result["validated"]:
                    print("Skipping work order (not needed)")
                    return "No work order needed"

            elif step == "create_work_order":
                return create_work_order(asset_id, context["failure"])

        return "Done"