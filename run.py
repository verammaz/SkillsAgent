from agent import SkillAgent

if __name__ == "__main__":

    agent = SkillAgent()

    query = "Why is Chiller 6 behaving abnormally and do we need a work order?"

    result = agent.run(query)

    print("\nFinal Output:")
    print(result)