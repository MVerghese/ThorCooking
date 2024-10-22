from Base_Agent import Base_Agent
import numpy as np
from Environment import CookingEnv
import json
from typing import List, Dict
LLAMA_PATH = "gemma-2-2b-it"
MODEL_PATH = "/home/atkesonlab2/models/"


class Action_Agent(Base_Agent):
    '''
    Given a prompt and a set of predicates representing the environment, outputs the next most probable action
    Input:
            prompt: str - task to complete
            predicates: List[str] - environment state
            actions: List[Dict] - potential actions to take
    Output: str - most likely next action
    '''

    def generate_next_action(self, prompt: str, predicates: List[str], actions: List[Dict], action_history: List[str]) -> str:
        final_prompt = "A person has been assigned the following task to complete: " + \
            prompt + "\nHere is the current state of the environment:\n"
        for predicate in predicates:
            final_prompt += predicate + "\n"

        final_prompt += "Here is a breakdown of actions the person has taken so far:\n"
        for act in action_history:
            final_prompt += act + "\n"

        final_prompt += "Based on the previous information, determine what the person needs to do next to reach their goal."
        #  If you are done with the task, respond with \"done\"
        actions_list = []
        for act in actions:
            actions_list.append(act["action"].replace(
                "Object", " " + act["objectType"] + " ").lower())

        actions_list.append("done")
        # print(final_prompt)
        # print(actions_list)

        action_probabilities = self.llm.eval_log_probs(
            final_prompt, actions_list, batch_size=1)
        print(action_probabilities)
        print(actions_list)
        res_idx = max(enumerate(action_probabilities),
                      key=lambda x: x[1])[0]
        if (res_idx == len(actions_list) - 1):
            return None
        return actions[res_idx]


def test_generate_next_action_blt():
    agent = Action_Agent(load_llm=True, load_narrator=False)
    env = CookingEnv()
    with open('Tasks/Make_A_BLT_0.json') as json_file:
        task_dict = json.load(json_file)
    env.load_task_state(task_dict, 8)
    action_history = []

    for i in range(9):
        action_history.append(task_dict["action_segments"][i]["action_text"])

    def filter_by_object(pred):
        pred = pred.lower()
        return "bacon" in pred or "lettuce" in pred or "tomato" in pred or "bread" in pred or "knife" in pred or "plate" in pred

    prompt = "Make a BLT"

    actions_list = env.generate_possible_actions(False)
    filtered_actions = list(
        filter(lambda x: filter_by_object(x["objectType"]), actions_list))

    predicates = env.generate_language_predicates()
    filtered_predicates = list(filter(filter_by_object, predicates))

    next_action = agent.generate_next_action(
        prompt, filtered_predicates, filtered_actions, action_history)

    if (next_action is None):
        return
    action_history.append(next_action["action"].replace(
        "Object", " " + next_action["objectType"] + " ").lower())
    print(next_action)

    for _ in range(1):
        env.step(next_action)
        actions_list = env.generate_possible_actions(False)
        filtered_actions = list(
            filter(lambda x: filter_by_object(x["objectType"]), actions_list))

        predicates = env.generate_language_predicates()
        filtered_predicates = list(filter(filter_by_object, predicates))

        next_action = agent.generate_next_action(
            prompt, filtered_predicates, filtered_actions, action_history)
        print(next_action)
        if (next_action == None):
            return
        action_history.append(next_action["action"].replace(
            "Object", " " + next_action["objectType"] + " ").lower())


def find_relevant_actions(goal: str, past_actions: List[str], failure: str, potential_actions: List):
    return


def test_fix_failure():
    successful_floorplans = [2, 3, 6, 9, 11, 12, 14, 15, 17, 23, 28, 29, 30]
    unsuccessful_floorplans = [1, 4, 5, 7, 8, 10,
                               13, 16, 18, 19, 20, 21, 22, 24, 25, 26, 27]

    agent = Base_Agent(load_llm=True, load_narrator=False)
    env = CookingEnv(scene_name="FloorPlan1")
    with open('Tasks/Make_A_BLT_0.json') as json_file:
        task_dict = json.load(json_file)
    env.load_task_state(task_dict, 18)
    valid_scene = env.check_success()
    print(valid_scene)
    action_history = []

    for i in range(18):
        action_history.append(task_dict["action_segments"][i]["action_text"])

    def filter_by_object(pred):
        pred = pred.lower()
        return "bacon" in pred or "lettuce" in pred or "tomato" in pred or "bread" in pred or "knife" in pred or "plate" in pred

    prompt = "Make a BLT"

    actions_list = env.generate_possible_actions(False)
    filtered_actions = list(
        filter(lambda x: filter_by_object(x["objectType"]), actions_list))

    predicates = env.generate_language_predicates()
    print(predicates)
    filtered_predicates = list(filter(filter_by_object, predicates))

    next_action = agent.generate_next_action(
        prompt, filtered_predicates, filtered_actions, action_history)

    if (next_action is None):
        return
    action_history.append(next_action["action"].replace(
        "Object", " " + next_action["objectType"] + " ").lower())
    print(next_action)

    for _ in range(1):
        env.step(next_action)
        actions_list = env.generate_possible_actions(False)
        filtered_actions = list(
            filter(lambda x: filter_by_object(x["objectType"]), actions_list))

        predicates = env.generate_language_predicates()
        filtered_predicates = list(filter(filter_by_object, predicates))

        next_action = agent.generate_next_action(
            prompt, filtered_predicates, filtered_actions, action_history)
        print(next_action)
        if (next_action == None):
            return
        action_history.append(next_action["action"].replace(
            "Object", " " + next_action["objectType"] + " ").lower())


if __name__ == "__main__":
    test_fix_failure()
