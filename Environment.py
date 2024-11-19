from ai2thor.controller import Controller
import time
import numpy as np
import json


# controller = Controller(
#     agentMode="default",
#     visibilityDistance=1.5,
#     scene="FloorPlan1",

#     # step sizes
#     gridSize=0.25,
#     snapToGrid=False,
#     rotateStepDegrees=30,

#     # image modalities
#     renderDepthImage=False,
#     renderInstanceSegmentation=False,

#     # camera properties
#     width=300,
#     height=300,
#     fieldOfView=90
# )
# # event = controller.step("MoveAhead")
# for obj in controller.last_event.metadata["objects"]:
#     print(obj["objectType"], obj["objectId"])
# event = controller.step(action='PickupObject', objectId="Knife|-01.70|+00.79|-00.22", forceAction = True, manualInteract=False)
# print(event.metadata["lastActionSuccess"])
# print(event.metadata["errorMessage"])
# print("Rotating")
# for i in range(100):
# 	event = controller.step(action='RotateRight')
# 	time.sleep(.5)

TASK_PATHS = {
    "make a blt": [
        "Tasks/Make_A_BLT_0_abridged.json",
        "Tasks/Make_A_BLT_1.json",
    ],
    "make a latte": ["Tasks/Make_A_Latte_0.json"],
}


def pos_dict_to_array(pos_dict):
    return np.array([pos_dict["x"], pos_dict["y"], pos_dict["z"]])


class CookingEnv:
    def __init__(self, scene_name="FloorPlan1"):
        self.scene_name = scene_name
        self.controller = Controller(
            agentMode="default",
            visibilityDistance=1.5,
            scene=scene_name,
            # step sizes
            gridSize=0.1,
            snapToGrid=False,
            rotateStepDegrees=30,
            # image modalities
            renderDepthImage=False,
            renderInstanceSegmentation=False,
            # camera properties
            width=300,
            height=300,
            fieldOfView=90,
        )
        self.goal_object_name = ""
        self.goal_object_type = ""
        self.goal_object_ID = ""
        self.default_receptacle_type = "CounterTop"
        self.action_language_templates = {
            "PickupObject": "Pick up the {objectType}",
            "PutObject": "Put the {heldObjectType} in the {objectType}",
            "PutObjectRecursive": "Put the {heldObjectType} on the {intermediateObjectType} in the {objectType}",
            "OpenObject": "Open the {objectType}",
            "CloseObject": "Close the {objectType}",
            "CookObject": "Cook the {objectType}",
            "SliceObject": "Slice the {objectType}",
            "ToggleObjectOn": "Turn on the {objectType}",
            "ToggleObjectOff": "Turn off the {objectType}",
            "BreakObject": "Break the {objectType}",
            "AddObject": "Add the {objectType} to the {goalObject}",
        }
        self.sanitize_words = {
            "LettuceSliced": "Sliced Lettuce",
            "TomatoSliced": "Sliced Tomato",
            "BreadSliced": "Sliced Bread",
            "StoveKnob": "Stove Knob",
            "StoveBurner": "Stove Burner",
            "CoffeeMachine": "Coffee Machine",
            "CellPhone": "Cell Phone",
            "ButterKnife": "Butter Knife",
            "GarbageCan": "Garbage Can",
            "PepperShaker": "Pepper Shaker",
            "CounterTop": "Counter Top",
        }
        self.get_obj_properties()
        self.current_task_dict = None

    def reset(self, scene_name=None):
        if scene_name is not None:
            self.scene_name = scene_name
        self.controller.reset(scene=self.scene_name)
        self.get_obj_properties()

    def close(self):
        self.controller.stop()

    def get_obj_properties(self):
        objects = self.controller.last_event.metadata["objects"]
        self.pickupable_objects = []
        self.receptacles = []
        self.openable_objects = []
        self.cookable_objects = []
        self.sliceable_objects = []
        self.toggleable_objects = []
        self.breakable_objects = []
        for obj in objects:
            if obj["pickupable"]:
                self.pickupable_objects.append(obj)
            if obj["receptacle"]:
                self.receptacles.append(obj)
            if obj["openable"]:
                self.openable_objects.append(obj)
            if obj["cookable"]:
                self.cookable_objects.append(obj)
            if obj["sliceable"]:
                self.sliceable_objects.append(obj)
            if obj["toggleable"]:
                self.toggleable_objects.append(obj)
            if obj["breakable"]:
                self.breakable_objects.append(obj)
        self.obj_property_dict = {
            "PickupObject": self.pickupable_objects,
            "PutObject": self.receptacles,
            "OpenObject": self.openable_objects,
            "CloseObject": self.openable_objects,
            "CookObject": self.cookable_objects,
            "SliceObject": self.sliceable_objects,
            "ToggleObjectOn": self.toggleable_objects,
            "ToggleObjectOff": self.toggleable_objects,
            "BreakObject": self.breakable_objects,
        }

    def get_all_object_types(self):
        object_types = []
        objects = self.controller.last_event.metadata["objects"]
        for obj in objects:
            object_types.append(obj["objectType"])
        object_types = list(set(object_types))
        return object_types

    def get_agent_pos(self):
        return pos_dict_to_array(
            self.controller.last_event.metadata["agent"]["position"]
        )

    def _pos_cost(self, pos_dict, object_location):
        pos_score = 0
        robot_angle = pos_dict["rotation"] / 180 * np.pi
        robot_angle = np.pi / 2 - robot_angle
        rob_to_obj_vec = object_location - pos_dict_to_array(pos_dict)
        # rob_to_obj_angle = np.arctan2(rob_to_obj_vec[2], rob_to_obj_vec[0])
        # rob_to_obj_angle = rob_to_obj_angle if rob_to_obj_angle >= 0 else 2*np.pi + rob_to_obj_angle
        # angle_diff = np.abs(robot_angle - rob_to_obj_angle)
        # if angle_diff > 2*np.pi:
        #     print(angle_diff)
        #     print(robot_angle, rob_to_obj_angle)
        #     1/0
        # # print(angle_diff)
        # angle_diff = angle_diff if angle_diff < np.pi else 2*np.pi - angle_diff
        robot_vec = np.array([np.cos(robot_angle), np.sin(robot_angle)])
        rob_to_obj_2d = np.array([rob_to_obj_vec[0], rob_to_obj_vec[2]])
        rob_to_obj_2d = rob_to_obj_2d / np.linalg.norm(rob_to_obj_2d)
        # print("Robot vec: ", robot_vec, "Rob to obj 2d: ", rob_to_obj_2d)
        angle_diff = np.arccos(np.dot(robot_vec, rob_to_obj_2d))
        pos_score += angle_diff * 10
        pos_score += np.linalg.norm(rob_to_obj_vec)
        pos_score += pos_dict["horizon"] * -1 + 60
        pos_score += 0 if pos_dict["standing"] else 100
        # print("Agent to obj vec: ", rob_to_obj_vec, "Agent rotation: ", pos_dict["rotation"], "Rotation cost: ", angle_diff, "Cost: ", pos_score)
        return pos_score

    def is_visible(self, objectID):
        objects = self.controller.last_event.metadata["objects"]
        for obj in objects:
            if obj["objectId"] == objectID:
                return obj["visible"]
        return False

    def get_object_distance(self, objectID):
        objects = self.controller.last_event.metadata["objects"]
        for obj in objects:
            if obj["objectId"] == objectID:
                return obj["distance"]
        return False

    def get_obj_in_frame(self, pos=(0.5, 0.5)):
        query = self.controller.step(
            action="GetObjectInFrame", x=pos[0], y=pos[1], checkVisible=False
        )

        object_id = query.metadata["actionReturn"]
        return object_id

    def move_to(self, x, z, theta, mode="teleport"):
        agent_pos_dict = self.controller.last_event.metadata["agent"]
        agent_pos_dict["position"]["x"] = x
        agent_pos_dict["position"]["z"] = z
        agent_pos_dict["rotation"]["y"] = theta
        self.move_to_dict(agent_pos_dict, mode)

    def move_to_dict(self, pos_dict, mode="teleport"):
        if mode == "teleport":
            event = self.controller.step("TeleportFull", **pos_dict)
            return event.metadata["lastActionSuccess"]
        elif mode == "navigate":
            raise NotImplementedError

    def obj_compare(self, query_obj, obj_list):
        for obj in obj_list:
            if query_obj["name"] == obj["name"]:
                return True
        return False

    def get_obj_by_ID(self, objectID):
        objects = self.controller.last_event.metadata["objects"]
        for obj in objects:
            if obj["objectId"] == objectID:
                return obj
        return None

    def get_closest_obj(
        self,
        object_name,
        penalize_objects_in_goal=False,
        penalize_objects_in_closed_receptacles=False,
    ):
        agent_position = pos_dict_to_array(
            self.controller.last_event.metadata["agent"]["position"]
        )
        event = self.controller.last_event
        objects = self.controller.last_event.metadata["objects"]
        obj_id = None
        obj_pos = None
        closest_dist = np.inf
        obj_found = False
        target_objects_in_goal = []
        target_objects_in_closed_receptacles = []
        for obj in objects:
            if obj["objectType"] == object_name:
                obj_found = True
                good_object = True
                if penalize_objects_in_goal:
                    objects_in_goal = self.get_obj_by_ID(self.goal_object_ID)[
                        "receptacleObjectIds"
                    ]
                    if obj["objectId"] in objects_in_goal:
                        good_object = False
                        target_objects_in_goal.append(obj["objectId"])
                if penalize_objects_in_closed_receptacles:
                    obj_parent_receptable = obj["parentReceptacles"]
                    if obj_parent_receptable == None:
                        obj_parent_receptable = []
                    for receptacle in obj_parent_receptable:
                        receptable_obj = self.get_obj_by_ID(receptacle)
                        # print("Object {} in receptacle: {}".format(obj["objectId"],receptable_obj["objectId"]))
                        if (
                            receptable_obj["openable"]
                            and receptable_obj["isOpen"] == False
                        ):
                            good_object = False
                            target_objects_in_closed_receptacles.append(obj["objectId"])
                            break
                if good_object:
                    if obj["distance"] < closest_dist:
                        closest_dist = obj["distance"]
                        obj_id = obj["objectId"]
                        obj_pos = pos_dict_to_array(obj["position"])
        if not obj_found:
            print("Error during search: object {} not found".format(object_name))
            return None
        if obj_id != None:
            return obj_id
        else:
            if len(target_objects_in_closed_receptacles) > 0:
                print(
                    "No valid objects found, penalizing objects in closed receptacles: ",
                    target_objects_in_closed_receptacles,
                )
                return target_objects_in_closed_receptacles[0]
            elif len(target_objects_in_goal) > 0:
                print(
                    "No valid objects found, penalizing objects in goal: ",
                    target_objects_in_goal,
                )
                return target_objects_in_goal[0]

    def move_to_obj(
        self,
        object_name,
        mode="closest",
        verbose=False,
        penalize_objects_in_goal=False,
        penalize_objects_in_closed_receptacles=False,
    ):
        if mode == "closest":
            agent_position = pos_dict_to_array(
                self.controller.last_event.metadata["agent"]["position"]
            )
            obj_id = self.get_closest_obj(
                object_name,
                penalize_objects_in_goal=penalize_objects_in_goal,
                penalize_objects_in_closed_receptacles=penalize_objects_in_closed_receptacles,
            )
            if not obj_id:
                print("Error during move to object: object not found")
                return False
            obj_pos = pos_dict_to_array(self.get_obj_by_ID(obj_id)["position"])

            # valid_bot_poses = self.controller.step(action="GetReachablePositions").metadata["actionReturn"]
            event = self.controller.step(
                action="GetInteractablePoses",
                objectId=obj_id,
                rotations=list(range(0, 360, 10)),
            )
            interactable_positions = event.metadata["actionReturn"]
            # if verbose:
            #     print(interactable_positions)
            if not interactable_positions:
                print("Error no interactable positions found")
                return False
            best_cost = np.inf
            pos_idx = 0
            for i, pos in enumerate(interactable_positions):
                cost = self._pos_cost(pos, obj_pos)
                if cost < best_cost:
                    # print("BEST COST: ", cost)
                    best_cost = cost
                    pos_idx = i
            success = self.move_to_dict(
                interactable_positions[pos_idx], mode="teleport"
            )
            event = self.controller.step("MoveAhead")

            return success

        elif mode == "specific":
            obj_id = object_name
            obj_pos = pos_dict_to_array(self.get_obj_by_ID(obj_id)["position"])
            event = self.controller.step(
                action="GetInteractablePoses",
                objectId=obj_id,
                rotations=list(range(0, 360, 10)),
            )
            interactable_positions = event.metadata["actionReturn"]
            # if verbose:
            #     print(interactable_positions)
            if not interactable_positions:
                print("Error no interactable positions found")
                return False
            best_cost = np.inf
            pos_idx = 0
            for i, pos in enumerate(interactable_positions):
                cost = self._pos_cost(pos, obj_pos)
                if cost < best_cost:
                    # print("BEST COST: ", cost)
                    best_cost = cost
                    pos_idx = i
            success = self.move_to_dict(
                interactable_positions[pos_idx], mode="teleport"
            )
            event = self.controller.step("MoveAhead")

            return success
        else:
            raise NotImplementedError

    def pickup_obj(
        self,
        object_name,
        mode="closest",
        verbose=False,
        penalize_objects_in_goal=False,
        penalize_objects_in_closed_receptacles=False,
    ):
        if mode == "closest":
            self.move_to_obj(object_name)
            agent_position = pos_dict_to_array(
                self.controller.last_event.metadata["agent"]["position"]
            )
            obj_id = self.get_closest_obj(
                object_name,
                penalize_objects_in_goal=penalize_objects_in_goal,
                penalize_objects_in_closed_receptacles=penalize_objects_in_closed_receptacles,
            )
            if not obj_id:
                print("Error during pickup: object not found")
                return False
        event = self.controller.step(
            action="PickupObject",
            objectId=obj_id,
            forceAction=True,
            manualInteract=False,
        )
        return event.metadata["lastActionSuccess"]

    def obj_interact(
        self,
        object_name,
        action,
        mode="specific",
        verbose=False,
        penalize_objects_in_goal=False,
        penalize_objects_in_closed_receptacles=False,
    ):
        obj_properties = self.obj_property_dict[action]
        if mode == "closest":
            self.move_to_obj(object_name)
            obj_id = self.get_closest_obj(
                object_name,
                penalize_objects_in_goal=penalize_objects_in_goal,
                penalize_objects_in_closed_receptacles=penalize_objects_in_closed_receptacles,
            )
            if not obj_id:
                print("Error during {}: object not found".format(action))
                return False
            event = self.controller.step(
                action=action, objectId=obj_id, forceAction=True
            )
            if not event.metadata["lastActionSuccess"]:
                print(
                    "Action {} on object {} failed due to: ".format(
                        action, object_name
                    ),
                    event.metadata["errorMessage"],
                )
                if (
                    action == "PutObject"
                    and "No valid positions to place object found"
                    in event.metadata["errorMessage"]
                ):
                    print("Attempting backup put")
                    try:
                        self.put_obj_backup(obj_id)
                    except:
                        print("Backup put failed")
                        return False
            return event.metadata["lastActionSuccess"]

        if mode == "specific":
            obj_id = object_name
            event = self.controller.step(
                action=action, objectId=obj_id, forceAction=True
            )
            if not event.metadata["lastActionSuccess"]:
                print(
                    "Action {} on object {} failed due to: ".format(
                        action, object_name
                    ),
                    event.metadata["errorMessage"],
                )
                if (
                    action == "PutObject"
                    and "No valid positions to place object found"
                    in event.metadata["errorMessage"]
                ):
                    print("Attempting backup put")
                    try:
                        self.put_obj_backup(obj_id)
                    except:
                        print("Backup put failed")
                        return False
            return event.metadata["lastActionSuccess"]
        else:
            raise NotImplementedError

    def add_object(
        self,
        object_name,
        goal_object_name,
        mode="closest",
        verbose=False,
        penalize_objects_in_goal=False,
        penalize_objects_in_closed_receptacles=False,
    ):
        self.move_to_obj(
            object_name,
            mode=mode,
            verbose=verbose,
            penalize_objects_in_goal=penalize_objects_in_goal,
            penalize_objects_in_closed_receptacles=penalize_objects_in_closed_receptacles,
        )
        success = self.pickup_obj(
            object_name,
            mode=mode,
            verbose=verbose,
            penalize_objects_in_goal=penalize_objects_in_goal,
            penalize_objects_in_closed_receptacles=penalize_objects_in_closed_receptacles,
        )
        if not success:
            return False
        self.move_to_obj(
            goal_object_name,
            mode="specific",
            verbose=verbose,
            penalize_objects_in_goal=penalize_objects_in_goal,
            penalize_objects_in_closed_receptacles=penalize_objects_in_closed_receptacles,
        )
        success = self.obj_interact(
            goal_object_name,
            "PutObject",
            mode="specific",
            verbose=verbose,
            penalize_objects_in_goal=penalize_objects_in_goal,
            penalize_objects_in_closed_receptacles=penalize_objects_in_closed_receptacles,
        )
        if not success:
            print(
                "Error during add object: put failed, leaving object on {}".format(
                    self.default_receptacle_type
                )
            )
            self.obj_interact(
                self.default_receptacle_type,
                "PutObject",
                mode="closest",
                verbose=verbose,
                penalize_objects_in_goal=penalize_objects_in_goal,
                penalize_objects_in_closed_receptacles=penalize_objects_in_closed_receptacles,
            )
            return False
        return success

    def put_obj_backup(self, objectId):

        event = self.controller.step(
            action="GetSpawnCoordinatesAboveReceptacle",
            objectId=objectId,
            anywhere=False,
        )
        interactable_positions = event.metadata["actionReturn"]
        # print(interactable_positions)
        if not interactable_positions:
            print("Error no interactable positions found for backup put")
            return False
        obj_pos = np.array([float(x) for x in objectId.split("|")[1:]])
        distances = [
            np.linalg.norm(pos_dict_to_array(pos) - obj_pos)
            for pos in interactable_positions
        ]
        sorted_indices = np.argsort(distances)
        success = False
        for i in sorted_indices:
            best_position = interactable_positions[i]
            # print(best_position)
            event = self.controller.step(
                action="PlaceObjectAtPoint", objectId=objectId, position=best_position
            )
            success = event.metadata["lastActionSuccess"]
            if success:
                break
            else:
                print(
                    "Backup put failed at position: {} due to: {}".format(
                        best_position, event.metadata["errorMessage"]
                    )
                )
        return success

    def get_last_event(self):
        return self.controller.last_event

    def sanitize_object_names(self, action_dict):
        action_dict = action_dict.copy()
        for key in action_dict:
            if (
                key == "objectType"
                or key == "heldObjectType"
                or key == "intermediateObjectType"
            ):
                if action_dict[key] in self.sanitize_words:
                    action_dict[key] = self.sanitize_words[action_dict[key]]
        return action_dict

    def generate_possible_actions(
        self, return_language_tags=False, remove_duplicates=True
    ):
        self.get_obj_properties()

        possible_actions = []
        objects = self.controller.last_event.metadata["objects"]
        agent = self.controller.last_event.metadata["agent"]
        held_object = None
        for obj in objects:
            if obj["isPickedUp"]:
                held_object = obj

        # PickupObject
        if held_object is None:
            for obj in self.pickupable_objects:
                possible_actions.append(
                    {
                        "action": "PickupObject",
                        "objectId": obj["objectId"],
                        "objectType": obj["objectType"],
                    }
                )
            for obj in self.pickupable_objects:
                possible_actions.append(
                    {
                        "action": "AddObject",
                        "objectId": obj["objectId"],
                        "objectType": obj["objectType"],
                        "goalObject": self.goal_object_name,
                        "goalObjectId": self.goal_object_ID,
                    }
                )

        # PutObject
        if held_object is not None:
            for obj in self.receptacles:
                possible_actions.append(
                    {
                        "action": "PutObject",
                        "heldObject": held_object["objectId"],
                        "objectId": obj["objectId"],
                        "heldObjectType": held_object["objectType"],
                        "objectType": obj["objectType"],
                    }
                )
                receptacle_object_IDs = [
                    objectID for objectID in obj["receptacleObjectIds"]
                ]
                # print("Receptacle IDS for {}:".format(obj["objectType"]), receptacle_object_IDs)
                receptacle_objects = []
                for search_obj in objects:
                    if search_obj["objectId"] in receptacle_object_IDs:
                        receptacle_objects.append(search_obj)
                for receptacle_obj in receptacle_objects:
                    possible_actions.append(
                        {
                            "action": "PutObjectRecursive",
                            "heldObject": held_object["objectId"],
                            "objectId": obj["objectId"],
                            "heldObjectType": held_object["objectType"],
                            "objectType": obj["objectType"],
                            "intermediateObjectType": receptacle_obj["objectType"],
                        }
                    )

        # OpenObject
        for obj in self.openable_objects:
            if obj["isOpen"]:
                possible_actions.append(
                    {
                        "action": "CloseObject",
                        "objectId": obj["objectId"],
                        "objectType": obj["objectType"],
                    }
                )
            else:
                possible_actions.append(
                    {
                        "action": "OpenObject",
                        "objectId": obj["objectId"],
                        "objectType": obj["objectType"],
                    }
                )
        # CookObject
        for obj in self.cookable_objects:
            if not obj["isCooked"]:
                possible_actions.append(
                    {
                        "action": "CookObject",
                        "objectId": obj["objectId"],
                        "objectType": obj["objectType"],
                    }
                )
        # SliceObject
        for obj in self.sliceable_objects:
            if not obj["isSliced"]:
                possible_actions.append(
                    {
                        "action": "SliceObject",
                        "objectId": obj["objectId"],
                        "objectType": obj["objectType"],
                    }
                )
        # ToggleObjectOn
        for obj in self.toggleable_objects:
            if not obj["isToggled"]:
                possible_actions.append(
                    {
                        "action": "ToggleObjectOn",
                        "objectId": obj["objectId"],
                        "objectType": obj["objectType"],
                    }
                )
            else:
                possible_actions.append(
                    {
                        "action": "ToggleObjectOff",
                        "objectId": obj["objectId"],
                        "objectType": obj["objectType"],
                    }
                )
        # BreakObject
        for obj in self.breakable_objects:
            if not obj["isBroken"]:
                possible_actions.append(
                    {
                        "action": "BreakObject",
                        "objectId": obj["objectId"],
                        "objectType": obj["objectType"],
                    }
                )
        if return_language_tags:
            action_language_tags = [
                self.action_language_templates[action["action"]].format(
                    **self.sanitize_object_names(action)
                )
                for action in possible_actions
            ]
            if remove_duplicates:
                new_actions = []
                new_language_tags = []
                for action, language_tag in zip(possible_actions, action_language_tags):
                    if language_tag not in new_language_tags:
                        new_actions.append(action)
                        new_language_tags.append(language_tag)
            return new_actions, new_language_tags
        else:
            return possible_actions

    def parse_action(self, action, mode="closest"):
        if mode == "closest":
            if action["action"] == "PickupObject":
                return self.pickup_obj(
                    action["objectType"],
                    mode=mode,
                    verbose=False,
                    penalize_objects_in_goal=True,
                    penalize_objects_in_closed_receptacles=True,
                )
            elif action["action"] == "PutObjectRecursive":
                return self.obj_interact(
                    action["objectType"],
                    "PutObject",
                    mode=mode,
                    verbose=False,
                    penalize_objects_in_goal=True,
                    penalize_objects_in_closed_receptacles=True,
                )
            elif action["action"] == "AddObject":
                return self.add_object(
                    action["objectType"],
                    self.goal_object_ID,
                    mode=mode,
                    verbose=False,
                    penalize_objects_in_goal=True,
                    penalize_objects_in_closed_receptacles=True,
                )
            else:
                return self.obj_interact(
                    action["objectType"],
                    action["action"],
                    mode=mode,
                    verbose=False,
                    penalize_objects_in_goal=True,
                    penalize_objects_in_closed_receptacles=True,
                )
        else:
            raise NotImplementedError

    def check_success(self):
        if self.current_task_dict is None:
            print("Error: no task loaded")
            return False
        objects = self.controller.last_event.metadata["objects"]
        success_conditions = self.current_task_dict["thor_object_success_condition"]
        success = True
        for condition in success_conditions:
            condition_success = False
            for obj in objects:
                if obj["objectType"] == condition["object"]:
                    # print("Checking object",obj["objectId"])
                    if condition["relation"] == "receptacle_contains":
                        receptacle_object_types = [
                            objectID.split("|")[0]
                            for objectID in obj["receptacleObjectIds"]
                        ]
                        # print("contained objects: ",receptacle_object_types)
                        if set(condition["arguments"]) <= set(receptacle_object_types):
                            # print("Success condition met: ",condition)
                            condition_success = True
                            break
                    elif condition["relation"] == "sliced":
                        condition_success = condition["arguments"][0] == obj["isSliced"]
                        if condition_success:
                            # print("Success condition met: ",condition)
                            break
                    else:
                        print("Error: invalid success condition")
            if not condition_success:
                success = False
                break
        return success

    def generate_language_predicates(self):
        objects = self.controller.last_event.metadata["objects"]
        predicates = []
        for obj in objects:
            if obj["isPickedUp"]:
                predicates.append("currently holding {}".format(obj["objectType"]))
            if obj["isCooked"]:
                predicates.append("{} is cooked".format(obj["objectType"]))
            if obj["isSliced"]:
                predicates.append("{} is sliced".format(obj["objectType"]))
            if obj["isToggled"]:
                predicates.append("{} is on".format(obj["objectType"]))
            if obj["isBroken"]:
                predicates.append("{} is broken".format(obj["objectType"]))
            if obj["isOpen"]:
                predicates.append("{} is open".format(obj["objectType"]))
            if (
                self.obj_compare(obj, self.receptacles)
                and len(obj["receptacleObjectIds"]) > 0
            ):
                receptacle_object_types = [
                    objectID.split("|")[0] for objectID in obj["receptacleObjectIds"]
                ]
                predicates.append(
                    "{} contains: {}".format(
                        obj["objectType"], ", ".join(receptacle_object_types)
                    )
                )
        return predicates

    def step(self, action):
        success = self.parse_action(action)
        objects_info = self.controller.last_event.metadata["objects"]
        agent_info = self.controller.last_event.metadata["agent"]
        return success, objects_info, agent_info

    def load_task_state(self, task_dict, action_index):
        # print(action_index, task_dict["valid_start_indexes"])
        assert action_index in task_dict["valid_start_indexes"]
        self.goal_object_name = task_dict["goal_object_name"]
        self.goal_object_type = task_dict["goal_object_type"]
        self.goal_object_ID = self.get_closest_obj(
            self.goal_object_type,
            penalize_objects_in_goal=False,
            penalize_objects_in_closed_receptacles=True,
        )

        self.current_task_dict = task_dict
        pre_task_actions = task_dict["pre_task_actions"]
        for action in pre_task_actions:
            print(action["action"], action["objectType"])
            success = self.parse_action(action)
            if not success:
                print(
                    "Error loading task state, pre task action {} failed".format(
                        action["action"]
                    )
                )
                return False
        action_segments = task_dict["action_segments"]
        for i in range(action_index):
            if action_segments[i]["thor_object"]:
                actionLabel = action_segments[i]["action"]
                objectType = action_segments[i]["object"]
                action = {"action": actionLabel, "objectType": objectType}
                print(actionLabel, objectType)
                success = self.parse_action(action)
                if not success:
                    print(
                        "Error loading task state, action {} object {} at index {} failed".format(
                            action["action"], action["objectType"], i
                        )
                    )
                    return False

    def set_gt_history(self):
        if self.current_task_dict is None:
            print("Error: no task loaded")
            return None
        action_segments = self.current_task_dict["action_segments"]
        self.history = [segment["action_text"] for segment in action_segments]

    def set_narration_dict(self, narration_dict):
        self.history = [element["narrations"] for element in narration_dict]

    def get_history(self):
        return self.history


def find_compatible_environments(task_dict, scene_nums):
    valid_scenes = []
    necessary_objects = task_dict["thor_objects"]
    for scene_num in scene_nums:
        scene_name = "FloorPlan" + str(scene_num)
        print(scene_name)
        env = CookingEnv(scene_name=scene_name)
        object_types = env.get_all_object_types()
        valid_scene = True
        for obj in necessary_objects:
            if obj not in object_types:
                valid_scene = False
                break
        last_valid_index = task_dict["valid_start_indexes"][-1]
        env.load_task_state(task_dict, last_valid_index)
        valid_scene = env.check_success()

        if valid_scene:
            valid_scenes.append(scene_num)
            print("Scene {} is compatible".format(scene_name))
        env.close()
    return valid_scenes


def setup_environment(scene_name, task_name, start_index, task_version=0, history="gt"):
    env = CookingEnv(scene_name=scene_name)
    task_dict_path = TASK_PATHS[task_name][task_version]
    # task_dict_path = "Tasks/Make_A_BLT_0.json"
    with open(task_dict_path, "r") as file:
        task_dict = json.load(file)
    env.load_task_state(task_dict, start_index)
    if history == "gt":
        env.set_gt_history()
    else:
        with open(history, "r") as file:
            narration_dict = json.load(file)
        env.set_narration_dict(narration_dict)
    return env


def main():
    task_dict_path = "Tasks/Make_A_BLT_0_abridged.json"
    with open(task_dict_path, "r") as file:
        task_dict = json.load(file)
    # scene_nums = list(range(1,31))
    scene_nums = [31]
    valid_scenes = find_compatible_environments(task_dict, scene_nums)
    print(valid_scenes)

    successful_floorplans = [1, 2, 3, 8, 9, 11, 12, 14, 16, 22, 23, 28, 29, 30]
    # task_dict_path = "Tasks/Make_A_BLT_0.json"
    # with open(task_dict_path, 'r') as file:
    # 	task_dict = json.load(file)
    # scene_nums = list(range(1,31))
    # valid_scenes = find_compatible_environments(task_dict,scene_nums)
    # print(valid_scenes)
    env = setup_environment("FloorPlan2", "make a blt", 8)
    actions, action_language_tags = env.generate_possible_actions(
        return_language_tags=True
    )
    # print(env.generate_language_predicates())
    print(action_language_tags)
    # success = env.check_success()
    # print(success)

    env = CookingEnv(scene_name="FloorPlan2")
    print(env.generate_language_predicates())
    task_dict_path = "Tasks/Make_A_BLT_0.json"
    with open(task_dict_path, "r") as file:
        task_dict = json.load(file)
    env.load_task_state(task_dict, 18)
    print("Task loaded")
    env.move_to_obj("Plate")
    time.sleep(10)

    print(env.generate_language_predicates())
    print(env.check_success())
    # print(env.get_all_object_types())

    for obj in env.controller.last_event.metadata["objects"]:
        print(obj["objectType"], obj["objectId"])
    print(env.get_agent_pos())
    success = env.move_to_obj("Pot", verbose=True)
    print(success)
    print(env.get_agent_pos())
    # time.sleep(5)
    success = env.pickup_obj("Pot", verbose=True)
    print(success)
    print(env.get_agent_pos())
    possible_actions, action_language_tags = env.generate_possible_actions(
        return_language_tags=True
    )
    print(len(possible_actions))
    print(action_language_tags)
    print("Rotating")
    time.sleep(10)
    for i in range(100):
        event = env.controller.step(action="RotateRight")
        print(env.is_visible("Sink|-01.90|+00.97|-01.50"))
        print(env.get_object_distance("Sink|-01.90|+00.97|-01.50"))
        print(env.get_obj_in_frame((0.5, 0.3)))
        # print(env.is_visible("HousePlant|-01.95|+00.89|-02.52"))
        # print(env.get_object_distance("HousePlant|-01.95|+00.89|-02.52"))

        time.sleep(0.5)


if __name__ == "__main__":
    main()
