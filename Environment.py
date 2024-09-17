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


def pos_dict_to_array(pos_dict):
    return np.array([pos_dict["x"], pos_dict["y"], pos_dict["z"]])


class CookingEnv:
    def __init__(self,scene_name = "FloorPlan1"):
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
            fieldOfView=90
        )
        self.action_language_templates = {
            "PickupObject": "Pick up the {objectType}",
            "PutObject": "Put the {heldObjectType} in the {objectType}",
            "OpenObject": "Open the {objectType}",
            "CloseObject": "Close the {objectType}",
            "CookObject": "Cook the {objectType}",
            "SliceObject": "Slice the {objectType}",
            "ToggleObjectOn": "Turn on the {objectType}",
            "ToggleObjectOff": "Turn off the {objectType}",
            "BreakObject": "Break the {objectType}",
            }
        self.get_obj_properties()


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
            "BreakObject": self.breakable_objects
        }
    def get_all_object_types(self):
        object_types = []
        objects = self.controller.last_event.metadata["objects"]
        for obj in objects:
            object_types.append(obj["objectType"])
        object_types = list(set(object_types))
        return object_types

    def get_agent_pos(self):
        return pos_dict_to_array(self.controller.last_event.metadata["agent"]["position"])

    def _pos_cost(self,pos_dict,object_location):
        pos_score = 0
        pos_score += np.linalg.norm(pos_dict_to_array(pos_dict) - object_location)
        pos_score += pos_dict["horizon"]*-1 + 60
        pos_score += 0 if pos_dict["standing"] else 100
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

    def get_obj_in_frame(self, pos = (.5,.5)):
        query = self.controller.step(
            action="GetObjectInFrame",
            x=pos[0],
            y=pos[1],
            checkVisible=False
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

    def move_to_obj(self, object_name, mode="closest", verbose=False):
        if mode == "closest":
            agent_position = pos_dict_to_array(self.controller.last_event.metadata["agent"]["position"])
            event = self.controller.last_event
            objects = self.controller.last_event.metadata["objects"]
            obj_id = None
            obj_pos = None
            closest_dist = np.inf
            obj_found = False
            for obj in objects:
                if obj["objectType"] == object_name:
                    obj_found = True
                    if verbose:
                        print(obj["objectId"])
                    # dist = np.linalg.norm(pos_dict_to_array(obj["position"]) - agent_position)
                    if obj["distance"] < closest_dist:
                        closest_dist = obj["distance"]
                        obj_id = obj["objectId"]
                        obj_pos = pos_dict_to_array(obj["position"])
            if not obj_found:
                print("Error object not found")
                return False

            # valid_bot_poses = self.controller.step(action="GetReachablePositions").metadata["actionReturn"]
            event = self.controller.step(
                action="GetInteractablePoses",
                objectId=obj_id,
            )
            interactable_positions = event.metadata["actionReturn"]
            # if verbose:
            #     print(interactable_positions)
            if len(interactable_positions) == 0:
                print("Error no interactable positions found")
                return False
            best_cost = np.inf
            pos_idx = 0
            for i, pos in enumerate(interactable_positions):
                cost = self._pos_cost(pos, obj_pos)
                if cost < best_cost:
                    best_cost = cost
                    pos_idx = i
            success = self.move_to_dict(interactable_positions[pos_idx], mode="teleport")
            event = self.controller.step("MoveAhead")

            return(success)
        else:
            raise NotImplementedError

    def pickup_obj(self, object_name, mode="closest", verbose=False):
        if mode == "closest":
            agent_position = pos_dict_to_array(self.controller.last_event.metadata["agent"]["position"])
            event = self.controller.last_event
            objects = self.controller.last_event.metadata["objects"]
            obj_id = None
            obj_pos = None
            closest_dist = np.inf
            obj_found = False
            for obj in objects:
                if obj["objectType"] == object_name and obj["pickupable"]:
                    obj_found = True
                    if verbose:
                        print(obj["objectId"])
                    # dist = np.linalg.norm(pos_dict_to_array(obj["position"]) - agent_position)
                    if obj["distance"] < closest_dist:
                        closest_dist = obj["distance"]
                        obj_id = obj["objectId"]
                        obj_pos = pos_dict_to_array(obj["position"])
            if not obj_found:
                print("Error object not found")
                return False  
        event = self.controller.step(action='PickupObject', objectId=obj_id, forceAction = True, manualInteract=False)
        return event.metadata["lastActionSuccess"]

    def obj_interact(self,object_name,action,mode="closest",verbose=False):
        obj_properties = self.obj_property_dict[action]
        if mode == "closest":
            agent_position = pos_dict_to_array(self.controller.last_event.metadata["agent"]["position"])
            event = self.controller.last_event
            objects = self.controller.last_event.metadata["objects"]
            obj_id = None
            obj_pos = None
            closest_dist = np.inf
            obj_found = False
            for obj in objects:
                if obj["objectType"] == object_name and obj in obj_properties:
                    obj_found = True
                    if verbose:
                        print(obj["objectId"])
                    # dist = np.linalg.norm(pos_dict_to_array(obj["position"]) - agent_position)
                    if obj["distance"] < closest_dist:
                        closest_dist = obj["distance"]
                        obj_id = obj["objectId"]
                        obj_pos = pos_dict_to_array(obj["position"])
            if not obj_found:
                print("Error object not found")
                return False
            event = self.controller.step(action=action, objectId=obj_id, forceAction = True)
            return event.metadata["lastActionSuccess"]
        else:
            raise NotImplementedError

    def generate_possible_actions(self,return_language_tags = False):
        
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
                possible_actions.append({
                    "action": "PickupObject",
                    "objectId": obj["objectId"],
                    "objectType": obj["objectType"]
                })
        # PutObject
        if held_object is not None:
            for obj in self.receptacles:
                possible_actions.append({
                    "action": "PutObject",
                    "heldObject": held_object["objectId"],
                    "objectId": obj["objectId"],
                    "heldObjectType": held_object["objectType"],
                    "objectType": obj["objectType"]
                })
        # OpenObject
        for obj in self.openable_objects:
            if obj["isOpen"]:
                possible_actions.append({
                    "action": "CloseObject",
                    "objectId": obj["objectId"],
                    "objectType": obj["objectType"]
                })
            else:
                possible_actions.append({
                    "action": "OpenObject",
                    "objectId": obj["objectId"],
                    "objectType": obj["objectType"]
                })
        # CookObject
        for obj in self.cookable_objects:
            if not obj["isCooked"]:
                possible_actions.append({
                    "action": "CookObject",
                    "objectId": obj["objectId"],
                    "objectType": obj["objectType"]
                })
        # SliceObject
        for obj in self.sliceable_objects:
            if not obj["isSliced"]:
                possible_actions.append({
                    "action": "SliceObject",
                    "objectId": obj["objectId"],
                    "objectType": obj["objectType"]
                })
        # ToggleObjectOn
        for obj in self.toggleable_objects:
            if not obj["isToggled"]:
                possible_actions.append({
                    "action": "ToggleObjectOn",
                    "objectId": obj["objectId"],
                    "objectType": obj["objectType"]
                })
            else:
                possible_actions.append({
                    "action": "ToggleObjectOff",
                    "objectId": obj["objectId"],
                    "objectType": obj["objectType"]
                })
        # BreakObject
        for obj in self.breakable_objects:
            if not obj["isBroken"]:
                possible_actions.append({
                    "action": "BreakObject",
                    "objectId": obj["objectId"],
                    "objectType": obj["objectType"]
                })
        if return_language_tags:
            action_language_tags = [self.action_language_templates[action["action"]].format(**action) for action in possible_actions]
            return possible_actions, action_language_tags
        else:
            return possible_actions

    def parse_action(self,action,mode="closest"):
        if mode == "closest":
            if action["action"] == "PickupObject":
                return self.pickup_obj(action["objectType"])
            elif action["action"] == "PutObject":
                return self.obj_interact(action["objectType"],action["action"])
            else:
                return self.obj_interact(action["objectType"],action["action"])
        else:
            raise NotImplementedError

    def check_success(self,success_dict):
        objects = self.controller.last_event.metadata["objects"]

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
            if obj in self.receptacles and len(obj["receptacleObjectIds"]) > 0:
                receptacle_object_types = [objectID.split("|")[0] for objectID in obj["receptacleObjectIds"]]
                predicates.append("{} contains: {}".format(obj["objectType"], ", ".join(receptacle_object_types)))
        return predicates


    def step(self,action):
        success = self.parse_action(action)
        objects = self.controller.last_event.metadata["objects"]
        agent = self.controller.last_event.metadata["agent"]
        return success, objects, agent



def find_compatible_environments(task_dict,scene_nums):
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
        if valid_scene:
            valid_scenes.append(scene_num)
    return valid_scenes

            



def main():
    # task_path = "Tasks/Make_A_BLT.json"
    # with open(task_path, 'r') as file:
    #     task_dict = json.load(file)

    # scene_nums = list(range(1,31))
    # valid_scenes = find_compatible_environments(task_dict,scene_nums)
    # print("Make a BLT valid scenes: ",valid_scenes)
    # task_path = "Tasks/Make_A_Latte.json"
    # with open(task_path, 'r') as file:
    #     task_dict = json.load(file)
    # valid_scenes = find_compatible_environments(task_dict,scene_nums)
    # print("Make a Latte valid scenes: ",valid_scenes)
    # 1/0
    env = CookingEnv()
    print(env.generate_language_predicates())
    valid_actions = env.generate_possible_actions()
    for action in valid_actions:
        if action["action"] == "OpenObject" and action["objectType"] == "Cabinet":
            selected_action = action
            break
    success, _, _ = env.step(selected_action)
    print(success)
    print(env.generate_language_predicates())
    # print(env.get_all_object_types())

    1/0
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
    possible_actions, action_language_tags = env.generate_possible_actions(return_language_tags=True)
    print(len(possible_actions))
    print(action_language_tags)
    print("Rotating")
    time.sleep(10)
    for i in range(100):
      event = env.controller.step(action='RotateRight')
      print(env.is_visible("Sink|-01.90|+00.97|-01.50"))
      print(env.get_object_distance("Sink|-01.90|+00.97|-01.50"))
      print(env.get_obj_in_frame((.5,.3)))
      # print(env.is_visible("HousePlant|-01.95|+00.89|-02.52"))
      # print(env.get_object_distance("HousePlant|-01.95|+00.89|-02.52"))

      time.sleep(.5)

if __name__ == '__main__':
    main()