from ai2thor.controller import Controller
import time
import LaViLa_Interface
import numpy as np


narrator = LaViLa_Interface.LaViLa_Interface(load_nar = True, load_dual = False)

controller = Controller(
    agentMode="default",
    visibilityDistance=1.5,
    scene="FloorPlan1",

    # step sizes
    gridSize=0.25,
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
event = controller.step("MoveAhead")
for obj in controller.last_event.metadata["objects"]:
    print(obj["objectType"])
for i in range(100):
	event = controller.step(action='RotateRight')
	time.sleep(.5)
