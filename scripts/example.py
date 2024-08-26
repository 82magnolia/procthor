from procthor.generation import PROCTHOR10K_ROOM_SPEC_SAMPLER, HouseGenerator
from ai2thor.controller import Controller
import copy
from PIL import Image
from matplotlib import pyplot as plt
import argparse
import os


def get_top_down_frame(controller: Controller):
    # Setup the top-down camera
    event = controller.step(action="GetMapViewCameraProperties", raise_for_failure=True)
    pose = copy.deepcopy(event.metadata["actionReturn"])

    bounds = event.metadata["sceneBounds"]["size"]
    max_bound = max(bounds["x"], bounds["z"])

    pose["fieldOfView"] = 50
    pose["position"]["y"] += 1.1 * max_bound
    pose["orthographic"] = False
    pose["farClippingPlane"] = 50
    del pose["orthographicSize"]

    # add the camera to the scene
    event = controller.step(
        action="AddThirdPartyCamera",
        **pose,
        skyboxColor="white",
        raise_for_failure=True,
    )
    top_down_frame = event.third_party_camera_frames[-1]
    top_down_img = Image.fromarray(top_down_frame)

    plt.imshow(top_down_img)
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_dir", help="Log directory for saving experiment results", default="./log/")
    parser.add_argument("--seed", help="Seed values for experimenting scene generation", default=0, type=int)
    parser.add_argument("--num_scenes", help="Number of scenes to generate", default=1, type=int)
    args = parser.parse_args()

    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir, exist_ok=True)

    house_generator = HouseGenerator(
        split='train',
        room_spec_sampler=PROCTHOR10K_ROOM_SPEC_SAMPLER,
        seed=args.seed
    )

    for _ in range(args.num_scenes):
        house, _ = house_generator.sample()
        house.validate(house_generator.controller)
        controller = house_generator.controller
        house.to_json(os.path.join(args.log_dir, "temp.json"))

        get_top_down_frame(controller)
