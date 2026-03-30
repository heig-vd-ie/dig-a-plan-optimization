import argparse
from experiments import *


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--grid",
        choices=[
            "boisy",
            "feeder_1",
            "feeder_2",
            "aumont",
            "autoroutes",
            "bel-air",
            "centre_ville",
            "st-aubin",
            "tout_vent",
            "zone_industrielle",
        ],
        required=True,
        help="Select which experiment variant to run",
    )

    parser.add_argument(
        "--scenario",
        choices=["Basic", "Sustainable", "Full"],
        required=True,
        help="Select which experiment variant to run",
    )

    return parser.parse_args()


def apply_args(payload: dict, grid, scenario):
    payload["profiles"]["scenario_name"] = scenario
    payload["grid"]["name"] = grid + "-" + scenario

    match grid:
        case "boisy":
            payload["grid"]["pp_file"] = ".cache/input/boisy/boisy_grid.p"
            payload["profiles"]["load_profiles"] = [
                ".cache/input/boisy/load_profiles/feeder_1",
                ".cache/input/boisy/load_profiles/feeder_2",
            ]
        case "feeder_1":
            payload["grid"]["pp_file"] = ".cache/input/boisy/feeders/feeder_feeder_1.p"
            payload["profiles"]["load_profiles"] = [
                ".cache/input/boisy/load_profiles/feeder_1"
            ]
        case "feeder_2":
            payload["grid"]["pp_file"] = ".cache/input/boisy/feeders/feeder_feeder_2.p"
            payload["profiles"]["load_profiles"] = [
                ".cache/input/boisy/load_profiles/feeder_2"
            ]
        case "aumont":
            payload["grid"][
                "pp_file"
            ] = ".cache/input/estavayer/feeders/feeder_aumont.p"
            payload["profiles"]["load_profiles"] = [
                ".cache/input/estavayer/load_profiles/aumont"
            ]
        case "autoroutes":
            payload["grid"][
                "pp_file"
            ] = ".cache/input/estavayer/feeders/feeder_autoroutes.p"
            payload["profiles"]["load_profiles"] = [
                ".cache/input/estavayer/load_profiles/autoroutes"
            ]
        case "bel-air":
            payload["grid"][
                "pp_file"
            ] = ".cache/input/estavayer/feeders/feeder_bel-air.p"
            payload["profiles"]["load_profiles"] = [
                ".cache/input/estavayer/load_profiles/bel-air"
            ]
        case "centre_ville":
            payload["grid"][
                "pp_file"
            ] = ".cache/input/estavayer/feeders/feeder_centre_ville.p"
            payload["profiles"]["load_profiles"] = [
                ".cache/input/estavayer/load_profiles/centre_ville"
            ]
        case "st-aubin":
            payload["grid"][
                "pp_file"
            ] = ".cache/input/estavayer/feeders/feeder_st-aubin.p"
            payload["profiles"]["load_profiles"] = [
                ".cache/input/estavayer/load_profiles/st-aubin"
            ]
        case "tout_vent":
            payload["grid"][
                "pp_file"
            ] = ".cache/input/estavayer/feeders/feeder_tout_vent.p"
            payload["profiles"]["load_profiles"] = [
                ".cache/input/estavayer/load_profiles/tout_vent"
            ]
        case "zone_industrielle":
            payload["grid"][
                "pp_file"
            ] = ".cache/input/estavayer/feeders/feeder_zone_industrielle.p"
            payload["profiles"]["load_profiles"] = [
                ".cache/input/estavayer/load_profiles/zone_industrielle"
            ]
        case _:
            raise ValueError("Wrong grid name")

    return payload


if __name__ == "__main__":
    args = parse_args()

    payload_file = PROJECT_ROOT / "experiments/expansion_benchmark/00-settings.json"
    payload = json.load(open(payload_file, "r"))

    payload = apply_args(payload, args.grid, args.scenario)

    response = requests.patch(
        f"http://{LOCALHOST}:{PY_PORT}/expansion/benchmark",
        json=payload,
    )
    print("Response status code:", response.status_code)
    print("Response JSON:", response.json())
