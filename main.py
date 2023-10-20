from dataclasses import dataclass
from pathlib import Path
import csv
from typing import Tuple, Dict, List
import json

from geopy.geocoders import Nominatim
# https://geopy.readthedocs.io/en/stable/#module-geopy.distance
from geopy.distance import distance
from geopy.exc import GeocoderUnavailable

import matplotlib.pyplot as plt
import networkx as nx

GEOLOCATOR = Nominatim(user_agent="supercharger_locator")
# https://www.tesla.com/model3
TESLA_RANGE = 535   # kilometers

@dataclass
class SuperchargerData:
    station_name: str
    city: str
    state: str
    zip_code: int
    street_address: str

    @property
    def lat_long(self) -> Tuple[float, float]:
        # From https://geopy.readthedocs.io/en/stable/#module-geopy.geocoders
        location = GEOLOCATOR.geocode(self.street_address)
        return (location.latitude, location.longitude)

    @property
    def as_dict(self) -> Dict:
        dict = self.__dict__
        dict.update({"lat_long": self.lat_long})
        return dict


def convert_csv_to_json():
    json_data = {"data": []}
    data_headers = None
    with open(str(Path(__file__).parent / "data/tesla_superchargers.csv")) as file:
        csvreader = csv.reader(file, delimiter=",")
        for row in csvreader:
            # Get name of each column from CSV (first row)
            if not data_headers:
                data_headers = row
                continue

            charger_data = SuperchargerData(
                *{data_headers[i]: row[i] for i in range(len(row))}
            )

            # Catching 2 Different errors here:
            #   1. Attribute errors for when GeoPy fails to find a location for the inputted address
            #   2. GeocoderUnavailable for when GeoPy times out when searching for the inputted location
            try:
                json_data["data"].append(charger_data.as_dict)
            except (AttributeError, GeocoderUnavailable):
                continue

    with open(
        str(Path(__file__).parent / "data/tesla_superchargers.json"), "w"
    ) as file:
        json.dump(json_data, file, indent=4)


def check_lat_long(lat_long: List[float]):
    return lat_long[0] > 20 and lat_long[0] < 50 and lat_long[1] < -60 and lat_long[1] > -130


def construct_graph():
    g = nx.Graph()

    with open(Path(__file__).parent / "data/tesla_superchargers.json", "r") as file:
        data = json.load(file)["data"]
    for i in range(len(data)):
        if check_lat_long(data[i]["lat_long"]):
            g.add_node(data[i]["station_name"], lat_long=(data[i]["lat_long"][1], data[i]["lat_long"][0]))
    for i in range(len(data)):
        for j in range(i+1, len(data)):
            station_1 = data[i]
            station_2 = data[j]
            if not check_lat_long(station_1["lat_long"]) or not check_lat_long(station_2["lat_long"]):
                continue
            
            d = int(distance(station_1["lat_long"], station_2["lat_long"]).km)
            if d < TESLA_RANGE:
                g.add_edge(station_1["station_name"], station_2["station_name"], weight=d)

    edges = [(u, v) for (u, v, d) in g.edges(data=True)]
        
    pos = nx.get_node_attributes(g, "lat_long")
        
    # nodes
    nx.draw_networkx_nodes(g, pos, node_size=7)

    # edges
    nx.draw_networkx_edges(
        g, pos, edgelist=edges, width=1, alpha=0.5, edge_color="b", style="dashed"
    )

    ax = plt.gca()
    ax.margins(0.08)
    plt.axis("off")
    plt.tight_layout()
    plt.show()
            


if __name__ == "__main__":
    # convert_csv_to_json()
    construct_graph()