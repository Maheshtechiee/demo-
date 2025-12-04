from __future__ import annotations

import json
import os
import threading
import time
import webbrowser
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

from flask import Flask, jsonify, render_template, request
import folium
import networkx as nx
from branca.element import MacroElement, Template
from shapely.geometry import Point, Polygon, mapping, LineString
from shapely.geometry.base import BaseGeometry


@dataclass(frozen=True)
class HazardZone:
    name: str
    category: str
    severity: float  # 0-1 scale where 1 is the most dangerous
    influence_radius_m: float
    description: str
    geometry: BaseGeometry


app = Flask(__name__)

# Bounding box for the working area (Tamil Nadu, India)
LAT_MIN, LAT_MAX = 8.0, 14.0
LON_MIN, LON_MAX = 76.0, 81.0
GRID_ROWS = GRID_COLS = 100
LAT_STEP = (LAT_MAX - LAT_MIN) / (GRID_ROWS - 1)
LON_STEP = (LON_MAX - LON_MIN) / (GRID_COLS - 1)
DEGREE_TO_M = 111_320  # Approximate conversion for small areas


def polygon_from_latlon(coords: Iterable[Tuple[float, float]]) -> Polygon:
    """Helper that receives (lat, lon) tuples and returns a shapely Polygon."""
    # Shapely expects (lon, lat) ordering
    lon_lat = [(lon, lat) for lat, lon in coords]
    return Polygon(lon_lat)


HAZARD_ZONES: List[HazardZone] = [
    HazardZone(
        name="Tirunelveli Safe Route SH39 to SH41",
        category="Safe",
        severity=0.0,
        influence_radius_m=200,
        description="Safe alternative route from SH39 to SH41 near Nainar Kovil.",
        geometry=LineString([(77.69083, 8.730717), (77.54, 9.18)]),
    ),
    HazardZone(
        name="Adyar River Flooding",
        category="Flood",
        severity=0.95,
        influence_radius_m=800,
        description="Historical flooding along Adyar River during monsoon season.",
        geometry=polygon_from_latlon(
            [
                (13.005, 80.250),
                (13.015, 80.250),
                (13.015, 80.240),
                (13.005, 80.240),
            ]
        ),
    ),
    HazardZone(
        name="Cooum River Flooding",
        category="Flood",
        severity=0.90,
        influence_radius_m=700,
        description="Frequent flooding in Cooum River basin areas.",
        geometry=polygon_from_latlon(
            [
                (13.080, 80.280),
                (13.090, 80.280),
                (13.090, 80.270),
                (13.080, 80.270),
            ]
        ),
    ),
    HazardZone(
        name="T. Nagar Traffic Accident Zone",
        category="Accident",
        severity=0.75,
        influence_radius_m=500,
        description="High accident rates in busy T. Nagar commercial area.",
        geometry=polygon_from_latlon(
            [
                (13.040, 80.230),
                (13.050, 80.230),
                (13.050, 80.220),
                (13.040, 80.220),
            ]
        ),
    ),
    HazardZone(
        name="Anna Nagar Accident Zone",
        category="Accident",
        severity=0.80,
        influence_radius_m=450,
        description="Frequent traffic accidents in Anna Nagar residential and commercial hub.",
        geometry=polygon_from_latlon(
            [
                (13.080, 80.210),
                (13.090, 80.210),
                (13.090, 80.200),
                (13.080, 80.200),
            ]
        ),
    ),
    HazardZone(
        name="Velachery Accident Zone",
        category="Accident",
        severity=0.70,
        influence_radius_m=400,
        description="High accident rates in Velachery due to heavy traffic and junctions.",
        geometry=polygon_from_latlon(
            [
                (12.970, 80.220),
                (12.980, 80.220),
                (12.980, 80.210),
                (12.970, 80.210),
            ]
        ),
    ),
    HazardZone(
        name="Tambaram Accident Zone",
        category="Accident",
        severity=0.65,
        influence_radius_m=350,
        description="Accident-prone areas in Tambaram due to urban congestion.",
        geometry=polygon_from_latlon(
            [
                (12.920, 80.120),
                (12.930, 80.120),
                (12.930, 80.110),
                (12.920, 80.110),
            ]
        ),
    ),
    HazardZone(
        name="Coimbatore Accident Zone",
        category="Accident",
        severity=0.85,
        influence_radius_m=600,
        description="High accident rates in Coimbatore due to industrial and traffic congestion.",
        geometry=polygon_from_latlon(
            [
                (11.000, 76.950),
                (11.010, 76.950),
                (11.010, 76.940),
                (11.000, 76.940),
            ]
        ),
    ),
    HazardZone(
        name="Madurai Accident Zone",
        category="Accident",
        severity=0.78,
        influence_radius_m=550,
        description="Frequent accidents in Madurai city center and highways.",
        geometry=polygon_from_latlon(
            [
                (9.920, 78.120),
                (9.930, 78.120),
                (9.930, 78.110),
                (9.920, 78.110),
            ]
        ),
    ),
    HazardZone(
        name="Trichy Accident Zone",
        category="Accident",
        severity=0.72,
        influence_radius_m=480,
        description="Accident-prone areas in Trichy due to river crossings and traffic.",
        geometry=polygon_from_latlon(
            [
                (10.790, 78.700),
                (10.800, 78.700),
                (10.800, 78.690),
                (10.790, 78.690),
            ]
        ),
    ),
    HazardZone(
        name="Salem Accident Zone",
        category="Accident",
        severity=0.68,
        influence_radius_m=420,
        description="High accident rates in Salem due to hilly terrain and junctions.",
        geometry=polygon_from_latlon(
            [
                (11.650, 78.150),
                (11.660, 78.150),
                (11.660, 78.140),
                (11.650, 78.140),
            ]
        ),
    ),
    HazardZone(
        name="Tirunelveli Accident Zone",
        category="Accident",
        severity=0.3,
        influence_radius_m=500,
        description="Low-risk alternative route in Tirunelveli due to coastal roads and traffic.",
        geometry=polygon_from_latlon(
            [
                (8.730, 77.700),
                (8.740, 77.700),
                (8.740, 77.690),
                (8.730, 77.690),
            ]
        ),
    ),
    HazardZone(
        name="Kanyakumari Flooding",
        category="Flood",
        severity=0.88,
        influence_radius_m=750,
        description="Coastal flooding in Kanyakumari during monsoon.",
        geometry=polygon_from_latlon(
            [
                (8.080, 77.550),
                (8.090, 77.550),
                (8.090, 77.540),
                (8.080, 77.540),
            ]
        ),
    ),
    HazardZone(
        name="Erode Accident Zone",
        category="Accident",
        severity=0.69,
        influence_radius_m=380,
        description="High accident rates in Erode due to textile industry traffic.",
        geometry=polygon_from_latlon(
            [
                (11.341, 77.717),
                (11.351, 77.717),
                (11.351, 77.707),
                (11.341, 77.707),
            ]
        ),
    ),
    HazardZone(
        name="Tiruppur Accident Zone",
        category="Accident",
        severity=0.71,
        influence_radius_m=390,
        description="Accident-prone areas in Tiruppur due to garment industry congestion.",
        geometry=polygon_from_latlon(
            [
                (11.108, 77.341),
                (11.118, 77.341),
                (11.118, 77.331),
                (11.108, 77.331),
            ]
        ),
    ),
    HazardZone(
        name="Vellore Accident Zone",
        category="Accident",
        severity=0.73,
        influence_radius_m=410,
        description="Frequent accidents in Vellore due to medical college and traffic.",
        geometry=polygon_from_latlon(
            [
                (12.916, 79.132),
                (12.926, 79.132),
                (12.926, 79.122),
                (12.916, 79.122),
            ]
        ),
    ),
    HazardZone(
        name="Kancheepuram Accident Zone",
        category="Accident",
        severity=0.67,
        influence_radius_m=360,
        description="Accidents in Kancheepuram due to temple tourism and highways.",
        geometry=polygon_from_latlon(
            [
                (12.834, 79.703),
                (12.844, 79.703),
                (12.844, 79.693),
                (12.834, 79.693),
            ]
        ),
    ),
    HazardZone(
        name="Tiruvallur Accident Zone",
        category="Accident",
        severity=0.74,
        influence_radius_m=430,
        description="High accident rates in Tiruvallur due to industrial areas.",
        geometry=polygon_from_latlon(
            [
                (13.145, 79.908),
                (13.155, 79.908),
                (13.155, 79.898),
                (13.145, 79.898),
            ]
        ),
    ),
    HazardZone(
        name="Tiruvannamalai Accident Zone",
        category="Accident",
        severity=0.66,
        influence_radius_m=350,
        description="Accidents in Tiruvannamalai due to hill roads and traffic.",
        geometry=polygon_from_latlon(
            [
                (12.225, 79.074),
                (12.235, 79.074),
                (12.235, 79.064),
                (12.225, 79.064),
            ]
        ),
    ),
    HazardZone(
        name="Viluppuram Accident Zone",
        category="Accident",
        severity=0.68,
        influence_radius_m=370,
        description="Accident-prone areas in Viluppuram due to coastal roads.",
        geometry=polygon_from_latlon(
            [
                (11.939, 79.492),
                (11.949, 79.492),
                (11.949, 79.482),
                (11.939, 79.482),
            ]
        ),
    ),
    HazardZone(
        name="Cuddalore Accident Zone",
        category="Accident",
        severity=0.70,
        influence_radius_m=400,
        description="Frequent accidents in Cuddalore due to port traffic.",
        geometry=polygon_from_latlon(
            [
                (11.748, 79.768),
                (11.758, 79.768),
                (11.758, 79.758),
                (11.748, 79.758),
            ]
        ),
    ),
    HazardZone(
        name="Nagapattinam Accident Zone",
        category="Accident",
        severity=0.65,
        influence_radius_m=340,
        description="Accidents in Nagapattinam due to coastal and agricultural traffic.",
        geometry=polygon_from_latlon(
            [
                (10.767, 79.842),
                (10.777, 79.842),
                (10.777, 79.832),
                (10.767, 79.832),
            ]
        ),
    ),
    HazardZone(
        name="Thanjavur Accident Zone",
        category="Accident",
        severity=0.72,
        influence_radius_m=420,
        description="High accident rates in Thanjavur due to temple and cultural traffic.",
        geometry=polygon_from_latlon(
            [
                (10.787, 79.137),
                (10.797, 79.137),
                (10.797, 79.127),
                (10.787, 79.127),
            ]
        ),
    ),
]


def node_to_latlon(node: Tuple[int, int]) -> Tuple[float, float]:
    lat = LAT_MIN + (node[0] * LAT_STEP)
    lon = LON_MIN + (node[1] * LON_STEP)
    return lat, lon


def latlon_to_node(lat: float, lon: float) -> Tuple[int, int]:
    if not (LAT_MIN <= lat <= LAT_MAX and LON_MIN <= lon <= LON_MAX):
        raise ValueError("Coordinate outside of supported bounds")

    row = round((lat - LAT_MIN) / LAT_STEP)
    col = round((lon - LON_MIN) / LON_STEP)
    return row, col


def clamp_coordinate(lat: float, lon: float) -> Tuple[float, float]:
    lat = max(min(lat, LAT_MAX), LAT_MIN)
    lon = max(min(lon, LON_MAX), LON_MIN)
    return lat, lon


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Approximate distance in meters using a spherical Earth approximation."""
    from math import asin, cos, radians, sin, sqrt

    d_lat = radians(lat2 - lat1)
    d_lon = radians(lon2 - lon1)
    a = sin(d_lat / 2) ** 2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(d_lon / 2) ** 2
    c = 2 * asin(sqrt(a))
    earth_radius = 6_371_000  # meters
    return earth_radius * c


def hazard_penalty(lat: float, lon: float) -> float:
    """Risk penalty coefficient for a coordinate based on surrounding hazards."""
    position = Point(lon, lat)
    penalty = 0.0
    for hazard in HAZARD_ZONES:
        distance_deg = hazard.geometry.distance(position)
        distance_m = distance_deg * DEGREE_TO_M
        if distance_m <= hazard.influence_radius_m:
            severity = hazard.severity
            # Inverse distance weighting with square falloff
            influence = (1 - (distance_m / hazard.influence_radius_m)) ** 2
            penalty += severity * influence
    return penalty


def edge_weight(node_a: Tuple[int, int], node_b: Tuple[int, int]) -> float:
    lat1, lon1 = node_to_latlon(node_a)
    lat2, lon2 = node_to_latlon(node_b)
    base_cost = haversine_distance(lat1, lon1, lat2, lon2)
    mid_lat = (lat1 + lat2) / 2
    mid_lon = (lon1 + lon2) / 2
    penalty = hazard_penalty(mid_lat, mid_lon)
    return base_cost * (1 + penalty)


def is_blocked(lat: float, lon: float) -> bool:
    location = Point(lon, lat)
    for hazard in HAZARD_ZONES:
        if hazard.severity >= 0.85 and hazard.geometry.contains(location):
            return True
    return False


def create_graph() -> nx.Graph:
    graph = nx.grid_graph(dim=[range(GRID_ROWS), range(GRID_COLS)])
    blocked_nodes = []
    for node in list(graph.nodes):
        lat, lon = node_to_latlon(node)
        if is_blocked(lat, lon):
            blocked_nodes.append(node)
    graph.remove_nodes_from(blocked_nodes)

    for u, v in graph.edges:
        graph[u][v]["weight"] = edge_weight(u, v)
    return graph


GRAPH = create_graph()
NODE_POINTS = {
    node: Point(node_to_latlon(node)[1], node_to_latlon(node)[0]) for node in GRAPH.nodes
}


def nearest_safe_node(lat: float, lon: float) -> Tuple[int, int]:
    clamped_lat, clamped_lon = clamp_coordinate(lat, lon)
    try:
        node = latlon_to_node(clamped_lat, clamped_lon)
        if node in GRAPH:
            return node
    except ValueError:
        pass

    target_point = Point(clamped_lon, clamped_lat)
    best_node = min(NODE_POINTS, key=lambda n: NODE_POINTS[n].distance(target_point))
    return best_node


class MapBinding(MacroElement):
    _template = Template(
        """
        {% macro script(this, kwargs) %}
            window.map = {{this._parent.get_name()}};
        {% endmacro %}
        """
    )


@app.route("/")
def index():
    start = (13.082, 80.270)  # T. Nagar area
    end = (13.056, 80.258)    # Adyar area

    map_center = ((start[0] + end[0]) / 2, (start[1] + end[1]) / 2)
    folium_map = folium.Map(location=map_center, zoom_start=13, control_scale=True)

    folium.Marker(location=start, popup="Start", icon=folium.Icon(color="green")).add_to(
        folium_map
    )
    folium.Marker(location=end, popup="End", icon=folium.Icon(color="blue")).add_to(
        folium_map
    )

    for hazard in HAZARD_ZONES:
        geo_json = json.loads(json.dumps(mapping(hazard.geometry)))
        color = "#B8860B" if hazard.category == "Flood" else "#8B0000"
        folium.GeoJson(
            geo_json,
            name=f"{hazard.category}: {hazard.name}",
            tooltip=f"{hazard.category}: {hazard.name}",
            style_function=lambda _, fill=color: {
                "fillColor": fill,
                "color": fill,
                "weight": 1,
                "fillOpacity": 0.45,
            },
            highlight_function=lambda *_: {"weight": 3, "color": "#2c7bb6"},
        ).add_to(folium_map)

    folium.LayerControl(collapsed=False).add_to(folium_map)
    folium_map.get_root().add_child(MapBinding())

    map_html = folium_map._repr_html_()
    return render_template(
        "index.html",
        map_html=map_html,
        lat_min=LAT_MIN,
        lat_max=LAT_MAX,
        lon_min=LON_MIN,
        lon_max=LON_MAX,
    )


@app.route("/hazards")
def hazards():
    features: List[Dict] = []
    for hazard in HAZARD_ZONES:
        geometry = json.loads(json.dumps(mapping(hazard.geometry)))
        features.append(
            {
                "type": "Feature",
                "geometry": geometry,
                "properties": {
                    "name": hazard.name,
                    "category": hazard.category,
                    "severity": hazard.severity,
                    "description": hazard.description,
                    "influence_radius_m": hazard.influence_radius_m,
                },
            }
        )
    return jsonify({"type": "FeatureCollection", "features": features})


@app.route("/find_path", methods=["POST"])
def find_path():
    data = request.get_json(force=True)
    try:
        start_lat = float(data["start_lat"])
        start_lon = float(data["start_lon"])
        end_lat = float(data["end_lat"])
        end_lon = float(data["end_lon"])
    except (KeyError, TypeError, ValueError) as exc:
        return jsonify({"error": f"Invalid input payload: {exc}"}), 400

    start_node = nearest_safe_node(start_lat, start_lon)
    end_node = nearest_safe_node(end_lat, end_lon)

    if start_node not in GRAPH or end_node not in GRAPH:
        return jsonify({"error": "Unable to locate navigable nodes for given coordinates."}), 400

    try:
        primary_path = nx.shortest_path(GRAPH, source=start_node, target=end_node, weight="weight")
        path_coords_list = [[node_to_latlon(node) for node in primary_path]]

        # For alternatives, penalize all previous paths to find distinct routes
        previous_paths = [primary_path]
        for _ in range(2):  # up to 2 alternatives
            temp_graph = GRAPH.copy()
            for prev_path in previous_paths:
                for u, v in zip(prev_path[:-1], prev_path[1:]):
                    if temp_graph.has_edge(u, v):
                        temp_graph[u][v]['weight'] *= 10  # increase weight significantly to discourage use
            try:
                alt_path = nx.shortest_path(temp_graph, source=start_node, target=end_node, weight="weight")
                path_coords_list.append([node_to_latlon(node) for node in alt_path])
                previous_paths.append(alt_path)
            except nx.NetworkXNoPath:
                break
    except nx.NetworkXNoPath:
        return jsonify({"error": "No safe path found that avoids hazard zones."}), 404

    return jsonify({"paths": path_coords_list})


def open_browser():
    time.sleep(1)  # Wait for server to start
    webbrowser.get('chrome').open('http://127.0.0.1:5000/')

if __name__ == "__main__":
    threading.Thread(target=open_browser).start()
    app.run(debug=True)
