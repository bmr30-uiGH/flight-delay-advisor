import pandas as pd

from src.reference_data import airport_label, build_route_reference, carrier_label


def test_build_route_reference_uses_bts_routes_and_distances():
    frame = pd.DataFrame(
        {
            "OP_CARRIER": ["AA", "AA", "DL", "DL"],
            "ORIGIN": ["JFK", "JFK", "ATL", "ATL"],
            "DEST": ["LAX", "LAX", "JFK", "JFK"],
            "DISTANCE": [2475, 2475, 760, 760],
        }
    )
    reference = build_route_reference(frame)
    assert "AA" in reference["carriers"]
    assert "JFK" in reference["origins"]
    assert "LAX" in reference["destinations_by_origin"]["JFK"]
    assert reference["distance_by_route"][("JFK", "LAX")] == 2475


def test_airport_label_prefers_directory_name():
    directory = {
        "JFK": {
            "name": "John F Kennedy International Airport",
            "municipality": "Queens",
            "served_city": "New York",
            "region": "New York",
            "country": "US",
        }
    }
    label = airport_label("JFK", directory)
    assert "JFK" in label
    assert "John F Kennedy" in label
    assert "New York" in label


def test_carrier_label_has_common_names_and_fallback():
    assert carrier_label("AA") == "AA — American Airlines"
    assert carrier_label("QQQ").startswith("QQQ —")


def test_parse_official_carrier_table():
    from src.reference_data import parse_carrier_directory

    html = """<table><tr><th>CARRIER</th><th>CARRIERNAME</th></tr>
    <tr><td>AA</td><td>American Airlines Inc.</td></tr>
    <tr><td>DL</td><td>Delta Air Lines Inc.</td></tr></table>"""
    names = parse_carrier_directory(html)
    assert names["AA"] == "American Airlines Inc."
    assert names["DL"] == "Delta Air Lines Inc."
