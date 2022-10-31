import sys
import requests
import itertools as it
import basis_set_exchange as bse


def count_entries(api_key: str, element_a: str, element_b: str = None) -> int:
    """Count the number of entries in the Materials Project database.
    Args:
        api_key (str): The Materials Project API key.
        element_a (str): The first element to search for.
        element_b (str): The second element to search for.
    Returns:
        int: The number of entries in the database.
    """
    # Create the query
    query = {
        "elements": ",".join([element_a, element_b]) if element_b else element_a,
        "deprecated": False,
    }

    url = "https://api.materialsproject.org/summary/"
    headers = {"X-API-KEY": api_key}
    response = requests.get(url, headers=headers, params=query)
    return response.json()["meta"]["total_doc"]


def get_elements():
    """Get a list of elements from the Basis Set Exchange"""
    return [bse.lut.element_sym_from_Z(_).capitalize() for _ in range(1, 101)]


if __name__ == "__main__":
    # get api key from args
    api_key = sys.argv[1]
    # print(count_entries(api_key, "Al"))
    elements = get_elements()
    # for element1, element2 in it.combinations(elements, 2):
    #     entrycount = count_entries(api_key, element1, element2)
    #     print(f"materialsproject,DFT,{element1},{element2},{entrycount}")
    for element in elements:
        entrycount = count_entries(api_key, element)
        print(f"materialsproject,DFT,{element},single,{entrycount}")
