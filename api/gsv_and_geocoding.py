import google_streetview.api as gsv
import googlemaps as gm
import os
from dotenv import load_dotenv

load_dotenv()


def cor_to_img(location, heading, pitch, size="600x300"):
    params = [
        {
            "size": size,
            "location": str(location),
            "heading": str(heading),
            "pitch": str(pitch),
            "key": str(os.getenv("api_key1")),
        }
    ]
    results = gsv.results(params)
    results.download_links(r"E:\git\Door_number_finding_system\images")


def loc_to_cor():
    client = gm.Client(key=str(os.getenv("api_key1")))  # todo
    geo = client.geocode("india")
    print(geo)


cor_to_img("11.333293, 77.718476", 90, 0)
# print(help(gsv))
