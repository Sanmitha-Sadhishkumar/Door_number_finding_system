import google_streetview.api as gsv
import googlemaps as gm
import os
from dotenv import load_dotenv

load_dotenv()

def cor_to_img(location, heading, pitch,size='600x300'):
		params = [{
		'size': size,
		'location': str(location),
		'heading': str(heading),
		'pitch': str(pitch),
		'key': str(os.getenv('api_key1'))
		}]
		results = gsv.results(params)
		print(results)
		results.download_links('E:/Summer project/images')

def loc_to_cor():
	client=gm.Client(key=str(os.getenv('api_key1'))) #todo
	geo=client.geocode('india')
	print(geo)
