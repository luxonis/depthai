import jsonlines
from typing import Literal, Dict, Union
from datetime import datetime
import requests
import os

CACHE_FILE_NAME = os.environ.get('CACHE_FILE_NAME', 'results_cache.jsonl')
CACHE_DIRECTORY = os.environ.get('CACHE_DIRECTORY', '')
CACHE_FILE_PATH = os.path.join(CACHE_DIRECTORY, CACHE_FILE_NAME)
STATS_SERVER_URL = os.environ.get('STATS_SERVER_URL', 'http://localhost')
API_KEY = os.environ.get('API_KEY', '1234')

print(f'Stats server API: {STATS_SERVER_URL}, API key: {API_KEY}')

def add_result(
	stage: Literal['calib', 'test'], 
	device_id: str, 
	device_type: str, 
	bootloader_version: str,
	library_version: str,
	start_time: datetime,
	end_time: datetime,
	result_data: Dict
):
	"""
	Adds a result to the cache.
	"""

	result = {
		'stage': stage,
		'device_id': device_id,
		'device_type': device_type,
		'bootloader_version': bootloader_version,
		'library_version': library_version,
		'start_time': start_time.isoformat(),
		'end_time': end_time.isoformat(),
		'result_data': result_data
	}

	# add to cache
	with jsonlines.open(CACHE_FILE_PATH, mode='a') as writer:
		writer.write(result)

def sync():
	"""
	Tries to upload the result to the server. If the server is not available, 
	the result stays in the cache and will be uploaded later.
	"""
	try:
		with jsonlines.open(CACHE_FILE_PATH, mode='r') as reader:
			results = [result for result in reader]
		
		response = requests.post(
			f'{STATS_SERVER_URL}/results', 
			json=results, 
			headers={'x-api-key': API_KEY},
			timeout=1
		)
		if response.status_code != 200:
			raise Exception(f'Status code: {response.status_code}, response: {response.text}')
		# clear cache
		with open(CACHE_FILE_PATH, 'w') as f:
			f.write('')
	except Exception as e:
		print(f'Failed to upload results: {e}')