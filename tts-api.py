import os
import io
import re
import time
import json
import requests
import pysbd
import pydub
import subprocess
from flask import Flask, request, send_file, abort, jsonify

api = Flask(__name__)

def hhmmss_to_seconds(string):
	new_time = 0
	separated_times = string.split(":")
	new_time = 60 * 60 * float(separated_times[0])
	new_time += 60 * float(separated_times[1])
	new_time += float(separated_times[2])
	return new_time

# Generation settings
tts_sample_rate = 40000
segmenter = pysbd.Segmenter(language="en", clean=True)
radio_starts = ["./on1.wav", "./on2.wav"]
radio_ends = ["./off1.wav", "./off2.wav", "./off3.wav", "./off4.wav"]
authorization_token = os.getenv("TTS_AUTHORIZATION_TOKEN", "coolio")

# TTS model mapping
voice_settings_json = {}
names_json = {}
use_voice_name_mapping = True

# Define the URL of the Flask server
server_url = "http://127.0.0.1:5003/"

# For health checks
req_count = 1

# Load the existing JSON data
with open("./tts_voices_mapping.json", "r") as file:
	voice_settings_json = json.load(file)
	if len(voice_settings_json) == 0:
		use_voice_name_mapping = False

# Extract only the voice name
def extract_names(json_data):
    names = []
    for item in json_data['voice_settings']:
        names.append(item['voice'])
    return json.dumps(names)

# Extract the default setting for a voice
def extract_voice_setting(voice, setting):
    setting_value = None
    for item in voice_settings_json['voice_settings']:
        if item['voice'] == voice:
            setting_value = item['settings'][setting]
            break
    return setting_value

names_json = extract_names(voice_settings_json)

#
# Filter Pipeline
#
def text_to_speech_handler(endpoint, voice, text, filter_complex, pitch, special_filters = []):
	global req_count

	filter_complex = filter_complex.replace("\"", "")
	data_bytes = io.BytesIO()
	final_audio = pydub.AudioSegment.empty()
	model = extract_voice_setting(voice, "edge_tts_voice")
	speed = extract_voice_setting(voice, "speed")

	for sentence in segmenter.segment(text):
		# Send the segmented audo as a request to the flask endpoint
		response = requests.get(server_url + endpoint, json={
			'model_name': voice,
			'speed': speed,
			'tts_text': sentence,
			'tts_voice': model,
			'f0_up_key': pitch,
			'f0_method': 'rmvpe',
			'index_rate': 1,
			'protect': 0.33})
		#client = Client(server_url)
		#client.view_api()
		req_count += 1
		print(f"Request {req_count} sent: {voice}:{sentence}")

		if response.status_code != 200:
			abort(500)

		# Piece together audio from segments
		# Use the edge mp3 for edge-tts voices, and the output wav for others
		sentence_audio = pydub.AudioSegment.from_file(io.BytesIO(response.content), "wav")
		sentence_silence = pydub.AudioSegment.silent(250, 40000)
		sentence_audio += sentence_silence
		final_audio += sentence_audio

		# ""Goldman-Eisler (1968) determined that typical speakers paused for an average of 250 milliseconds (ms), with a range from 150 to 400 ms.""
		# (https://scholarsarchive.byu.edu/cgi/viewcontent.cgi?article=10153&context=etd)

	final_audio.export(data_bytes, format="wav")
	
	filter_complex = filter_complex.replace("%SAMPLE_RATE%", str(tts_sample_rate))
	ffmpeg_result = None
	if filter_complex != "":
		ffmpeg_result = subprocess.run(["ffmpeg", "-f", "wav", "-i", "pipe:0", "-filter_complex", filter_complex, "-c:a", "libvorbis", "-b:a", "64k", "-f", "ogg", "pipe:1"], input=data_bytes.read(), capture_output = True)
	else:
		if "silicon" in special_filters:
			ffmpeg_result = subprocess.run(["ffmpeg", "-f", "wav", "-i", "pipe:0", "-i", "./SynthImpulse.wav", "-i", "./RoomImpulse.wav", "-filter_complex", "[0] aresample=44100 [re_1]; [re_1] apad=pad_dur=2 [in_1]; [in_1] asplit=2 [in_1_1] [in_1_2]; [in_1_1] [1] afir=dry=10:wet=10 [reverb_1]; [in_1_2] [reverb_1] amix=inputs=2:weights=8 1 [mix_1]; [mix_1] asplit=2 [mix_1_1] [mix_1_2]; [mix_1_1] [2] afir=dry=1:wet=1 [reverb_2]; [mix_1_2] [reverb_2] amix=inputs=2:weights=10 1 [mix_2]; [mix_2] equalizer=f=7710:t=q:w=0.6:g=-6,equalizer=f=33:t=q:w=0.44:g=-10 [out]; [out] alimiter=level_in=1:level_out=1:limit=0.5:attack=5:release=20:level=disabled", "-c:a", "libvorbis", "-b:a", "64k", "-f", "ogg", "pipe:1"], input=data_bytes.read(), capture_output = True)
		else:
			ffmpeg_result = subprocess.run(["ffmpeg", "-f", "wav", "-i", "pipe:0", "-c:a", "libvorbis", "-b:a", "64k", "-f", "ogg", "pipe:1"], input= data_bytes.read(), capture_output = True)
	ffmpeg_metadata_output = ffmpeg_result.stderr.decode()
	#print(f"ffmpeg result size: {len(ffmpeg_result.stdout)} stderr = \n{ffmpeg_metadata_output}")
	export_audio = io.BytesIO(ffmpeg_result.stdout)
	if "radio" in special_filters:
		radio_audio = pydub.AudioSegment.from_file(random.choice(radio_starts), "wav")
		radio_audio += pydub.AudioSegment.from_file(io.BytesIO(ffmpeg_result.stdout), "ogg")
		radio_audio += pydub.AudioSegment.from_file(random.choice(radio_ends), "wav")
		new_data_bytes = io.BytesIO()
		radio_audio.export(new_data_bytes, format="ogg")
		export_audio = io.BytesIO(new_data_bytes.getvalue())
	matched_length = re.search(r"time=([0-9:\\.]+)", ffmpeg_metadata_output)
	hh_mm_ss = matched_length.group(1)
	length = hhmmss_to_seconds(hh_mm_ss)

	response = send_file(export_audio, as_attachment=True, download_name='identifier.ogg', mimetype="audio/ogg")
	response.headers['audio-length'] = length

	# Write the stuff for debugging
	with open("last_output.ogg", "wb") as f:
		f.write(export_audio.getbuffer())

	return response

#
# API Endpoints
#
@api.route("/tts")
def text_to_speech_normal():
	#if authorization_token != request.headers.get("Authorization", ""):
	#	abort(401)

	voice = request.args.get("voice", '')
	text = request.json.get("text", '')
	pitch = request.args.get("pitch", '0')
	special_filters = request.args.get("special_filters", '')
	# Combine player-set pitch with JSON-defined pitch
	pitch = str(int(pitch) + int(extract_voice_setting(voice, "pitch")))
	special_filters = special_filters.split("|")
	silicon = request.args.get("silicon", False)
	if pitch == "":
		pitch = "0"
	if text == '':
		text = request.json.get("text", '')
	if silicon:
		special_filters = ["silicon"]

	identifier = request.args.get("identifier", '')
	filter_complex = request.args.get("filter", '')
	#print(f"text_to_speech_handler(/generate-tts, {voice}, {text}, {filter_complex}, {pitch}, {special_filters})")
	return text_to_speech_handler("/generate-tts", voice, text, filter_complex, pitch, special_filters)

# Return available models/voices
@api.route("/tts-voices")
def voices_list():
	global req_count

	if use_voice_name_mapping:
		print("/tts-voices: " + str(names_json))
		req_count += 1
		return names_json
	else:
		abort(500)

# Return available pitch
@api.route("/pitch-available")
def pitch_available():
    response = requests.get(f"{server_url}/pitch-available")

    if response.status_code != 200:
        abort(500)
    return "Pitch available", 200

# Return health status
@api.route("/health-check")
def tts_health_check():
    global req_count

    try:
        # Send a GET request to the Flask server
        response = requests.get(f"{server_url}/health-check")

        # Check the status code of the response
        if response.status_code == 200:
            req_count += 1
            return jsonify({"status": "OK", "req_count": f"{req_count}"}), 200
        else:
            return jsonify({"status": "ERROR", "message": f"Server responded with status code {response.status_code}"}), 500

    except Exception as e:
        return jsonify({"status": "unhealthy", "message": str(e)}), 500

# Entrypoint
if __name__ == "__main__":
	if os.getenv('TTS_LD_LIBRARY_PATH', "") != "":
		os.putenv('LD_LIBRARY_PATH', os.getenv('TTS_LD_LIBRARY_PATH'))
	from waitress import serve
	print("Starting API!")
	serve(api, host="0.0.0.0", port=5002, threads=2, backlog=8, connection_limit=24, channel_timeout=10)

