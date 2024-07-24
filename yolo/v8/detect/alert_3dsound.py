import time
from openal.audio import SoundSink, SoundSource
from openal.loaders import load_wav_file

from json.decoder import JSONDecodeError
import json
import time

crowd_detect = True

json_file_path = "obs_state.json"
with open(json_file_path, 'r') as j:
            contents = json.loads(j.read())
            obs_hist = contents['state']

# Alert sound options:
# pop
# bounce
# droplet 13, 18
# beep

sink = SoundSink()
sink.activate()

source = [SoundSource(position=[-5, 0, 0]),SoundSource(position=[5, 0, 0]), SoundSource(position=[0, 0, 5]), SoundSource(position=[0, 0, 0])] ## Left, Right, Mid

source[0].looping = True
source[1].looping = True
source[2].looping = True
source[3].looping = True

data = load_wav_file("sounds/beep-07.wav")
crowd_sound = load_wav_file("sounds/crowd.wav")

source[0].queue(data)
source[1].queue(data)
source[2].queue(data)
source[3].queue(crowd_sound)

if not crowd_detect:
    # print("without crowd")
    while True:
        try:
            with open(json_file_path, 'r') as j:
                contents = json.loads(j.read())

                obs_current = contents['state']
                time.sleep(0.3)
                for a in [0,1,2]:
                    if obs_current[a]:
                        sink.play(source[a])
                        sink.update()
                    else:
                        sink.pause(source[a])
                        sink.update()

        except JSONDecodeError:
            continue

else:
    # print("with crowd")
    while True:
        try:
            with open(json_file_path, 'r') as j:
                contents = json.loads(j.read())

                obs_current = contents['state']
                if obs_current[3]:
                    for a in [0,1,2]:
                        sink.pause(source[a])
                        sink.update()
                    sink.play(source[3])
                    sink.update()
                    time.sleep(1.06)
                    sink.pause(source[3])
                    sink.update()
                    time.sleep(5)
                    continue

                for a in [0,1,2,3]:
                    if obs_current[a]:
                        sink.play(source[a])
                        sink.update()
                    else:
                        sink.pause(source[a])
                        sink.update()

        except JSONDecodeError:
            continue