import socket
import traceback
import time
import json
from configs.config import get_args
from utils.bvh_utils import load_bvh_to_tensor
from utils.model import AniBot
from configs.config import driver_types, skeleton_types, LimbHyper
ENCODING = 'utf8'


def parse_buffer(buf: bytes):
    msg = buf.decode(encoding=ENCODING)
    msg = msg.replace("'", '"')
    msg = json.loads(msg)
    # print(msg, type(msg))
    return msg

def run_optimize(params: dict):
    hyper, learnable = get_args()

    motions = load_bvh_to_tensor('./saves/dog_motion1.bvh')
    motion = motions['joint_rot']
    pos = motions['joint_pos']
    lengths = motions['bone_lengths']

    joints = 2*(3 + params['skel_type'])
    limbs = joints - 2

    hyper.skeleton_type = skeleton_types[params['skel_type']]
    hyper.drive_type = driver_types[params['drive_type']]

    hyper.driver_radius = params['driver_radius']
    hyper.driver_offset = params['driver_offset']

    hyper.bone_tail = params['bone_tail']
    hyper.base_r = params['base_r']
    hyper.slave_r = params['slave_r']

    hyper.spring_c = params['spring_c']

    for i in range(limbs):
        limb = LimbHyper()
        limb.length = params['bone_lengths'][i]
        limb.bend_angle = params['bend_limits'][i]
        limb.stretch_angle = params['stretch_limits'][i]
        hyper.skeleton_general[i] = limb

    steps = int(params['steps'])

    bot = AniBot(hyper, learnable, motion)
    bot.optimize(steps)

    res = bot.get_result().detach().to('cpu').numpy()
    parsed = {'base_spring_t': [],
              'slave_spring_t': [],
              'driver_phase': [],
              'driver_radius': []}
    parsed['base_spring_t'] = res[:, 0].tolist()
    parsed['slave_spring_t'] = res[:, 1].tolist()
    parsed['driver_phase'] = res[:, 3].tolist()
    parsed['driver_radius'] = res[:, 4].tolist()
    return json.dumps(parsed)



host = ""
port = 800
address = (host, port)
time_now = time.strftime("%Y-%m-%d %H:%S:%M", time.localtime())
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
s.bind(address)
s.listen(1)

while True:
    print("Waiting for connections...")
    try:
        client_connection, client_address = s.accept()
    except KeyboardInterrupt:
        raise
    except:
        traceback.print_exc()
        continue

    try:
        print ("Got connection from", client_connection.getpeername())
        while True:
            # client_connection.settimeout(5)
            buf = client_connection.recv(4096)
            if len(buf) == 0: # break,跳出while循环
                break
            else:
                # client_connection.send(bytes(analysis(buf.decode()), encoding="utf8"))
                parsed = parse_buffer(buf)
                res = run_optimize(parsed)
                client_connection.send(bytes(str(res), encoding=ENCODING))
                print('msg sent.')
    except (KeyboardInterrupt, SystemError):
        raise
    except:
        traceback.print_exc()

try:
    client_connection.close()
except KeyboardInterrupt:
    raise
except:
    traceback.print_exc()

