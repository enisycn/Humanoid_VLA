"""Trusted local process framing; only builtins cross the NumPy1/2 boundary."""
import math
import pickle
import struct

SCHEMA = 'stock_SmolVLA_local_inference_v1'
TASK = 'push the plate to the front of the stove'
IMAGE_KEYS = ('observation.images.image', 'observation.images.image2')
RGB_SHAPE = (2, 256, 256, 3)
MAX_BATCH = 64
MAX_MESSAGE = 64 * 1024 * 1024


def read_exact(stream, count):
    chunks = []
    while count:
        value = stream.read(count)
        if not value:
            raise EOFError('Local inference pipe closed inside a message')
        chunks.append(value)
        count -= len(value)
    return b''.join(chunks)


def send(stream, message):
    value = pickle.dumps(message, protocol=4)
    if len(value) > MAX_MESSAGE:
        raise ValueError('Local inference message exceeds declared bound')
    # Unbuffered pipe writes can be short even though the pipe is blocking.
    for payload in (struct.pack('!Q', len(value)), value):
        view = memoryview(payload)
        while view:
            written = stream.write(view)
            if written is None or written <= 0:
                raise EOFError('Local inference pipe closed inside a write')
            view = view[written:]
    stream.flush()


def receive(stream):
    length, = struct.unpack('!Q', read_exact(stream, 8))
    if not 0 < length <= MAX_MESSAGE:
        raise ValueError('Invalid local inference frame length')
    result = pickle.loads(read_exact(stream, length))
    if type(result) is not dict:
        raise ValueError('Local inference envelope must be a builtin dict')
    return result


def validate_observation(message):
    expected = {'op', 'request_id', 'batch', 'rgb2', 'own8', 'task'}
    if set(message) != expected or message['op'] != 'act' or message['task'] != TASK:
        raise ValueError('Only actual RGB2, own8 and the fixed instruction are admitted')
    batch = message['batch']
    if type(batch) is not int or not 1 <= batch <= MAX_BATCH:
        raise ValueError('Batch must be an integer in [1,64]')
    if type(message['rgb2']) is not bytes or len(message['rgb2']) != batch * 2 * 256 * 256 * 3:
        raise ValueError('RGB must contain exactly uint8[B,2,256,256,3] bytes')
    if type(message['own8']) is not bytes or len(message['own8']) != batch * 8 * 4:
        raise ValueError('State must contain exactly little-endian float32[B,8] bytes')
    if not all(math.isfinite(v) for v in struct.unpack('<' + 'f' * (batch * 8), message['own8'])):
        raise ValueError('Robot own8 must be finite')
    return batch


def observation_message(rgb2, own8, request_id):
    """Array-like inputs are checked/copied as bytes, never cast or re-oriented."""
    shape = tuple(rgb2.shape)
    if len(shape) != 5 or shape[1:] != RGB_SHAPE or str(rgb2.dtype) != 'uint8':
        raise ValueError('Provide already official-oriented RGB uint8[B,2,256,256,3]')
    batch = shape[0]
    if tuple(own8.shape) != (batch, 8) or str(own8.dtype) != 'float32':
        raise ValueError('Provide the existing official own8 convention as float32[B,8]')
    if getattr(own8.dtype, 'byteorder', '=') not in ('<', '='):
        raise ValueError('own8 must use the native little-endian float32 representation')
    result = dict(op='act', request_id=request_id, batch=batch,
                  rgb2=rgb2.tobytes(order='C'), own8=own8.tobytes(order='C'), task=TASK)
    validate_observation(result)
    return result
