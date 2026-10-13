import os
import sys
import types


def configure():
    os.environ.setdefault('TF_USE_LEGACY_KERAS', '1')
    os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')

    if 'silence_tensorflow' not in sys.modules:
        stub = types.ModuleType('silence_tensorflow')
        stub.silence_tensorflow = lambda: None
        sys.modules['silence_tensorflow'] = stub

    src = os.environ.get('OLYMPUS_SRC')
    if src and src not in sys.path:
        sys.path.insert(0, src)
