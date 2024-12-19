#  Copyright (c) 2018-present, Cruise LLC
#
#  This source code is licensed under the Apache License, Version 2.0,
#  found in the LICENSE file in the root directory of this source tree.
#  You may not use this file except in compliance with the License.
#  Authored by Liyan Chen (liyanc@cs.utexas.edu) on 10/6/24


from ..lib import pshattn

DEV_MGR = pshattn.DeviceManagerSingleton.instance()
ENABLE_NVTX = False


def get_dev_manager():
    return DEV_MGR


def init_dev(dev_id: int):
    assert isinstance(dev_id, int), f'Expect dev_id is an int, got {type(dev_id)}'
    DEV_MGR.init_dev_info(dev_id)

def get_sync_stream():
    return DEV_MGR.get_sync_stream()

def set_sync_stream(is_syncing: bool):
    return DEV_MGR.set_sync_stream(is_syncing)


def set_nvtx(flag: bool):
    ENABLE_NVTX = flag

def get_nvtx():
    return ENABLE_NVTX
