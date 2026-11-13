import runpy
import sys
from pathlib import Path

import valhalla_base

valhalla_base.BIN = Path(r"C:\Ai\Valhalla-main\target\release\valhalla_base.exe")
# This checkout's official tests import helpers across sibling tool dirs
# (and one absent module, stubbed): put every tools/* dir on the path.
for _p in Path(r"C:\Ai\Valhalla-main\tools").iterdir():
    if _p.is_dir():
        sys.path.insert(0, str(_p))
sys.path.insert(0, r"C:\Ai\rogue-products\valhalla-server\tests\stubs")
sys.argv = [sys.argv[1]] + sys.argv[2:]
runpy.run_path(sys.argv[0], run_name="__main__")
