import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from native_util import main_for_task, run_l4_solver

if __name__ == "__main__":
    sys.exit(main_for_task(run_l4_solver))
