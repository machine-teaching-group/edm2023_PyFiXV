import sys
import time
from enum import Enum, auto

sys.path.append(".")

from src.utils.codex_utils import get_n_queries_Codex_C, get_n_queries_Codex_E


class ECStatus(Enum):
    INIT = auto()
    START = auto()
    PAUSE = auto()
    END = auto()


class End2EndClerk:
    def __init__(self):
        self.status = ECStatus.INIT

        self.cached_elapsed_time = None
        self.cached_n_calls_Codex_E, self.cached_n_outputs_Codex_E = None, None
        self.cached_n_calls_Codex_C, self.cached_n_outputs_Codex_C = None, None

        self.start_time = None
        self.start_n_calls_Codex_E, self.start_n_outputs_Codex_E = None, None
        self.start_n_calls_Codex_C, self.start_n_outputs_Codex_C = None, None

    def _reset_cache(self):
        self.cached_elapsed_time = 0
        self.cached_n_calls_Codex_E, self.cached_n_outputs_Codex_E = 0, 0
        self.cached_n_calls_Codex_C, self.cached_n_outputs_Codex_C = 0, 0

    def _reset_start(self):
        self.start_time = time.time()
        self.start_n_calls_Codex_E, self.start_n_outputs_Codex_E = get_n_queries_Codex_E()
        self.start_n_calls_Codex_C, self.start_n_outputs_Codex_C = get_n_queries_Codex_C()

    def start(self):
        self.status = ECStatus.START

        self._reset_cache()
        self._reset_start()

    def pause(self):
        if not self.status == ECStatus.START:
            return

        self.status = ECStatus.PAUSE

        elapsed_time = time.time() - self.start_time
        self.cached_elapsed_time += elapsed_time

        current_n_calls_Codex_E, current_n_outputs_Codex_E = get_n_queries_Codex_E()
        self.cached_n_calls_Codex_E += current_n_calls_Codex_E - self.start_n_calls_Codex_E
        self.cached_n_outputs_Codex_E += current_n_outputs_Codex_E - self.start_n_outputs_Codex_E

        current_n_calls_Codex_C, current_n_outputs_Codex_C = get_n_queries_Codex_C()
        self.cached_n_calls_Codex_C += current_n_calls_Codex_C - self.start_n_calls_Codex_C
        self.cached_n_outputs_Codex_C += current_n_outputs_Codex_C - self.start_n_outputs_Codex_C

    def unpause(self):
        if self.status != ECStatus.PAUSE:
            return

        self.status = ECStatus.START

        self._reset_start()

    def end(self):
        if not self.status == ECStatus.START:
            return None

        self.pause()
        self.status = ECStatus.END

        return {
            "elapsed_time": self.cached_elapsed_time,
            "n_calls_Codex_E": self.cached_n_calls_Codex_E,
            "n_outputs_Codex_E": self.cached_n_outputs_Codex_E,
            "n_calls_Codex_C": self.cached_n_calls_Codex_C,
            "n_outputs_Codex_C": self.cached_n_outputs_Codex_C,
        }
