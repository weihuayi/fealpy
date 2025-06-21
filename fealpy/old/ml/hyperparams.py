
from typing import List, Dict, Literal, Optional, Any, Generator, Tuple
from time import time
import json
import os


class AutoTest():

    class HyperParams():
        def __init__(self, data: Dict[str, Any]) -> None:
            self.__data = data
        def __len__(self):
            return len(self.__data)
        def __getattr__(self, key):
            return self.__data.get(key, None)

    def __init__(self, settings: Dict[str, List]) -> None:
        self.keys = list(settings.keys())
        self.values = [settings[k] for k in self.keys]
        self.reset()
        self._lens = tuple((len(vs) for vs in self.values))
        self._running: bool = False
        self._autosave: Optional[_Saver] = None

    def __len__(self):
        return len(self.keys)

    def reset(self):
        self._state = [0] * len(self.keys)
        self._step = 0
        self._records = []
        self._time_global = 0.0
        self._time_current = 0.0

    def get(self, *idx: int):
        """Output current parameter combination as a list."""
        return tuple(self.values[i][k] for i, k in enumerate(idx))

    def as_dict(self, *idx: int):
        """Outout current parameter value as a dict."""
        vals = self.get(*idx)
        return {self.keys[i]: vals[i] for i in range(len(self))}

    def record(self, **msg):
        """
        Record the information generated during the test.

        Example
        ---
        ```
        for w in autotest.run():
            # training codes #

            autotest.record(time=..., loss=..., error=...)
        ```

        Raise
        ---
        Raise Exception if call `record` when test is not running.
        """
        if not self._running:
            raise Exception('Can not record when test is not running.')

        rec = {}
        rec['params'] = self.as_dict(*self._state)
        rec['msg'] = msg

        if self._autosave is None:
            self._records.append(rec)
        else:
            self._autosave.save(rec)

    def set_autosave(self, filename: Optional[str],
                     mode: Literal['append', 'replace']='append'):
        """Control the autosave behavior. If `filename` is not None, automatically
        save the records to file when calling `record`.

        Args
        ---
        filename: str or None.
        mode: `'append'` or `'replace'`.
            the 'append' mode: always append new records to file.
            the 'replace' mode: records will be cleared when new test starts.

        Warning
        ---
        When setting `filename` to `None`, autosave stops and all records be discarded.
        """
        if filename is not None:
            self._autosave = _Saver(filename=filename, mode=mode)
        else:
            self._autosave = None

    def next_idx(self):
        cursor = 0
        while self._state[cursor] >= self._lens[cursor] - 1:
            self._state[cursor] = 0
            cursor += 1
            if cursor >= len(self.keys):
                return 1
        self._state[cursor] += 1
        self._step += 1
        return 0

    def test_runtime(self):
        """Return time from the start of all the tests to the preesent.
        Return `0.0` if the test is not running."""
        if self._running:
            return time() - self._time_global
        return 0.0

    def item_runtime(self):
        """Return the time since the start of current test to
        the present.
        Return `0.0` if the test is not running."""
        if self._running:
            return time() - self._time_current
        return 0.0

    def run(self, auto_reset: bool=True):
        """Start the test."""
        flag = 0
        self._running = True
        print(f"自动测试启动...")

        if auto_reset:
            print("重置测试器状态")
            self.reset()

        if self._autosave is not None:
            print(f"已启用记录自动保存：{self._autosave.filename}")
            self._autosave.start()

        self._time_global = time()
        while flag == 0:
            print(f"\n===== 进行第 {self._step} 项测试 =====")
            self.print_current_info()

            self._time_current = time()
            yield AutoTest.HyperParams(self.as_dict(*self._state))
            print(f"该项结束，用时 {self.item_runtime():.3f} 秒")
            flag = self.next_idx()

        print(f"\n所有测试项结束")
        self._running = False

    def print_current_info(self):
        for k, v in self.as_dict(*self._state).items():
            print(f"{k} = {v}")

    def save_record(self, filename: str):
        """Save records as a `.json` file.
        When using autosave, this can not save anything."""
        with open(filename, 'w', encoding='utf-8') as a:
            a.write(json.dumps(self._records, indent=4))


class _Saver():
    def __init__(self, filename: str, mode: Literal['append', 'replace']) -> None:
        self._filename = filename
        if mode in ('append', 'replace'):
            self._mode = mode
        else:
            raise ValueError(f'Unknown mode {mode}')
        self._data: List[Dict[str, Any]] = []

    @property
    def filename(self):
        return self._filename


    def start(self):
        """Start: Try to read old data from the file."""
        if self._mode == 'append':
            if os.path.exists(self._filename):
                with open(self._filename, 'r', encoding='utf-8') as a:
                    data = json.loads(a.read())
                    if not isinstance(data, list):
                        raise TypeError(f'Object in the existed file must be a list, but found {type(data)}.')
                    self._data = data


    def save(self, obj: Dict):
        self._data.append(obj)

        with open(self._filename, 'w', encoding='utf-8') as a:
            a.write(json.dumps(self._data, indent=4))


def _timer_core() -> Generator[None, str, List[Tuple[str, float]]]:
    label = "start"
    label_list: List[str] = []
    time_list: List[float] = []

    while True:
        label_list.append(label)
        time_list.append(time())
        if label == "stop":
            break
        label = yield

    return list(zip(label_list, time_list))


def timer() -> Generator[None, str, None]:
    while True:
        result = yield from _timer_core()
        print(f"Timer stopped.")
        print(
"=================================================\n"
"  ID     Label          Time         Proportion  \n"
"-------------------------------------------------"
    )
        prev_time = result[0][1]
        total_time = result[-1][1] - prev_time

        for i in range(1, len(result)):
            label = result[i][0]
            time = result[i][1]
            delta = time - prev_time
            prev_time = time
            p = delta / total_time * 100
            if len(label) < 12:
                label += " " * (12 - len(label))
            print(f"  {i}\t {label}\t{delta:.5f}     \t{p:.3f}%")

        print(
"================================================="
    )
