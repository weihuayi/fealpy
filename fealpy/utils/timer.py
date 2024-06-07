
from typing import List, Dict, Literal, Optional, Any, Generator, Tuple
from time import time

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
