
from typing import List, Generator, Tuple
from time import time


def _timer_core() -> Generator[None, str, List[Tuple[str, float]]]:
    tag_list: List[str] = [None, ]
    time_list: List[float] = [time(), ]

    while True:
        tag = yield
        if tag is None:
            break
        tag_list.append(tag)
        time_list.append(time())

    return list(zip(tag_list, time_list))


def timer() -> Generator[None, str, None]:
    """Build a timer.

    Yields:
        None.
    """
    while True:
        result = yield from _timer_core()
        print(f"Timer received None and paused.")
        print(
"=================================================\n"
"   ID       Time        Proportion(%)    Label\n"
"-------------------------------------------------"
    )
        prev_time = result[0][1]
        total_time = result[-1][1] - prev_time

        for i in range(1, len(result)):
            i_text = f"{i}".rjust(3)
            label = result[i][0]
            time = result[i][1]

            delta = time - prev_time
            if delta > 1.0:
                time_text = f"{delta:.3f}".rjust(7) + " [s] "
            elif delta > 0.001:
                time_text = f"{delta*1e3:.3f}".rjust(7) + " [ms]"
            else:
                time_text = f"{delta*1e6:.3f}".rjust(7) + " [us]"

            prev_time = time
            p = delta / total_time * 100
            p_text = f"{p:.3f}".rjust(12)

            print("  " + "    ".join([i_text, time_text, p_text, label]))

        print(
"================================================="
    )
