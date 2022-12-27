from threading import Thread, Event
from queue import Queue, Empty, Full
from typing import Iterable, Callable


def t_feeder(
    it: Iterable,
    q: Queue,
    *,
    stop: Callable[[], bool] = (lambda: False),
    signal: Callable[[], None] = (lambda: None),
    throw: Callable[[Exception], None] = None,
    timeout: float = 0.5,
) -> None:
    """Feed the outputs from `next` into the queue `q`"""
    try:
        item = next(it)
        while not stop():
            try:
                q.put(item, True, timeout=timeout)
            except Full:
                continue
            item = next(it)

    except StopIteration:
        pass

    except BaseException as e:
        if not callable(throw):
            raise
        throw(e)

    finally:
        signal()


def t_worker(
    fn: Callable,
    qi: Queue,
    qo: Queue,
    *,
    stop: Callable[[], bool] = (lambda: False),
    depleted: Callable[[], bool] = (lambda: False),
    signal: Callable[[], None] = (lambda: None),
    throw: Callable[[Exception], None] = None,
    timeout: float = 0.5,
) -> None:
    try:
        # keep regularly checking the termination flag until we get a value
        while not stop():
            try:
                input = qi.get(True, timeout=timeout)
            except Empty:
                if depleted():
                    break
                continue

            qo.put(fn(input), True, timeout=timeout)

    except BaseException as e:
        if not callable(throw):
            raise
        throw(e)

    finally:
        signal()


def is_set(fun: Callable[[Iterable], bool], *evts: Event) -> Callable[[], bool]:
    def predicate() -> bool:
        return fun(e.is_set() for e in evts)

    return predicate


def threadpool_unordered_map(
    n_threads: int,
    target: Callable,
    jobs: Iterable,
    *,
    timeout: float = 0.5,
) -> Iterable:
    f_terminated, f_depleted, q_err = Event(), Event(), Queue()

    # the task to queue thread
    q_input, q_output = Queue(2 * n_threads), Queue()
    threads = [
        Thread(
            target=t_feeder,
            args=(iter(jobs), q_input),
            kwargs=dict(
                stop=f_terminated.is_set,
                signal=f_depleted.set,
                throw=q_err.put_nowait,
                timeout=timeout,
            ),
            daemon=True,
            name="Feeder",
        )
    ]

    # the evaluator threads
    f_signals = []
    for j in range(n_threads):
        f_sig = Event()
        threads.append(
            Thread(
                target=t_worker,
                args=(target, q_input, q_output),
                kwargs=dict(
                    stop=f_terminated.is_set,
                    depleted=f_depleted.is_set,
                    signal=f_sig.set,
                    throw=q_err.put_nowait,
                    timeout=timeout,
                ),
                daemon=True,
                name=f"Worker-{j:02d}",
            )
        )
        f_signals.append(f_sig)

    # check if all workers and the feeder have terminated
    stop = is_set(all, f_depleted, *f_signals)
    try:
        for t in threads:
            t.start()

        # the main thread yields results from the output queue
        while not stop():
            try:
                yield q_output.get(True, timeout=timeout)

            except Empty:
                pass

            # re-raise pending exceptions
            with q_err.mutex:
                if q_err.queue:
                    raise q_err.queue.popleft()

    finally:
        f_terminated.set()
        for t in threads:
            t.join()
