from importlib import import_module


timer_module = import_module("fealpy.utils.timer")


def test_timer_uses_perf_counter(monkeypatch, capsys):
    clock = iter((10.0, 10.25, 11.0, 20.0))
    monkeypatch.setattr(timer_module, "perf_counter", lambda: next(clock))

    tmr = timer_module.timer()
    next(tmr)
    tmr.send("read")
    tmr.send("convert")
    tmr.send(None)
    tmr.close()

    output = capsys.readouterr().out
    assert "250.000 [ms]" in output
    assert "750.000 [ms]" in output
    assert "25.000" in output
    assert "75.000" in output
    assert "read" in output
    assert "convert" in output
