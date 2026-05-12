import pytest

import main


def _parse(argv):
    return main._build_parser().parse_args(argv)


def test_parser_requires_mode():
    with pytest.raises(SystemExit):
        _parse([])


def test_parser_evaluate_ok():
    args = _parse(["evaluate"])
    assert args.mode == "evaluate"
    assert args.data_dir.replace("\\", "/").endswith("test")


def test_parser_evaluate_with_data_dir(tmp_path):
    args = _parse(["evaluate", "--data-dir", str(tmp_path)])
    assert args.data_dir == str(tmp_path)


def test_parser_process_requires_input_and_output():
    with pytest.raises(SystemExit):
        _parse(["process"])
    with pytest.raises(SystemExit):
        _parse(["process", "--input", "x.wav"])
    with pytest.raises(SystemExit):
        _parse(["process", "--output", "x.csv"])


def test_parser_process_ok():
    args = _parse(["process", "--input", "in.wav", "--output", "out.csv"])
    assert args.mode == "process"
    assert args.input == "in.wav"
    assert args.output == "out.csv"


def test_parser_stream_requires_port_and_output():
    with pytest.raises(SystemExit):
        _parse(["stream"])
    with pytest.raises(SystemExit):
        _parse(["stream", "--port", "COM3"])
    with pytest.raises(SystemExit):
        _parse(["stream", "--output", "out.csv"])


def test_parser_stream_ok_with_defaults():
    args = _parse(["stream", "--port", "COM3", "--output", "out.csv"])
    assert args.mode == "stream"
    assert args.port == "COM3"
    assert args.timeout == 2.0


def test_parser_stream_custom_timeout():
    args = _parse(["stream", "--port", "COM3", "--output", "out.csv", "--timeout", "5.5"])
    assert args.timeout == pytest.approx(5.5)


@pytest.mark.parametrize("val", ["-0.1", "1.5", "abc"])
def test_parser_tau1_out_of_range_raises_systemexit(val):
    with pytest.raises(SystemExit):
        _parse(["evaluate", "--tau1", val])


@pytest.mark.parametrize("val", ["-0.1", "1.5", "abc"])
def test_parser_tau2_out_of_range_raises_systemexit(val):
    with pytest.raises(SystemExit):
        _parse(["evaluate", "--tau2", val])


@pytest.mark.parametrize("val", ["0.0", "1.0"])
def test_parser_tau_valid_boundaries(val):
    args = _parse(["evaluate", "--tau1", val, "--tau2", val])
    assert args.tau1 == float(val)
    assert args.tau2 == float(val)


def test_parser_tau_default_none():
    args = _parse(["evaluate"])
    assert args.tau1 is None
    assert args.tau2 is None
