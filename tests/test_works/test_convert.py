from hotpot.works import convert


class _RecordingMolecule:
    def __init__(self, *, has_metal=False):
        self.has_metal = has_metal
        self.build_options = None
        self.optimize_options = None
        self.write_calls = []
        self.ligand_copy = None

    def __copy__(self):
        self.ligand_copy = _RecordingMolecule()
        return self.ligand_copy

    def build3d(self, **options):
        self.build_options = options

    def remove_metals(self):
        self.has_metal = False

    def optimize(self, **options):
        self.optimize_options = options

    def write(self, path, fmt=None, **options):
        self.write_calls.append((path, fmt, options))


class _FinishedProcess:
    instances = []

    def __init__(self, *, target, args, kwargs):
        self.target = target
        self.args = args
        self.kwargs = kwargs
        self.exitcode = 0
        self.started = False
        self.join_calls = []
        self.terminate_calls = 0
        self.kill_calls = 0
        self.instances.append(self)

    def start(self):
        self.started = True

    def is_alive(self):
        return False

    def join(self, timeout=None):
        self.join_calls.append(timeout)

    def terminate(self):
        self.terminate_calls += 1

    def kill(self):
        self.kill_calls += 1


class _StubbornProcess:
    def __init__(self):
        self.alive = True
        self.calls = []

    def terminate(self):
        self.calls.append("terminate")

    def join(self, timeout=None):
        self.calls.append(("join", timeout))

    def is_alive(self):
        return self.alive

    def kill(self):
        self.calls.append("kill")
        self.alive = False


class _TimedOutProcess(_StubbornProcess):
    instances = []

    def __init__(self, *, target, args, kwargs):
        super().__init__()
        self.target = target
        self.args = args
        self.kwargs = kwargs
        self.exitcode = None
        self.instances.append(self)

    def start(self):
        return None


def test_build3d_writes_only_the_final_frame_by_default():
    molecule = _RecordingMolecule()

    convert._build3d(
        molecule,
        "result.sdf",
        "sdf",
        None,
        "preview.sdf",
        timeout=3.0,
    )

    assert molecule.build_options["timeout"] == 3.0
    assert [call[2]["write_single"] for call in molecule.write_calls] == [True, True]


def test_build3d_writes_movie_frames_for_complex_and_ligand():
    molecule = _RecordingMolecule(has_metal=True)

    convert._build3d(
        molecule,
        "complex.sdf",
        "sdf",
        "ligand.sdf",
        "preview.sdf",
        timeout=4.0,
        save_movie=True,
    )

    ligand = molecule.ligand_copy
    assert ligand.optimize_options["timeout"] == 4.0
    assert ligand.optimize_options["save_movie"] is True
    assert ligand.write_calls[0][2]["write_single"] is False
    assert [call[2]["write_single"] for call in molecule.write_calls] == [False, False]


def test_conversion_forwards_timeout_and_joins_natural_exit(monkeypatch, tmp_path):
    _FinishedProcess.instances = []
    molecule = _RecordingMolecule()
    monkeypatch.setattr(convert.hp, "MolReader", lambda *_: iter([molecule]))
    monkeypatch.setattr(convert.mp, "Process", _FinishedProcess)

    convert.convert_smiles_to_3dmol(
        ["CC"],
        str(tmp_path),
        nproc=1,
        timeout=2.5,
    )

    process = _FinishedProcess.instances[0]
    assert process.kwargs["timeout"] == 2.5
    assert process.join_calls == [None]
    assert process.terminate_calls == 0
    assert process.kill_calls == 0


def test_terminate_process_joins_then_kills_and_joins_again():
    process = _StubbornProcess()

    convert._terminate_process(process)

    assert process.calls == [
        "terminate",
        ("join", convert._PROCESS_SHUTDOWN_TIMEOUT),
        "kill",
        ("join", convert._PROCESS_SHUTDOWN_TIMEOUT),
    ]


def test_conversion_timeout_reaps_the_outer_worker(monkeypatch, tmp_path):
    _TimedOutProcess.instances = []
    molecule = _RecordingMolecule()
    clock = iter([0.0, 11.0])
    monkeypatch.setattr(convert.hp, "MolReader", lambda *_: iter([molecule]))
    monkeypatch.setattr(convert.mp, "Process", _TimedOutProcess)
    monkeypatch.setattr(convert.time, "monotonic", lambda: next(clock))

    convert.convert_smiles_to_3dmol(
        ["CC"],
        str(tmp_path),
        nproc=1,
        timeout=0.5,
    )

    assert _TimedOutProcess.instances[0].calls == [
        "terminate",
        ("join", convert._PROCESS_SHUTDOWN_TIMEOUT),
        "kill",
        ("join", convert._PROCESS_SHUTDOWN_TIMEOUT),
    ]
