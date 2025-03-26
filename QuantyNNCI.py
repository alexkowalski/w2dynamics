import pathlib
import os
import sys
import subprocess
import tempfile
import re
import gc
from textwrap import dedent
from math import ceil, fsum
from numbers import Number
from enum import Enum
from threading import Lock
from collections import deque

import numpy as np
from scipy.integrate import simpson
import keras
from bitstring import Bits


_NBITSKEY = 24
_QUANTY_PATCHED = False
_DEBUG_LOGFILES = None
# _DEBUG_LOGFILES = (open("/tmp/quanty.stdin", "a"),
#                    open("/tmp/quanty.stdout", "a"),
#                    open("/tmp/quanty.stderr", "a"))


RNG = np.random.default_rng()


_last_nf = None
def _auto_nf(nf=None):
    global _last_nf
    if nf is not None:
        assert nf == int(nf) > 0, "nf must be a positive integer"
        _last_nf = int(nf)
    return _last_nf


def bits_to_1darray(bits):
    return np.frombuffer(bits.tobytes(), dtype=np.uint8)


def bits_from_uint8array(arr, length):
    return Bits(bytes=arr.tobytes(), length=length)


def bitsset_to_2darray(bitsset, nf=None, boolarrays=True):
    if nf is None:
        nf = len(next(iter(bitsset)))
    result = np.fromiter(
        (np.frombuffer(bits.tobytes(), dtype=np.uint8) for bits in bitsset),
        dtype=np.dtype((np.uint8, ceil(nf/8))),
        count=len(bitsset)
    )
    if boolarrays:
        return np.unpackbits(result, axis=1, count=nf)
    else:
        return result


def wfdict_to_arrays(wfdict, nf=None, _complex=None, boolarrays=True):
    if nf is None:
        nf = len(next(iter(wfdict)))
    bitstrings = np.zeros((len(wfdict), ceil(nf / 8)),
                          dtype=np.uint8)
    coeffs = np.zeros((len(wfdict),),
                      dtype=(np.complex128
                             if _complex or _complex is None
                             else np.float64))
    for i, (k, v) in enumerate(wfdict.items()):
        bitstrings[i, :] = np.frombuffer(k.tobytes(), dtype=np.uint8)
        coeffs[i] = v
    if _complex is None and np.all(np.isreal(coeffs)):
        coeffs = np.real(coeffs)
    if boolarrays:
        bitstrings = np.unpackbits(bitstrings, axis=1, count=nf)
    return bitstrings, coeffs


class _QuantyManager:
    _singleton = None

    @classmethod
    def get_singleton(cls):
        if cls._singleton is None:
            cls._singleton = _QuantyManager()
        return cls._singleton

    def __init__(self):
        self._dir = pathlib.Path(__file__).parent
        self._process = subprocess.Popen(
            [
                #"valgrind", "-q", "--log-file=vg.log", "--leak-check=full",
                self._dir / 'Quanty',
                self._dir / 'quantyrepl.lua'
            ],
            text=True,
            bufsize=1,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )

        if _DEBUG_LOGFILES is not None:
            from types import MethodType
            stdin_write = self._process.stdin.write
            stdout_readline = self._process.stdout.readline
            stderr_readline = self._process.stderr.readline

            def tracked_stdin_write(self, *a, **kw):
                _DEBUG_LOGFILES[0].write(*a, **kw)
                stdin_write(*a, **kw)

            def tracked_stdout_readline(self, *a, **kw):
                out = stdout_readline(*a, **kw)
                _DEBUG_LOGFILES[1].write(out)
                return out

            def tracked_stderr_readline(self, *a, **kw):
                out = stderr_readline(*a, **kw)
                _DEBUG_LOGFILES[2].write(out)
                return out

            self._process.stdin.write = MethodType(tracked_stdin_write, self._process.stdin)
            self._process.stdout.readline = MethodType(tracked_stdout_readline, self._process.stdout)
            self._process.stderr.readline = MethodType(tracked_stderr_readline, self._process.stderr)

        self._quanty_preamble = ""
        line = self._process.stdout.readline()
        while line != "\n":
            self._quanty_preamble += line
            line = self._process.stdout.readline()

        self._varcounter = 0
        # self._livevars = []

        self._process_lock = Lock()
        self._defer_destroy_deque_lock = Lock()
        self._defer_destroy_deque = deque()

    def new_var(self, obj):
        self._varcounter += 1
        varname = f"pyautovar{self._varcounter}"
        # print(f"{varname} created", flush=True)
        # self._livevars.append(varname)
        return varname

    # def print_livevars(self):
    #     print(self._livevars, flush=True)

    def destroy_var(self, obj):
        # print(f"{obj._qvar} destruction requested", flush=True)
        if not self._process_lock.acquire(blocking=False):
            if not self._defer_destroy_deque_lock.acquire(blocking=False):
                raise RuntimeError("destroy_var can't acquire either lock")
            self._defer_destroy_deque.append(obj._qvar)
            self._defer_destroy_deque_lock.release()
            # print(f"{obj._qvar} destruction deferred", flush=True)
        else:
            self._process.stdin.write(f"{obj._qvar} = nil\n")
            self._quanty_finalize_command()
            self._quanty_get_output()
            self._quanty_check_error()
            self._process_lock.release()
            # print(f"{obj._qvar} destroyed", flush=True)
            # self._livevars.remove(obj._qvar)

    def _destroy_deferred(self):
        while True:
            with self._defer_destroy_deque_lock:
                try:
                    nextvar = self._defer_destroy_deque.popleft()
                except IndexError:
                    break
            self.run_command(f"{nextvar} = nil\n")
            # print(f"{nextvar} destroyed", flush=True)
            # self._livevars.remove(obj._qvar)

    def _quanty_finalize_command(self):
        """Write input terminator."""
        self._process.stdin.write("\nEOF\n")
        self._process.stdin.flush()

    def _quanty_check_error(self):
        error = ''
        line = self._process.stderr.readline()
        while line not in ("EOF\n", ""):
            error += line
            line = self._process.stderr.readline()

        error = error.strip()
        if len(error) > 0:
            raise RuntimeError(
                f"Got error message from Quanty: {error}"
            )

    def _quanty_get_output(self):
        output = ''
        line = self._process.stdout.readline()
        while line not in ("EOF\n", ""):
            output += line
            line = self._process.stdout.readline()

        return output

    def var_from_quanty_file(self, file, target):
        with self._process_lock:
            with open(file, "r") as f:
                # write only our variable name instead of that in the
                # first line, then just pipe the rest to quanty in
                # ~page-sized chunks
                firstline = f.readline()
                self._process.stdin.write(f"{target._qvar}")
                self._process.stdin.write(firstline[firstline.index('='):])
                self._process.stdin.flush()
                # srcfd = f.fileno()
                # while os.splice(srcfd, self._process.stdin.fileno(), 4096) > 0:
                #     pass
                while len(chunk := f.read(4096)) > 0:
                    self._process.stdin.write(chunk)
                    self._process.stdin.flush()
            self._quanty_finalize_command()
            output = self._quanty_get_output()
            self._quanty_check_error()
            return output
        self._destroy_deferred()

    def run_command(self, command, *args):
        with self._process_lock:
            self._process.stdin.write(command.format(*[arg._qvar for arg in args]))
            self._quanty_finalize_command()
            output = self._quanty_get_output()
            self._quanty_check_error()
        self._destroy_deferred()
        return output

    def stream_command_input(self, prelude_inputgen, *args):
        with self._process_lock:
            command, input_generator = prelude_inputgen
            self._process.stdin.write(command.format(*[arg._qvar for arg in args]))
            self._quanty_finalize_command()

            for inputstr in input_generator():
                self._process.stdin.write(inputstr)

            output = self._quanty_get_output()
            self._quanty_check_error()
        self._destroy_deferred()
        return output

    def parse_command_output(self, cmd_linefn_finfn, *args):
        with self._process_lock:
            command, process_line, finish = cmd_linefn_finfn
            self._process.stdin.write(command.format(*[arg._qvar for arg in args]))
            self._quanty_finalize_command()

            line = self._process.stdout.readline()
            while line != "EOF\n":
                process_line = process_line(line)
                line = self._process.stdout.readline()

            self._quanty_check_error()
        self._destroy_deferred()
        return finish()

    def parse_wf_from_quanty(self, wf):
        with self._process_lock:
            self._process.stdin.write(f"print({wf._qvar}.N)\n")
            self._quanty_finalize_command()
            targetlen = int(self._quanty_get_output())
            self._quanty_check_error()

            self._process.stdin.write(f"{wf._qvar}.Print()\n")
            self._quanty_finalize_command()

            wf.wfdict = {}
            wf.nf = None
            wf._complex = False

            line = self._process.stdout.readline()
            while not line.startswith("#"):
                if m := re.match(r"QComplex\s*=\s*([01])", line):
                    if m.group(1) == "0":
                        wf._complex = False
                    elif m.group(1) == "1":
                        wf._complex = True
                    else:
                        raise RuntimeError("Read unexpected QComplex value "
                                           "from Quanty")
                elif m := re.match(r"NFermionic modes\s*=\s*(\d+)", line):
                    wf.nf = int(m.group(1))
                line = self._process.stdout.readline()
            line = self._process.stdout.readline()

            count = 0
            if wf._complex:
                while len(res := line.split()) == 4:
                    # if Bits(hex=res[3])[:wf.nf] in wf.wfdict:
                    #     print("ERROR: duplicate entry")
                    #     print(line)
                    wf.wfdict[Bits(hex=res[3])[:wf.nf]] = (
                        float(res[1]) + 1.0j * float(res[2])
                    )
                    count += 1
                    line = self._process.stdout.readline()
            else:
                while len(res := line.split()) == 3:
                    # if Bits(hex=res[2])[:wf.nf] in wf.wfdict:
                    #     print("ERROR: duplicate entry")
                    #     print(line)
                    wf.wfdict[Bits(hex=res[2])[:wf.nf]] = float(res[1])
                    count += 1
                    line = self._process.stdout.readline()

            while line != "EOF\n":
                line = self._process.stdout.readline()

            self._quanty_check_error()
            if len(wf.wfdict) != targetlen:
                raise RuntimeError(f"Mismatch after parse between expected wf "
                                   f"length {targetlen} (len from quanty) and "
                                   f"received wf length {len(wf.wfdict)} / {count}")

        self._destroy_deferred()
        return wf

    def send_wf_to_quanty(self, wf, nbitskey=None):
        if nbitskey is None:
            nbitskey = wf._nbitskey
        FASTLOAD_REAL = dedent("""\
        function fastload_from_py_real(file, nf, nterms)
           local result = {}
           for i = 1, nterms, 1 do
              local bitstr = file:read("*l")
              local coeff = file:read("*n")
              file:read("*l")
              rawset(result, #result + 1, {bitstr, coeff})
           end
           result = NewWavefunction(nf, 0, result, {{"NBitsKey", """ + str(nbitskey) + """}})
           return result
        end
        """
        )
        FASTLOAD_COMPLEX = dedent("""\
        function fastload_from_py_complex(file, nf, nterms)
           local result = {}
           for i = 1, nterms, 1 do
              local bitstr = file:read("*l")
              local real = file:read("*n")
              file:read("*l")
              local imag = file:read("*n")
              file:read("*l")
              rawset(result, #result + 1, {bitstr, real + I * imag})
           end
           result = NewWavefunction(nf, 0, result, {{"NBitsKey", """ + str(nbitskey) + """}})
           return result
        end
        """
        )
        with self._process_lock:
            targetlen = len(wf.wfdict)

            if wf._complex:
                self._process.stdin.write(
                    FASTLOAD_COMPLEX +
                    f"{wf._qvar} = fastload_from_py_complex("
                    f"io.stdin, "
                    f"{wf.nf}, "
                    f"{targetlen})\n"
                )
                self._quanty_finalize_command()
                for k, v in wf.wfdict.items():
                    self._process.stdin.write(
                        f"{k.bin}\n{v.real}\n{v.imag}\n"
                    )
            else:
                self._process.stdin.write(
                    FASTLOAD_REAL +
                    f"{wf._qvar} = fastload_from_py_real("
                    f"io.stdin, "
                    f"{wf.nf}, "
                    f"{targetlen})\n",
                )
                self._quanty_finalize_command()
                for k, v in wf.wfdict.items():
                    self._process.stdin.write(
                        f"{k.bin}\n{v.real}\n"
                    )

            self._quanty_get_output()
            self._quanty_check_error()

            self._process.stdin.write(f"print({wf._qvar}.N)\n")
            self._quanty_finalize_command()
            quantylen = int(self._quanty_get_output())
            self._quanty_check_error()

            if quantylen != targetlen:
                raise RuntimeError(f"Mismatch after send between sent wf "
                                   f"{targetlen} and received wf length "
                                   f"{quantylen} (len from quanty)")

        self._destroy_deferred()
        return wf

    def _set_number(self, varname, number):
        if number.imag == 0:
            return self.run_command(
                f"{varname} = {number.real}\n"
            )
        else:
            return self.run_command(
                f"{varname} = {number.real} + I * {number.imag}\n"
            )

    def _get_number(self, varname):
        res = _qm().parse_command_output(
            (f"pyqmtempvar1 = {varname} + I * 0\n"
             "print(string.format('%25.17e', pyqmtempvar1.real))\n"
             "print(string.format('%25.17e', pyqmtempvar1.imag))\n"
             "pyqmtempvar1 = nil\n",
             *parse_nums()),
        )
        return (res[0] if res[1] == 0 else res[0] + 1.0j * res[1])

    def _execute_luafile(self, filename):
        return self.run_command(f"dofile('{filename}')\n")

    def __del__(self):
        if self.__class__._singleton is self:
            self.__class__._singleton = None
        try:
            self._process.stdin.write("os.exit(true)\nEOF\n")
            self._process.stdin.flush()
            try:
                self._process.communicate(timeout=5)
            except subprocess.TimeoutExpired:
                self._process.kill()
        except BrokenPipeError:
            self._process.kill()

def _qm():
    return _QuantyManager.get_singleton()


class _QuantyObject:
    def __init__(self, *args, **kwargs):
        self._qvar = _qm().new_var(self)
        super().__init__(*args, **kwargs)

    def clear(self):
        _qm().destroy_var(self)

    def __del__(self):
        try:
            self.clear()
        except (TypeError, BrokenPipeError):
            pass


def parse_nums():
    nums = []

    def process_line(line):
        if len(line := line.strip()) > 0:
            nums.append(float(line))
        return process_line

    def finish():
        return nums

    return process_line, finish


class Impl(Enum):
    QUANTY = 1
    PYTHON = 2


class WaveFunction(_QuantyObject):
    def __init__(self, nf=None, *args,
                 keep_quanty_synced=True,
                 keep_python_synced=False,
                 wfdict=None, nbitskey=_NBITSKEY, **kwargs):
        """
        nf: Number of fermionic single-particle states
        """
        nf = _auto_nf(nf)
        super().__init__(*args, **kwargs)
        self.nf = nf
        self.wfdict = {}
        self._complex = False
        self._nbitskey = nbitskey
        _qm().run_command(
            f"{{}} = NewWavefunction({self.nf}, 0, {{{{}}}}, {{{{{{{{'NBitsKey', {self._nbitskey}}}}}}}}})\n",
            self
        )

        self._ahead = None
        if not keep_quanty_synced:
            raise NotImplementedError(
                "not keeping Quanty synced would seriously affect "
                "functionality at the moment"
            )
        self._keep_quanty_synced = keep_quanty_synced
        self._keep_python_synced = keep_python_synced

        if wfdict is not None:
            if len(wfdict) > 0 and isinstance(next(iter(wfdict.items()))[0], Bits):
                self.wfdict = wfdict
                self._mark_python_ahead()
            else:
                for k, v in wfdict.items():
                    self.add_term(k, v)

    def is_quanty_ahead(self):
        return self._ahead == Impl.QUANTY

    def _mark_quanty_ahead(self):
        if self._ahead is not None and self._ahead != Impl.QUANTY:
            raise RuntimeError("attempting to mark Quanty ahead "
                               "while something else is ahead")
        self._ahead = Impl.QUANTY
        if self._keep_python_synced:
            self._parse_from_quanty()
        return self

    def is_python_ahead(self):
        return self._ahead == Impl.PYTHON

    def _mark_python_ahead(self):
        if self._ahead is not None and self._ahead != Impl.PYTHON:
            raise RuntimeError("attempting to mark Python ahead "
                               "while something else is ahead")
        self._ahead = Impl.PYTHON
        if self._keep_quanty_synced:
            self._send_to_quanty()
        return self

    def pysyncd(self):
        if self._ahead is not None and self._ahead != Impl.PYTHON:
            self._parse_from_quanty()
        return self

    def unload(self):
        if self.is_python_ahead():
            self._send_to_quanty()

        del self.wfdict
        self.wfdict = {}
        gc.collect()

        return self._mark_quanty_ahead()

    def qusyncd(self):
        if self._ahead is not None and self._ahead != Impl.QUANTY:
            self._send_to_quanty()
        return self

    def _len_from_quanty(self):
        return int(_qm().parse_command_output(
            ("print({}.N)\n",
             *parse_nums()),
            self
        )[0])

    def duplicate_hash_count(self):
        if not _QUANTY_PATCHED:
            raise NotImplementedError()
        return int(_qm().parse_command_output(
            ("print({}.NDuplicateHashes)\n",
             *parse_nums()),
            self
        )[0])

    def __len__(self):
        if self.is_python_ahead() or self._ahead is None:
            return len(self.wfdict)
        else:
            return self._len_from_quanty()

    def _nf_from_quanty(self):
        return int(_qm().parse_command_output(
            ("print({}.NF)\n",
             *parse_nums()),
            self
        )[0])

    def get_nf(self):
        if self.is_python_ahead() or self._ahead is None:
            return self.nf
        else:
            return self._nf_from_quanty()

    def get_dict(self):
        return self.pysyncd().wfdict

    def _complex_from_quanty(self):
        return _qm().run_command("print({}.Complex)\n", self).startswith("true")

    def is_complex(self):
        if self.is_python_ahead() or self._ahead is None:
            return self._complex
        else:
            return self._complex_from_quanty()

    def pprint(self, sort_descending=False):
        if self.is_quanty_ahead():
            self._parse_from_quanty()
        indexwidth = len(str(len(self) - 1))
        if self._complex:
            print(f"#{' ' * indexwidth}  {'determinant':>{self.nf-2}} {'real':>25} {'imag':>25}")
            if sort_descending:
                for i, k in enumerate(sorted(self.wfdict.keys(),
                                             key=lambda k: abs(self.wfdict[k]),
                                             reverse=True)):
                    v = self.wfdict[k]
                    print(f"{i:<{indexwidth}} {k.bin:>13} {v.real:25.17e} {v.imag:25.17e}")
            else:
                for i, (k, v) in enumerate(self.wfdict.items()):
                    print(f"{i:<{indexwidth}} {k.bin:>13} {v.real:25.17e} {v.imag:25.17e}")
        else:
            print(f"#{' ' * indexwidth}  {'determinant':>{self.nf-2}} {'real':>25}")
            if sort_descending:
                for i, k in enumerate(sorted(self.wfdict.keys(),
                                             key=lambda k: abs(self.wfdict[k]),
                                             reverse=True)):
                    v = self.wfdict[k]
                    print(f"{i:<{indexwidth}} {k.bin:>13} {v:25.17e}")
            else:
                for i, (k, v) in enumerate(self.wfdict.items()):
                    print(f"{i:<{indexwidth}} {k.bin:>13} {v:25.17e}")

    @classmethod
    def load_file(cls, file, quanty=False):
        if quanty:
            obj = cls(1)
            _qm().var_from_quanty_file(file, obj)
            obj._mark_quanty_ahead()
            return obj

        with np.load(file) as data:
            nf = int(data['nf'])
            obj = cls(nf)
            if np.iscomplexobj(data['coeffs']):
                obj._complex = True
            else:
                obj._complex = False
            for bytes, coeff in zip(data['bitstrings'], data['coeffs']):
                obj.wfdict[Bits(bytes=bytes.tobytes(), length=nf)] = coeff
            obj._mark_python_ahead()

        return obj

    def to_numpy_arrays(self, boolarrays=True):
        if self.is_quanty_ahead():
            self._parse_from_quanty()
        return wfdict_to_arrays(self.wfdict, self.nf, self._complex, boolarrays)

    def store_file(self, file):
        if self.is_quanty_ahead():
            self._parse_from_quanty()
        with open(file, "xb"):
            bitstrings, coeffs = self.to_numpy_arrays(boolarrays=False)
            np.savez_compressed(file,
                                nf=self.nf,
                                bitstrings=bitstrings,
                                coeffs=coeffs)

    def _parse_from_quanty(self, force=False, error_if_unneeded=True):
        if not force:
            if self._ahead == Impl.PYTHON:
                raise RuntimeError(
                    "Reading from Quanty would cause loss of information"
                )
            elif self._ahead is None:
                if error_if_unneeded:
                    raise RuntimeError(
                        "Reading from Quanty is not necessary"
                    )
                else:
                    return self

        _qm().parse_wf_from_quanty(self)
        self._ahead = None
        return self

    def _parse_from_quanty_slow(self, force=False, error_if_unneeded=True):
        if not force:
            if self._ahead == Impl.PYTHON:
                raise RuntimeError(
                    "Reading from Quanty would cause loss of information"
                )
            elif self._ahead is None:
                if error_if_unneeded:
                    raise RuntimeError(
                        "Reading from Quanty is not necessary"
                    )
                else:
                    return self

        targetlen = self._len_from_quanty()

        self.wfdict = {}
        self.nf = None
        self._complex = False
        # integer index, float coefficient, determinant bitstring
        realregex = re.compile(r"(?a)\s*\d+\s+"
                               r"([-+\d.eE]+)\s+"
                               r"([0-9ABCDEF]+)\s*")
        # integer index, float real and imag part, determinant bitstring
        cmplregex = re.compile(r"(?a)\s*\d+\s+"
                               r"([-+\d.eE]+)\s+"
                               r"([-+\d.eE]+)\s+"
                               r"([0-9ABCDEF]+)\s*")

        def process_header(line):
            if m := re.match(r"QComplex\s*=\s*([01])", line):
                if m.group(1) == "0":
                    self._complex = False
                elif m.group(1) == "1":
                    self._complex = True
                else:
                    raise RuntimeError("Read unexpected QComplex value "
                                       "from Quanty")
            elif m := re.match(r"NFermionic modes\s*=\s*(\d+)", line):
                self.nf = int(m.group(1))
            elif m := re.match(r"#", line):
                return (process_complex_term
                        if self._complex
                        else process_real_term)
            return process_header

        def process_real_term(line):
            if m := realregex.fullmatch(line):
                self.wfdict[
                    Bits(hex=m.group(2))[:self.nf]
                ] = float(m.group(1))
            return process_real_term

        def process_complex_term(line):
            if m := cmplregex.fullmatch(line):
                self.wfdict[
                    Bits(hex=m.group(3))[:self.nf]
                ] = float(m.group(1)) + 1.0j * float(m.group(2))
            return process_complex_term

        def finish():
            return None

        _qm().parse_command_output(("{}.Print()\n", process_header, finish),
                                   self)
        self._ahead = None
        if len(self.wfdict) != targetlen:
            raise RuntimeError(f"Mismatch after parse between expected wf "
                               f"length {targetlen} (len from quanty) and "
                               f"received wf length {len(wf.wfdict)}")

        return self

    def _send_to_quanty(self, force=False, error_if_unneeded=True):
        if not force:
            if self._ahead == Impl.QUANTY:
                raise RuntimeError(
                    "Writing to Quanty would cause loss of information"
                )
            elif self._ahead is None:
                if error_if_unneeded:
                    raise RuntimeError(
                        "Writing to Quanty is not necessary"
                    )
                else:
                    return self

        _qm().send_wf_to_quanty(self)
        self._ahead = None
        return self

    def _send_to_quanty_slow(self, force=False):
        FASTLOAD_REAL = dedent("""\
        function fastload_from_py_real(file, nf, nterms)
           local result = {{}}
           for i = 1, nterms, 1 do
              local bitstr = file:read("*l")
              local coeff = file:read("*n")
              file:read("*l")
              rawset(result, #result + 1, {{bitstr, coeff}})
           end
           result = NewWavefunction(nf, 0, result)
           return result
        end
        """
        )
        FASTLOAD_COMPLEX = dedent("""\
        function fastload_from_py_complex(file, nf, nterms)
           local result = {{}}
           for i = 1, nterms, 1 do
              local bitstr = file:read("*l")
              local real = file:read("*n")
              file:read("*l")
              local imag = file:read("*n")
              file:read("*l")
              rawset(result, #result + 1, {{bitstr, real + I * imag}})
           end
           result = NewWavefunction(nf, 0, result)
           return result
        end
        """
        )
        if not force:
            if self._ahead == Impl.QUANTY:
                raise RuntimeError(
                    "Writing to Quanty would cause loss of information"
                )
            elif self._ahead is None:
                raise RuntimeError(
                    "Writing to Quanty is not necessary"
                )

        targetlen = len(self.wfdict)

        if self._complex:
            def inputgen():
                for k, v in self.wfdict.items():
                    yield f"{k.bin}\n"
                    yield f"{v.real}\n"
                    yield f"{v.imag}\n"

            _qm().stream_command_input((FASTLOAD_COMPLEX +
                                        f"{{}} = fastload_from_py_complex("
                                        f"io.stdin, "
                                        f"{self.nf}, "
                                        f"{len(self.wfdict)})\n",
                                        inputgen), self)
        else:
            def inputgen():
                for k, v in self.wfdict.items():
                    yield f"{k.bin}\n"
                    yield f"{v.real}\n"

            _qm().stream_command_input((FASTLOAD_REAL +
                                        f"{{}} = fastload_from_py_real("
                                        f"io.stdin, "
                                        f"{self.nf}, "
                                        f"{len(self.wfdict)})\n",
                                        inputgen), self)
        self._ahead = None
        if (quantylen := self._len_from_quanty()) != targetlen:
            raise RuntimeError(f"Mismatch after send between sent wf "
                               f"{targetlen} and received wf length "
                               f"{quantylen} (len from quanty)")

    def add_term(self, detstr, coeff):
        if len(detstr) != (nf := self.get_nf()):
            raise ValueError(f"{len(detstr)=} but should be {nf}")
        if not isinstance(coeff, Number):
            raise ValueError("coeff must be a number")

        if self.is_python_ahead() or self._ahead is None:
            if isinstance(detstr, Bits):
                detbits = detstr
            else:
                detbits = Bits(bin=detstr, length=nf)
            current = self.wfdict.get(detbits,
                                      0.0)
            self.wfdict[detbits] = current + coeff

        if self.is_quanty_ahead() or self._ahead is None:
            if isinstance(detstr, Bits):
                detstr = detstr.bin
            if coeff.imag == 0:
                if _QUANTY_PATCHED:
                    _qm().run_command(
                        f"{{}}.IAdd(NewWavefunction({self.nf}, 0, {{{{{{{{'{detstr}', {coeff}}}}}}}}}, {{{{{{{{'NBitsKey', {self._nbitskey}}}}}}}}}))\n",
                        self
                    )
                else:
                    _qm().run_command(
                        f"{{}} = {{}} + NewWavefunction({self.nf}, 0, {{{{{{{{'{detstr}', {coeff}}}}}}}}}, {{{{{{{{'NBitsKey', {self._nbitskey}}}}}}}}})\n",
                        self, self
                    )

            else:
                self._complex = True
                if _QUANTY_PATCHED:
                    _qm().run_command(
                        f"{{}}.IAdd(NewWavefunction({self.nf}, 0, {{{{{{{{'{detstr}', {coeff.real} + I * {coeff.imag}}}}}}}}}, {{{{{{{{'NBitsKey', {self._nbitskey}}}}}}}}}))\n",
                        self
                    )
                else:
                    _qm().run_command(
                        f"{{}} = {{}} + NewWavefunction({self.nf}, 0, {{{{{{{{'{detstr}', {coeff.real} + I * {coeff.imag}}}}}}}}}, {{{{{{{{'NBitsKey', {self._nbitskey}}}}}}}}})\n",
                        self, self
                    )

    def normalize(self):
        self = self.qusyncd()
        _qm().run_command("{}.Normalize()\n", self)
        self._mark_quanty_ahead()
        return self

    def norm(self):
        self = self.qusyncd()
        if _QUANTY_PATCHED:
            norm = _qm().parse_command_output(
                (f"pytempvar1 = {{}}.Norm\n"
                 "print(string.format('%25.17e', pytempvar1))\n"
                 "pytempvar1 = nil\n",
                 *parse_nums()),
                self
            )[0]
        else:
            norm = _qm().parse_command_output(
                (f"pytempvar1 = sqrt({{}} * {{}}) + I * 0\n"
                 "print(string.format('%25.17e', pytempvar1.real))\n"
                 "pytempvar1 = nil\n",
                 *parse_nums()),
                self, self
            )[0]
        return norm

    def safer_norm(self):
        return fsum((abs(x)**2 for x in self.get_dict().values()))

    def safer_scalarproduct(self, other):
        if not isinstance(other, WaveFunction):
            return TypeError("safer_scalarproduct: other must be wf")

        selfdict = self.get_dict()
        otherdict = other.get_dict()
        if len(selfdict) > len(otherdict):
            smallerdict = other.get_dict()
            largerdict = self.get_dict()
        else:
            smallerdict = self.get_dict()
            largerdict = other.get_dict()
        if self.is_complex() or other.is_complex():
            return (fsum((selfdict[k].real * otherdict[k].real
                          for k in smallerdict))
                    - fsum((selfdict[k].imag * otherdict[k].imag
                            for k in smallerdict))
                    + 1.0j * fsum((selfdict[k].real * otherdict[k].imag
                                   for k in smallerdict))
                    + 1.0j * fsum((selfdict[k].imag * otherdict[k].real
                                   for k in smallerdict)))
        else:
            return fsum((v * largerdict[k] for k, v in smallerdict.items()))

    def apply_cutoff(self, cutoff):
        self = self.qusyncd()
        _qm().run_command(f"{{}}.Chop({cutoff})\n", self)
        self._mark_quanty_ahead()
        return self

    def get_maxnbasis_cutoff(self, nbasis_max):
        if nbasis_max == 0:
            return np.finfo(np.double).max
        if _QUANTY_PATCHED:
            self = self.qusyncd()
            cutoff = _qm().parse_command_output(
                (f"pytempvar1 = {{}}.MaxsizeCutoff({nbasis_max})\n"
                 "print(string.format('%25.17e', pytempvar1))\n"
                 "pytempvar1 = nil\n",
                 *parse_nums()),
                self
            )[0]
        else:
            if nbasis_max >= len(self):
                cutoff = 0.0
            else:
                self = self.pysyncd()
                coeffs = (
                    np.sort(np.abs(np.fromiter(self.get_dict().values(), dtype=np.complex128)), kind='stable')
                    if self.is_complex()
                    else
                    np.sort(np.abs(np.fromiter(self.get_dict().values(), dtype=np.float64)), kind='stable')
                )
                start = coeffs.size - nbasis_max - 1
                end = coeffs.size - nbasis_max + 1
                cutoff = np.mean(coeffs[start:end])
                del coeffs
                self.unload()
        return cutoff

    def apply_maxnbasis(self, nbasis_max):
        if len(self) <= nbasis_max:
            return self

        cutoff = self.get_maxnbasis_cutoff(nbasis_max)
        return self.apply_cutoff(cutoff)

    def truncate(self, cutoff=None, nbasis_max=None):
        maxcutoff = -1

        if cutoff is not None:
            maxcutoff = cutoff

        if nbasis_max is not None:
            maxcutoff = max(maxcutoff, self.get_maxnbasis_cutoff(nbasis_max))

        if maxcutoff >= 0.0:
            self.apply_cutoff(maxcutoff)

        return self

    def __add__(self, other):
        self = self.qusyncd()
        other = other.qusyncd()

        if not isinstance(other, WaveFunction):
            return NotImplemented
        res = self.__class__(self.nf)
        _qm().run_command("{} = {} + {}\n", res, self, other)
        res._mark_quanty_ahead()
        return res

    def __iadd__(self, other):
        self = self.qusyncd()

        if not isinstance(other, WaveFunction):
            return NotImplemented
        other = other.qusyncd()

        if _QUANTY_PATCHED:
            _qm().run_command("{}.IAdd({})\n", self, other)
        else:
            _qm().run_command("{} = {} + {}\n", self, self, other)

        self._mark_quanty_ahead()
        return self

    def __sub__(self, other):
        self = self.qusyncd()

        if not isinstance(other, WaveFunction):
            return NotImplemented
        other = other.qusyncd()

        res = self.__class__(self.nf)
        _qm().run_command("{} = {} - {}\n", res, self, other)
        res._mark_quanty_ahead()
        return res

    def __isub__(self, other):
        self = self.qusyncd()

        if not isinstance(other, WaveFunction):
            return NotImplemented
        other = other.qusyncd()

        if _QUANTY_PATCHED:
            _qm().run_command("{}.ISub({})\n", self, other)
        else:
            _qm().run_command("{} = {} - {}\n", self, self, other)

        self._mark_quanty_ahead()
        return self

    def __mul__(self, other):
        self = self.qusyncd()

        if isinstance(other, Number):
            res = self.__class__(self.nf)
            if other.imag == 0.0:
                _qm().run_command(f"{{}} = {other.real} * {{}}\n", res, self)
            else:
                _qm().run_command(f"{{}} = ({other.real} + I * {other.imag}) * {{}}\n",
                                  res, self)
            res._mark_quanty_ahead()
            return res
        else:
            return NotImplemented

    def __imul__(self, other):
        self = self.qusyncd()

        if isinstance(other, Number):
            if _QUANTY_PATCHED:
                if other.imag == 0.0:
                    _qm().run_command(f"{{}}.Scale({other.real})\n", self)
                else:
                    _qm().run_command(f"{{}}.Scale({other.real} + I * {other.imag})\n",
                                      self)
            else:
                if other.imag == 0.0:
                    _qm().run_command(f"{{}} = {{}} * {other.real}\n", self, self)
                else:
                    _qm().run_command(f"{{}} = {{}} * ({other.real} + I * {other.imag})\n",
                                      self, self)
            self._mark_quanty_ahead()
            return self
        else:
            return NotImplemented

    def __truediv__(self, other):
        self = self.qusyncd()

        if isinstance(other, Number):
            res = self.__class__(self.nf)
            if other.imag == 0.0:
                _qm().run_command(f"{{}} = {{}} / {other.real}\n", res, self)
            else:
                _qm().run_command(f"{{}} = {{}} / ({other.real} + I * {other.imag})\n",
                                  res, self)
            res._mark_quanty_ahead()
            return res
        else:
            return NotImplemented

    def __itruediv__(self, other):
        self = self.qusyncd()

        if isinstance(other, Number):
            if _QUANTY_PATCHED:
                other = 1.0 / other
                if other.imag == 0.0:
                    _qm().run_command(f"{{}}.Scale({other.real})\n", self)
                else:
                    _qm().run_command(f"{{}}.Scale({other.real} + I * {other.imag})\n",
                                      self)
            else:
                if other.imag == 0.0:
                    _qm().run_command(f"{{}} = {{}} / {other.real}\n", self, self)
                else:
                    _qm().run_command(f"{{}} = {{}} / ({other.real} + I * {other.imag})\n",
                                      self, self)
            self._mark_quanty_ahead()
            return self
        else:
            return NotImplemented

    def __rmul__(self, other):
        return self.__mul__(other)

    def __neg__(self):
        return -1.0 * self

    def __pos__(self):
        return self

    def __matmul__(self, other):
        self = self.qusyncd()

        if isinstance(other, WaveFunction):
            other = other.qusyncd()
            # need temporary var for complex number and add I * 0 in case it is real
            res = _qm().parse_command_output(
                ("pytempvar1 = {} * {} + I * 0\n"
                 "print(string.format('%25.17e', pytempvar1.real))\n"
                 "print(string.format('%25.17e', pytempvar1.imag))\n"
                 "pytempvar1 = nil\n",
                 *parse_nums()),
                 self, other
            )
            return (res[0] if res[1] == 0 else res[0] + 1.0j * res[1])
        elif isinstance(other, Operator):
            res = self.__class__(self.nf)
            _qm().run_command("{} = {} * {}\n", res, self, other)
            res._mark_quanty_ahead()
            return res
        else:
            return NotImplemented

    def __imatmul__(self, other):
        self = self.qusyncd()

        if isinstance(other, Operator):
            _qm().run_command("{} = {} * {}\n", self, self, other)
            self._mark_quanty_ahead()
            return self
        else:
            return NotImplemented

    def __deepcopy__(self, memo=None):
        res = self.__class__(self.nf)
        res.wfdict = self.wfdict.copy()
        res._complex = self._complex
        res._ahead = self._ahead
        res._keep_python_synced = self._keep_python_synced
        res._keep_quanty_synced = self._keep_quanty_synced
        _qm().run_command("{} = Clone({})\n", res, self)
        return res


class Operator(_QuantyObject):
    # TODO: as for WaveFunction
    def __init__(self, nf=None, *args, **kwargs):
        """
        nf: Number of fermionic single-particle states
        """
        nf = _auto_nf(nf)
        super().__init__(*args, **kwargs)
        self.nf = nf
        # TODO: uninitialized should maybe be made unusable (note: NO
        # IT SHOULD NOT), or at least the unit matrix prefactor taken
        # as a required argument
        _qm().run_command(
            f"{{}} = NewOperator({self.nf}, 0, {{{{{{{{0}}}}}}}})\n",
            self
        )

    def get_nf(self):
        return self.nf

    def print_from_quanty(self):
        print(_qm().run_command("{}.Print()\n", self))

    def eigh(self, psis, preserve=True):
        if isinstance(psis, WaveFunction):
            if preserve:
                psis = psis.__deepcopy__()
            _qm().run_command("Eigensystem({}, {}, "
                              "{{{{'ExpandBasis', false}}, "
                              "{{'Epsilon', 0}}, "
                              "{{'Zero', 1e-14}}}})\n",
                              self, psis)
            psis._mark_quanty_ahead()
            return psis
        else:
            # Assume sequence (or iterable) of wave functions
            psis = [psi.__deepcopy__() if preserve else psi
                    for psi in psis]

            _qm().run_command("pytempvar1 = {{}}\n")
            for i, wf in enumerate(psis):
                _qm().run_command(f"pytempvar1[{i + 1}] = {{}}\n", wf)

            _qm().run_command("Eigensystem({}, pytempvar1, "
                              "{{{{'ExpandBasis', false}}, "
                              "{{'Epsilon', 0}}, "
                              "{{'Zero', 1e-14}}}})\n",
                              self)

            for psi in psis:
                psi._mark_quanty_ahead()
            _qm().run_command("pytempvar1 = nil\n")
            return psis

    def braket(self, bras, kets):
        if isinstance(bras, WaveFunction) and isinstance(kets, WaveFunction):
            # need temporary var for complex number and add I * 0 in case it is real
            res = _qm().parse_command_output(
                ("pytempvar1 = Braket({}, {}, {}) + I * 0\n"
                 "print(string.format('%25.17e', pytempvar1.real))\n"
                 "print(string.format('%25.17e', pytempvar1.imag))\n"
                 "pytempvar1 = nil\n",
                 *parse_nums()),
                bras, self, kets
            )
            return (res[0] if res[1] == 0 else res[0] + 1.0j * res[1])

        # assume iterable(s)
        _qm().run_command("pytempvar1 = {{}}\n")
        for i, wf in enumerate(bras if hasattr(bras, '__iter__') else [bras]):
            _qm().run_command(f"pytempvar1[{i + 1}] = {{}}\n", wf)
        _qm().run_command("pytempvar2 = {{}}\n")
        for j, wf in enumerate(kets if hasattr(kets, '__iter__') else [kets]):
            _qm().run_command(f"pytempvar2[{j + 1}] = {{}}\n", wf)

        if bras == kets:
            # Call BraketDiagonal; note that BraketDiagonal would not
            # in fact give "expected" results anyway if iterating over
            # bras and kets results in actually different wave
            # functions, which is why the interface does not let you
            # choose which one to use yourself
            _qm().run_command(
                "pytempvar2 = BraketDiagonal(pytempvar1, {}, pytempvar2)\n",
                self
            )

            res = []
            for m in range(i + 1):
                val = _qm().parse_command_output(
                    (f"pytempvar1 = pytempvar2[{m + 1}] + I * 0\n"
                     f"print(string.format('%25.17e', pytempvar1.real))\n"
                     f"print(string.format('%25.17e', pytempvar1.imag))\n"
                     f"pytempvar1 = nil\n",
                     *parse_nums()),
                )
                res.append(val[0]
                           if val[1] == 0.0
                           else val[0] + 1.0j * val[1])
            _qm().run_command("pytempvar1 = nil\n"
                              "pytempvar2 = nil\n")
            return res
        else:
            _qm().run_command(
                "pytempvar2 = Braket(pytempvar1, {}, pytempvar2)\n",
                self
            )
            res = []
            for m in range(i + 1):
                row = []
                res.append(row)
                for n in range(j + 1):
                    val = _qm().parse_command_output(
                        (f"pytempvar1 = pytempvar2[{m + 1}][{n + 1}] + I * 0\n"
                         f"print(string.format('%25.17e', pytempvar1.real))\n"
                         f"print(string.format('%25.17e', pytempvar1.imag))\n"
                         f"pytempvar1 = nil\n",
                         *parse_nums()),
                    )
                    row.append(val[0]
                               if val[1] == 0.0
                               else val[0] + 1.0j * val[1])
            _qm().run_command("pytempvar1 = nil\n"
                              "pytempvar2 = nil\n")
            return res

    @staticmethod
    def op_c(index, nf=None):
        nf = _auto_nf(nf)
        res = Operator(nf)
        _qm().run_command(
            f"{{}} = NewOperator('An', {nf}, {index})\n",
            res
        )
        return res

    @staticmethod
    def op_cdag(index, nf=None):
        nf = _auto_nf(nf)
        res = Operator(nf)
        _qm().run_command(
            f"{{}} = NewOperator('Cr', {nf}, {index})\n",
            res
        )
        return res

    @staticmethod
    def op_number(index, nf=None):
        nf = _auto_nf(nf)
        res = Operator(nf)
        _qm().run_command(
            f"{{}} = NewOperator('Number', {nf}, {index}, {index})\n",
            res
        )
        return res

    def __add__(self, other):
        if isinstance(other, Operator):
            res = self.__class__(self.nf)
            _qm().run_command("{} = {} + {}\n", res, self, other)
            return res
        elif isinstance(other, Number):
            res = self.__class__(self.nf)
            if other.imag == 0.0:
                _qm().run_command(f"{{}} = {{}} + {other.real}\n", res, self)
            else:
                _qm().run_command(f"{{}} = {{}} + ({other.real} +  I * {other.imag})\n",
                                  res, self)
            return res
        else:
            return NotImplemented

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        if isinstance(other, Operator):
            res = self.__class__(self.nf)
            _qm().run_command("{} = {} - {}\n", res, self, other)
            return res
        elif isinstance(other, Number):
            res = self.__class__(self.nf)
            if other.imag == 0.0:
                _qm().run_command(f"{{}} = {{}} - {other.real}\n", res, self)
            else:
                _qm().run_command(f"{{}} = {{}} - ({other.real} +  I * {other.imag})\n",
                                  res, self)
            return res
        else:
            return NotImplemented

    def __rsub__(self, other):
        if isinstance(other, Number):
            res = self.__class__(self.nf)
            if other.imag == 0.0:
                _qm().run_command(f"{{}} = {other.real} - {{}}\n", res, self)
            else:
                _qm().run_command(f"{{}} = ({other.real} +  I * {other.imag}) - {{}}\n",
                                  res, self)
            return res
        else:
            return NotImplemented

    def __mul__(self, other):
        if isinstance(other, Number):
            res = self.__class__(self.nf)
            if other.imag == 0.0:
                _qm().run_command(f"{{}} = {other.real} * {{}}\n", res, self)
            else:
                _qm().run_command(f"{{}} = ({other.real} + I * {other.imag}) * {{}}\n",
                                  res, self)
            return res
        else:
            return NotImplemented

    def __truediv__(self, other):
        if isinstance(other, Number):
            res = self.__class__(self.nf)
            if other.imag == 0.0:
                _qm().run_command(f"{{}} = {{}} / {other.real}\n", res, self)
            else:
                _qm().run_command(f"{{}} = {{}} / ({other.real} + I * {other.imag})\n",
                                  res, self)
            return res
        else:
            return NotImplemented

    def __rmul__(self, other):
        return self.__mul__(other)

    def __neg__(self):
        return -1.0 * self

    def __pos__(self):
        return self

    def __matmul__(self, other):
        if isinstance(other, Operator):
            res = self.__class__(self.nf)
            _qm().run_command("{} = {} * {}\n", res, self, other)
            return res
        elif isinstance(other, WaveFunction):
            res = other.__class__(self.nf)
            _qm().run_command("{} = {} * {}\n", res, self, other)
            res._mark_quanty_ahead()
            return res
        else:
            return NotImplemented

    def __deepcopy__(self, memo=None):
        res = self.__class__(self.nf)
        _qm().run_command("{} = Clone({})\n", res, self)
        return res


def _setup_old_model(modelfile, nbath, nfill=None,
                     eb=0.0, ef=-np.sqrt(10), t=1.0, V=np.sqrt(0.1),
                     U=np.sqrt(10), mu=0, eoffset=0):
    _qm()._set_number("NBATH", nbath)
    if nfill is None:
        nfill = nbath + 1
    _qm()._set_number("Nfill", nfill)
    _qm()._set_number("E_b", eb)
    _qm()._set_number("Ef", ef)
    _qm()._set_number("t", t)
    _qm()._set_number("V", V)
    _qm()._set_number("U", U)
    _qm()._set_number("mu", mu)
    _qm()._set_number("Eoffset", eoffset)
    _qm()._execute_luafile(modelfile)
    H = Operator(_qm()._get_number("NF"))
    E_BATH = [_qm()._get_number(f"Ef"), _qm()._get_number(f"Ef")]
    V_BATH = [0, 0]
    for i in range(2, 2 * nbath + 2):
        E_BATH.append(_qm()._get_number(f"E_BATH[{i}]"))
        V_BATH.append(_qm()._get_number(f"V_BATH[{i}]"))
    _qm().run_command("{} = Ham\nHam = nil\n", H)
    return H, E_BATH, V_BATH


def initial_singlet_guess(nbath=None):
    nf = _auto_nf(2 * (nbath + 1) if nbath is not None else None)
    nbath = nf // 2 - 1
    wfguess = WaveFunction(
        nf,
        wfdict={
            '10' + (nbath - 1) * '1' + '01' + (nbath - 1) * '0': 1/np.sqrt(2),
            '01' + (nbath - 1) * '1' + '10' + (nbath - 1) * '0': -1/np.sqrt(2),
        }
    )
    return wfguess


def initial_guess(impuritypos, nbath=None, fillings=None, spins=None, fillspins=None):
    nf = _auto_nf(2 * nbath + len(impuritypos) if nbath is not None else None)
    if nbath is None:
        nbath = (nf - len(impuritypos)) // 2

    if fillspins is None:
        if spins is None:
            spins = (0,)
        if fillings is None:
            fillings = (nf//2,)

        fillspins = []
        for spin in spins:
            for filling in fillings:
                fillspins.append((filling, spin))

    fillspins = [(f, s) for f, s in list(set(fillspins))
                 if 0 < f < nf and abs(s) <= nf//2 and abs(s) <= f]

    basislist = []
    for filltarget, spintarget in fillspins:
        # start by filling the first filltarget states
        guess = bytearray(b'1' * filltarget + b'0' * (nf - filltarget))

        if (guess[1::2].count(b'1') - guess[0::2].count(b'1') - spintarget) % 2 == 1:
            # we can only change the spin (= 2 * spin qn) by even
            # amounts, so there are no states with this
            # combination of filling and spin
            continue
        while (spin := guess[1::2].count(b'1') - guess[0::2].count(b'1')) != spintarget:
            # if the spin is not spintarget, move one electron at
            # a time to get it there while preferring to occupy
            # states at the beginning
            if spin > spintarget:
                highest_occ_up = guess[1::2].rfind(b'1') * 2 + 1
                lowest_free_down = guess[0::2].find(b'0') * 2
                if highest_occ_up < 0 or lowest_free_down < 0:
                    break
                guess[highest_occ_up] = ord('0')
                guess[lowest_free_down] = ord('1')
            else:
                highest_occ_down = guess[0::2].rfind(b'1') * 2
                lowest_free_up = guess[1::2].find(b'0') * 2 + 1
                if highest_occ_down < 0 or lowest_free_up < 0:
                    break
                guess[highest_occ_down] = ord('0')
                guess[lowest_free_up] = ord('1')
        if (guess[1::2].count(b'1') - guess[0::2].count(b'1')) != spintarget:
            continue

        basislist.append(guess)

        # add extra guesses with combinations of different
        # possible occupations of impurity orbitals (assuming bath
        # levels are energy-ordered, impurity levels and
        # interaction could be very significant for the right
        # choice)
        for setting in range(1, 2**len(impuritypos)):
            extraguess = bytearray(guess)
            for pos in impuritypos:
                setting, changepos = divmod(setting, 2)
                mod = pos % 2
                if changepos:
                    if extraguess[pos] == ord('1'):
                        swaptarget = b'0'
                    else:
                        swaptarget = b'1'

                    swappos = 2 * extraguess[mod::2].find(swaptarget) + mod
                    while swappos in impuritypos:
                        swappos = 2 * extraguess[mod::2].find(swaptarget, swappos + 1) + mod

                    if swappos < 0:
                        break

                    extraguess[pos], extraguess[swappos] = (
                        extraguess[swappos], extraguess[pos]
                    )
            else:
                basislist.append(extraguess)

    if len(basislist) == 0:
        raise ValueError("no states for any combination of provided fillings and spins")
    value = 1/len(basislist) if len(basislist) != 0 else 0
    wfguess = WaveFunction(
        nf,
        wfdict={
            string.decode('ascii'): value for string in basislist
        }
    )
    return wfguess


def step_extend_diagonalize(H, wfgs, maxadd=np.inf, maxtot=np.inf, log=lambda x: None, extend=True):
    if extend:
        wfguess = H @ wfgs
        log(f"step_extend_diagonalize: basis size before/after extension {len(wfgs)}/{len(wfguess)}")
    else:
        log(f"step_extend_diagonalize: requested diagonalization only of provided {len(wfgs)}")
        wfguess = wfgs
    wfguess = H.eigh(wfguess, preserve=False)

    if len(wfguess) > maxtot:
        wfguess.apply_maxnbasis(maxtot)
        log(f"step_extend_diagonalize: result basis truncated to {len(wfguess)} due to {maxtot=}")
        wfguess = H.eigh(wfguess, preserve=False)

    if maxadd < len(wfguess) - len(wfgs):
        oldbasis = wfgs.get_dict().keys()
        guessdict = wfguess.get_dict()
        addedbasis = guessdict.keys() - oldbasis
        if len(addedbasis) <= maxadd:
            return wfguess
        newcomponents = [(k, guessdict[k])
                         for k in addedbasis]
        newcomponents.sort(key=lambda x: abs(x[1]), reverse=True)
        newcomponents = newcomponents[:maxadd]
        # diagonalizing again is rather heavy, maybe dropping
        # complement from wfguess and normalizing would be enough
        # (certainly if we only need the ground state as input for
        # another step)
        wfguess = wfgs.__deepcopy__()
        guessdict = wfguess.get_dict()
        guessdict |= dict(newcomponents)
        wfguess._send_to_quanty(force=True)
        log(f"step_extend_diagonalize: result basis truncated to {len(wfguess)} due to {maxadd=}")
        wfguess = H.eigh(wfguess, preserve=False)

    return wfguess


def step_extend_semirandom(H, wfgs, wfpool, num_add_optimal, num_add_random, log=lambda x: None):
    oldbasis = wfgs.get_dict().keys()

    candidatebasis = (
        wfpool
        .get_dict()
        .keys()
        - oldbasis
    )

    num_newstates = len(candidatebasis)
    wffull = wfgs + WaveFunction(
        wfgs.get_nf(),
        wfdict={s: 1e-8
                for s in candidatebasis}
    )
    log(f"step_extend_semirandom: basis sizes (gs in, pool in, total in) {len(wfgs)} {len(wfpool)} {len(wffull)}")
    wffull = H.eigh(wffull, preserve=False)

    if num_newstates <= num_add_optimal + num_add_random:
        log(f"step_extend_semirandom: result on total basis fitting requested limit")
        return wffull

    fulldict = wffull.get_dict()
    del wffull

    candidatebasis = [(k, fulldict[k]) for k in candidatebasis]
    candidatebasis.sort(key=lambda x: abs(x[1]), reverse=True)
    optaddbasis, randomaddbasis = (candidatebasis[:num_add_optimal],
                                   candidatebasis[num_add_optimal:])
    del candidatebasis
    RNG.shuffle(randomaddbasis)
    randomaddbasis = randomaddbasis[:num_add_random]
    log(f"step_extend_semirandom: result basis extended by optimal / random {len(optaddbasis)} {len(randomaddbasis)}")

    return H.eigh(wfgs
                  + WaveFunction(wfgs.get_nf(),
                                 wfdict=dict(optaddbasis + randomaddbasis)),
                  preserve=False)


def diagonalize_random_partial_extension(H, wfgs, wfext, nprediag, log=lambda x: None):
    oldbasis = wfgs.get_dict().keys()
    old_length = len(oldbasis)

    # extend
    candidatebasis = (
        wfext
        .get_dict()
        .keys()
        - oldbasis
    )
    extra_length = len(candidatebasis)
    log(f"diagonalize_random_partial_extension: basis (gs in, ext in, selectable prediag)) {len(wfgs)} {len(wfext)} {extra_length}")

    # randomly select 'prediag' new basis states to include in
    # diagonalization
    prediagbasis = RNG.choice(bitsset_to_2darray(candidatebasis, wfext.get_nf(), boolarrays=False),
                              size=nprediag,
                              replace=False,
                              axis=0,
                              shuffle=False)

    log(f"diagonalize_random_partial_extension: selected prediag {len(prediagbasis)}")

    value = 1/max(2, prediagbasis.shape[0])
    nf = wfgs.get_nf()
    prediagbasis = WaveFunction(
        nf,
        wfdict={bits_from_uint8array(bitsarray, nf): value
                for bitsarray in prediagbasis}
    )
    prediagbasis._send_to_quanty(force=True)

    # diagonalize
    wfguess = prediagbasis
    del prediagbasis
    wfguess += wfgs
    wfguess = H.eigh(wfguess, preserve=False)

    return wfguess, extra_length


def lanczos_matrix_simple(op, ntri, H, wfgs, log=lambda x: None, cache_a_b_vecs=None):
    if (cache_a_b_vecs is not None
        and len(cache_a_b_vecs) > 0
        and len(cache_a_b_vecs[-1]) == 4):
            log("lanczos_matrix: Continuing from provided vectors")
            a, b, fcurr, fnext = cache_a_b_vecs.pop()
    else:
        log("lanczos_matrix: Starting by applying op to wfgs")
        fnext = op @ wfgs
        log("lanczos_matrix: Calculating normalization factor")
        fnext2 = np.sqrt(fnext @ fnext)
        fnext /= fnext2
        fcurr = WaveFunction(fnext.get_nf())
        log("lanczos_matrix: Computing H expectation value")
        a = [H.braket(fnext, fnext)]
        b = [fnext2]

    for i in range(ntri):
        fprev = fcurr
        fcurr = fnext
        log(f"lanczos_matrix: Next Krylov basis vector, iteration {i}")
        fnext = H @ fcurr - a[-1] * fcurr - b[-1] * fprev
        log("lanczos_matrix: Calculating normalization factor")
        fnext2 = np.sqrt(fnext @ fnext)
        fnext /= fnext2
        log("lanczos_matrix: Computing H expectation value")
        a.append(H.braket(fnext, fnext))
        b.append(fnext2)

        if cache_a_b_vecs is not None:
            cache_a_b_vecs.append((a, b, fcurr, fnext))
            try:
                cache_a_b_vecs.pop(-2)
            except IndexError:
                pass

    return np.array(a), np.array(b)


def lanczos_matrix(op, ntri, H, wfgs, log=lambda x, **kw: None,
                   cache_a_b_vecs=None, cutoff=None, maxns=None):
    log(f"lanczos_matrix: truncation options {cutoff=}, {maxns=},", end=" ")
    if (cache_a_b_vecs is not None
        and len(cache_a_b_vecs) > 0
        and len(cache_a_b_vecs[0]) == 4):
            log("start with cache,", end=" ", flush=True)
            a, b, fcurr, fnext = cache_a_b_vecs.pop(0)
            log("{a.size=}", end=" ", flush=True)
            hfnext = (H @ fnext).truncate(cutoff, maxns)
    else:
        log("start with op @ wfgs,", end=" ", flush=True)
        fnext = (op @ wfgs).truncate(cutoff, maxns)
        log("ff", end=" ", flush=True)
        fnext2 = fnext.norm()
        log("n", end=" ", flush=True)
        fnext /= fnext2
        fcurr = WaveFunction(fnext.get_nf())
        log("Hf", end=" ", flush=True)
        hfnext = (H @ fnext).truncate(cutoff, maxns)
        log("fHf", end=" ", flush=True)
        a = [fnext @ hfnext]
        b = [fnext2]

    for i in range(ntri):
        #_qm().print_livevars()
        fcurr, fnext = fnext, fcurr
        fcurr *= a[-1]
        #_qm().print_livevars()
        log(f"|{i}>", end=" ", flush=True)
        fnext += hfnext
        fnext -= fcurr
        log("ff", end=" ", flush=True)
        fnext2 = fnext.norm()
        log("n", end=" ", flush=True)
        fnext /= fnext2
        log("Hf", end=" ", flush=True)
        hfnext = (H @ fnext).truncate(cutoff, maxns)
        log("fHf", end=" ")
        log(f"{len(hfnext)=}", end=" ", flush=True)
        a.append(fnext @ hfnext)
        b.append(fnext2)
        fcurr *= (-b[-1]/a[-2])

    if cache_a_b_vecs is not None:
        fcurr.unload()
        fnext.unload()
        cache_a_b_vecs.append((a, b, fcurr, fnext))

    log("done", flush=True)
    return np.array(a), np.array(b)


def lanczos_eigh(op, ntri, H, wfgs, log=lambda x: None, cache_a_b_vecs=None,
                 cutoff=None, maxns=None):
    a, b = lanczos_matrix(op, ntri, H, wfgs, log=log,
                          cache_a_b_vecs=cache_a_b_vecs,
                          cutoff=cutoff, maxns=maxns)
    evals, evecs = np.linalg.eigh(np.diagflat(a, 0)
                                  + np.diagflat(b[1:], -1))
    return evals, evecs, b[0]


def lanczos_greens_function(index, H, wfgs, ntri, egs=None, log=lambda x: None,
                            cache=(None, None), cutoff=None, maxns=None):
    wfgs.unload()
    if egs is None:
        egs = H.braket(wfgs, wfgs)

    log(f"lanczos_greens_function: op = c_{index},", end=" ")
    c_evals, c_evecs, c_b0 = lanczos_eigh(Operator.op_c(index, H.get_nf()),
                                          ntri, H, wfgs, log=log,
                                          cache_a_b_vecs=cache[0],
                                          cutoff=cutoff, maxns=maxns)
    c_gs_comp = c_evecs[0, :]

    log(f"lanczos_greens_function: op = cdag_{index},", end=" ")
    cdag_evals, cdag_evecs, cdag_b0 = lanczos_eigh(Operator.op_cdag(index, H.get_nf()),
                                                   ntri, H, wfgs, log=log,
                                                   cache_a_b_vecs=cache[1],
                                                   cutoff=cutoff, maxns=maxns)
    cdag_gs_comp = cdag_evecs[0, :]

    def greens_function(omega):
        omega = np.asarray(omega)

        def match_shapes(eigq):
            return np.reshape(eigq,
                              (*eigq.shape,
                               *(1 for _ in omega.shape)))

        ccomp = match_shapes(c_gs_comp)
        cdcomp = match_shapes(cdag_gs_comp)
        cval = match_shapes(c_evals)
        cdval = match_shapes(cdag_evals)

        return np.sum(
            np.abs(cdcomp * cdag_b0)**2
            / (omega[np.newaxis, ...] + egs - cdval)
            + np.abs(ccomp * c_b0)**2
            / (omega[np.newaxis, ...] - egs + cval),
            axis=0
        )

    return greens_function


def hdist(bits1, bits2):
    """Hamming distance of two bitstrings"""
    return (bits1 ^ bits2).count(1)


def chebyshev_moments(op, nmom, H, wfgs, wmin, wmax, egs=None):
    if egs is None:
        egs = H.braket(wfgs, wfgs)

    H = (H - egs - wmax) * (2 / (wmax - wmin)) + 1
    fbase = op @ wfgs
    #fbase /= np.sqrt(fbase @ fbase)
    fcurr = fbase
    fnext = H @ fcurr
    #fnext /= np.sqrt(fnext @ fnext)
    moms = [fbase @ fbase / 2, fbase @ fnext] + [0] * (2 * nmom - 1)
    for i in range(2, nmom + 1):
        fprev = fcurr
        fcurr = fnext
        fnext = (2 * H) @ fcurr - fprev
        #fnext /= np.sqrt(fnext @ fnext)
        moms[i] = fbase @ fnext
        if i >= nmom // 2:
            moms[2 * i - 1] = 2 * (fnext @ fcurr) - moms[1]
            moms[2 * i] = 2 * (fnext @ fnext) - 2 * moms[0]
    return np.array(moms)


def mgs(phis):
    """Modified Gram-Schmidt orthonormalization of a list of wave functions."""
    for i in range(len(phis)):
        phi = phis[i]
        for prior_phi in phis[:i]:
            phi -= (phi @ prior_phi) * prior_phi
        phi /= np.sqrt(phi @ phi)
        phis[i] = phi


def chebyshev_moments_reorth(op, nmom_base, nmom_re, H, wfgs, wmin, wmax, egs=None, log=lambda x: None):
    if egs is None:
        egs = H.braket(wfgs, wfgs)

    H = (H - egs - wmax) * (2 / (wmax - wmin)) + 1
    psis = []
    psis.append(op @ wfgs)
    basefactor = np.sqrt(psis[0] @ psis[0])
    psis.append(H @ psis[-1])
    for i in range(2, nmom_base + 1):
        log(f"Cheb reorth phase 1: generating psi {i}")
        psis.append((2 * H) @ psis[-1] - psis[-2])
    mgs(psis)

    Hmat = np.zeros((len(psis), len(psis)), dtype=np.complex128)
    for i in range(len(psis)):
        H_psi_i = H @ psis[i]
        for j in range(i, len(psis)):
            Hmat[i, j] = (H_psi_i @ psis[j] + np.conj(psis[j] @ H_psi_i)) / 2
            if i == j:
                Hmat[i, j] = np.real(Hmat[i, j])
            Hmat[j, i] = np.conj(Hmat[i, j])
    print(f"{np.sqrt(np.sum(np.abs(Hmat)**2))=}")

    fbase = np.zeros((len(psis),), dtype=np.complex128)
    fbase[0] = basefactor # FIXME: or normalized 1.0 ?
    #fbase /= np.sqrt(fbase @ fbase)
    fcurr = fbase
    fnext = Hmat @ fcurr
    #fnext /= np.sqrt(fnext @ fnext)
    moms = [np.conj(fbase) @ fbase / 2, np.conj(fbase) @ fnext] + [0] * (2 * nmom_re - 1)
    for i in range(2, nmom_re + 1):
        fprev = fcurr
        fcurr = fnext
        fnext = (2 * Hmat) @ fcurr - fprev
        print(f"{np.sqrt(np.sum(np.abs(fnext)**2))=}")
        #fnext /= np.sqrt(fnext @ fnext)
        moms[i] = np.conj(fbase) @ fnext
        if i >= nmom_re // 2:
            moms[2 * i - 1] = 2 * (np.conj(fnext) @ fcurr) - moms[1]
            moms[2 * i] = 2 * (np.conj(fnext) @ fnext) - 2 * moms[0]
    return np.array(moms)


def chebyshev_spectral_function(index, H, wfgs, nmom, wmin, wmax, egs=None, damping=None):
    if damping is None:
        damping = lambda n: lorentz_damping_factor(n, 2 * nmom + 1, 4)
    if egs is None:
        egs = H.braket(wfgs, wfgs)

    c_moms = chebyshev_moments(Operator.op_c(index, H.get_nf()),
                               nmom, -H, wfgs, wmin, wmax)
    cdag_moms = chebyshev_moments(Operator.op_cdag(index, H.get_nf()),
                                  nmom, H, wfgs, wmin, wmax)

    a = (wmax - wmin) / 2
    sf = (np.polynomial.chebyshev.Chebyshev(c_moms * damping(np.arange(2 * nmom + 1)))
          + np.polynomial.chebyshev.Chebyshev(cdag_moms * damping(np.arange(2 * nmom + 1))))/a

    def spectrum(omega):
        return sf(omega / a)

    return spectrum


def chebyshev_spectral_function_reorth(index, H, wfgs, nmom_base, nmom_re, wmin, wmax, egs=None, damping=None, log=lambda x: None):
    if damping is None:
        damping = lambda n: lorentz_damping_factor(n, 2 * nmom_re + 1, 4)
    if egs is None:
        egs = H.braket(wfgs, wfgs)

    c_moms = chebyshev_moments_reorth(Operator.op_c(index, H.get_nf()),
                                      nmom_base, nmom_re, -H, wfgs, wmin, wmax, log=log)
    cdag_moms = chebyshev_moments_reorth(Operator.op_cdag(index, H.get_nf()),
                                         nmom_base, nmom_re, H, wfgs, wmin, wmax, log=log)

    a = (wmax - wmin) / 2
    sf = (np.polynomial.chebyshev.Chebyshev(c_moms * damping(np.arange(2 * nmom_re + 1)))
          + np.polynomial.chebyshev.Chebyshev(cdag_moms * damping(np.arange(2 * nmom_re + 1))))/a

    def spectrum(omega):
        return sf(omega / a)

    return spectrum


def jackson_damping_factor(n, N):
    return ((N - n + 1) * np.cos(np.pi * n / (N + 1)) + np.sin(np.pi * n / (N + 1)) / np.tan(np.pi / (N + 1))) / (N + 1)


def lorentz_damping_factor(n, N, lamda):
    return np.sinh(lamda * (1 - n / N)) / np.sinh(lamda)


def nn_loop_old(model, fit_kwargs, H, wfgs, steps_full,
                amounts_nn_add, amounts_nn_prediag,
                maxlim_nn_add, maxlim_nn_prediag,
                # only stop full diagonalization steps early when the
                # wave function reaches size
                full_iter_maxsize=None,
                full_iter_truncsize=None,
                # do multiple extensions to ensure minimum size of the pool
                nn_pool_minsize=None,
                # for addition of states after prediction, take by
                # 'cutoff' or by 'amount'
                take_by='amount',
                # nn output: 'logcoeff' for regression for
                # log(coefficient), 'categorical' for >< cutoff
                nn_output='logcoeff',
                diagonalize_after_every_nn=False,
                diagonalize_after_last_nn=True,
                log=lambda x: None,
                nn_iter_callbacks=(), nn_extra_channels=()):

    try:
        wfgs = wfgs.pop()  # calling code does not need to keep a reference
    except:
        pass

    lastsize = None
    for i in range(steps_full):
        lastsize = len(wfgs)
        if full_iter_maxsize is not None and lastsize >= full_iter_maxsize:
            log(f"{lastsize=} > {full_iter_maxsize=}, proceeding to NN selection steps")
            break
        log(f"Starting extension-diagonalization step {i} (full extension)")
        wfgs = step_extend_diagonalize(H, wfgs, log=log)
        gc.collect()

    if full_iter_truncsize is not None and len(wfgs) > full_iter_truncsize:
        log(f"truncating full ext. wf according to {full_iter_truncsize=}")
        wfgs.truncate(nbasis_max=full_iter_truncsize)

    wf_primary = wfgs

    histories = []
    reports = []
    for i, (amount_nn_add, amount_nn_prediag,
            maxlim_nn_add, maxlim_nn_prediag) in enumerate(zip(
                amounts_nn_add, amounts_nn_prediag, maxlim_nn_add, maxlim_nn_prediag)):
        for fn in nn_iter_callbacks:
            fn()

        lastsize = len(wfgs)
        report = {'basis_size_start': lastsize}
        log(f"Starting step {i} (NN, prediag {amount_nn_prediag}, add {amount_nn_add})")

        # Diagonalize with a random selection from the new basis
        # states obtained by applying H
        wf_extended = H @ wfgs
        last_extended_len = 0
        while (nn_pool_minsize is not None
               and (extended_len := len(wf_extended)) - lastsize < nn_pool_minsize):
            if extended_len == last_extended_len:
                log("Apparently no further extension possible, "
                    "proceeding in spite of unfulfilled nn_pool_minsize")
                break
            log(f"{extended_len=} - {lastsize=} < {nn_pool_minsize=}, continuing with extension")
            wf_extended = H @ wf_extended
            last_extended_len = extended_len

        log(f"Extended basis size {len(wf_extended)}")
        report['basis_size_extended'] = len(wf_extended)
        num_avail = report['basis_size_extended'] - report['basis_size_start']
        report['available_unselected'] = num_avail

        def auto_rel(amount, reference):
            if amount is None:
                return None
            if 0 < float(amount) < 1:
                return int(float(amount) * reference)
            return int(amount)

        amount_nn_add = auto_rel(amount_nn_add, num_avail)
        amount_nn_prediag = auto_rel(amount_nn_prediag, num_avail)
        if (maxlim_nn_add := auto_rel(maxlim_nn_add, num_avail)) is not None:
            amount_nn_add = min(amount_nn_add, maxlim_nn_add)
        if (maxlim_nn_prediag := auto_rel(maxlim_nn_prediag, num_avail)) is not None:
            amount_nn_prediag = min(amount_nn_prediag, maxlim_nn_prediag)

        report.update({'requested_add': amount_nn_add,
                       'requested_prediag': amount_nn_prediag})
        report['ratio_req_add_of_available'] = amount_nn_add / report['available_unselected']
        wf_prediag, extra_length = diagonalize_random_partial_extension(
            H, wfgs, wf_extended, amount_nn_prediag, log=log
        )
        report['basis_size_prediag'] = len(wf_prediag)

        # Get just the non-primary states (for training) and then just
        # new states (for cutoff condition / sorting)
        primary_keys = wf_primary.get_dict().keys()
        wfdict_nonprim = {k: v for k, v in wf_prediag.get_dict().items()
                          if k not in primary_keys}
        gs_keys = wfgs.get_dict().keys()
        wfdict_new = {k: v for k, v in wfdict_nonprim.items()
                      if k not in gs_keys}
        del wfgs
        gc.collect()

        nonprim_bs, nonprim_cs = wfdict_to_arrays(wfdict_nonprim,
                                                  wf_prediag.get_nf(),
                                                  wf_prediag.is_complex(),
                                                  boolarrays=True)

        new_bs, new_cs = wfdict_to_arrays(wfdict_new,
                                          wf_prediag.get_nf(),
                                          wf_prediag.is_complex(),
                                          boolarrays=True)

        # Selecting according to the cutoff or categorical (below /
        # above cutoff) classification requires a cutoff, taking
        # exactly the requested amount according to log(|coeff|)
        # regression does not
        if take_by == 'cutoff' or nn_output == 'categorical':
            new_sort = np.argsort(np.abs(new_cs)**2)[::-1]
            new_bs = new_bs[new_sort]
            new_cs = new_cs[new_sort]

            target_fraction = amount_nn_add / extra_length
            target_of_new = min(int(target_fraction * new_cs.shape[0]), new_cs.shape[0] - 1)
            cutoffAbsSq = np.mean(np.abs(new_cs)[target_of_new : target_of_new + 1]**2)
            cutoffAbs = np.sqrt(cutoffAbsSq)
            log(f"Determined cutoff: {cutoffAbsSq=}, {cutoffAbs=}")
            report["cutoff_abs"] = cutoffAbs
            report["cutoff_abs_sq"] = cutoffAbsSq
        else:
            cutoffAbsSq = None
            cutoffAbs = None

        # Train on non-primary basis states
        # x_train = nonprim_bs
        # y_train = nonprim_cs

        # Add extra channels
        def aec(xdata):
            return (np.stack((xdata,
                              *[np.repeat(ch[np.newaxis, :],
                                          xdata.shape[0],
                                          axis=0)
                                for ch in nn_extra_channels]),
                             axis=-1)
                    if len(nn_extra_channels) > 0
                    else xdata)

        # FIXME: train not just on new states, but still keep states
        # used for training before out of validation and test sets
        # Train on new states
        dataset_indices = np.arange(new_cs.shape[0])
        RNG.shuffle(dataset_indices)
        x_train = aec(new_bs[dataset_indices[:int(0.7*dataset_indices.size)], :])
        y_train = new_cs[dataset_indices[:int(0.7*dataset_indices.size)]]

        x_validate = aec(new_bs[dataset_indices[int(0.7*dataset_indices.size):int(0.85*dataset_indices.size)], :])
        y_validate = new_cs[dataset_indices[int(0.7*dataset_indices.size):int(0.85*dataset_indices.size)]]
        y_validate = (-np.log(np.abs(y_validate))
                      if nn_output == 'logcoeff'
                      else keras.utils.to_categorical(np.abs(y_validate) >= cutoffAbs))

        x_test = aec(new_bs[dataset_indices[int(0.85*dataset_indices.size):], :])
        y_test = new_cs[dataset_indices[int(0.85*dataset_indices.size):]]
        y_test = (-np.log(np.abs(y_test))
                   if nn_output == 'logcoeff'
                   else keras.utils.to_categorical(np.abs(y_test) >= cutoffAbs))

        log(f"Training set size: {x_train.shape[0]}")
        report["train_size_total"] = x_train.shape[0]
        if cutoffAbs is not None:
            report["train_size_above_cutoff"] = np.sum((np.abs(y_train) >= cutoffAbs))
            log(f"Training set elements above cutoff: {report['train_size_above_cutoff']}")
            report["ratio_train_above_cutoff"] = report["train_size_above_cutoff"] / y_train.shape[0]
        del nonprim_bs, nonprim_cs
        gc.collect()

        y_train = (-np.log(np.abs(y_train))
                   if nn_output == 'logcoeff'
                   else keras.utils.to_categorical(np.abs(y_train) >= cutoffAbs))

        if nn_output == 'categorical':
            histories.append(model.fit(x_train, y_train,
                                       **fit_kwargs,
                                       validation_data=(x_validate, y_validate),
                                       class_weight={0: 0.5 / (1.0 - report["ratio_train_above_cutoff"]),
                                                     1: 0.5 / report["ratio_train_above_cutoff"]}))
        else:
            histories.append(model.fit(x_train, y_train,
                                       **fit_kwargs,
                                       validation_data=(x_validate, y_validate)))
        log("Test set:")
        model.evaluate(x_test, y_test, verbose=2)

        # Predict for basis states after extension not included in
        # last diagonalization
        x_predict = bitsset_to_2darray(wf_extended.get_dict().keys()
                                       - wf_prediag.get_dict().keys(),
                                       nf=wf_extended.get_nf(), boolarrays=True)
        report["nn_predicted"] = x_predict.shape[0]
        del wf_extended
        gc.collect()
        y_predict = model.predict(aec(x_predict)
                                  if len(nn_extra_channels) > 0
                                  else x_predict,
                                  verbose=2)

        nn_added = 0

        if nn_output == 'logcoeff':
            y_predict = np.reshape(y_predict, (y_predict.shape[0],))

            if take_by == 'cutoff':
                cutoffMinusLogAbs = -np.log(cutoffAbs)
                wfgs = wf_prediag.pysyncd()
                del wf_prediag
                gc.collect()

                gs_dict = wfgs.get_dict()
                gs_nf = wfgs.get_nf()
                for bs, cs in zip(x_predict, y_predict):
                    if cs <= cutoffMinusLogAbs:
                        bits = Bits(bytes=np.packbits(bs).tobytes(), length=gs_nf)
                        assert bits not in gs_dict, \
                            "Prediction calculated for state in prediag wf"
                        gs_dict[bits] = np.exp(-cs)
                        nn_added += 1

                wfgs._mark_python_ahead()
                wfgs.apply_cutoff(cutoffAbs)
                report["nn_added"] = nn_added
                report["ratio_nn_added_of_predicted"] = (nn_added / report["nn_predicted"]
                                                         if report["nn_predicted"] != 0
                                                         else 0.0)
            elif take_by == 'amount':
                new_bs = np.concatenate((new_bs, x_predict), axis=0)
                new_cs = np.concatenate((new_cs, np.exp(-y_predict)), axis=0)
                new_sort = np.argsort(np.abs(new_cs))[::-1]
                new_bs = new_bs[new_sort]
                new_cs = new_cs[new_sort]

                wfgs = wf_prediag.pysyncd()
                del wf_prediag
                gc.collect()

                gs_dict = wfgs.get_dict()
                gs_nf = wfgs.get_nf()
                for i, (bs, cs) in enumerate(zip(new_bs, new_cs)):
                    bits = Bits(bytes=np.packbits(bs).tobytes(), length=gs_nf)
                    if i < amount_nn_add:
                        gs_dict[bits] = cs
                    elif bits in gs_dict:
                        del gs_dict[bits]

                wfgs._mark_python_ahead()
            else:
                raise ValueError(f"take_by must be cutoff or amount, not {take_by}")

        elif nn_output == 'categorical':

            if take_by == 'cutoff':
                wfgs = wf_prediag.pysyncd()
                del wf_prediag
                gc.collect()

                gs_dict = wfgs.get_dict()
                gs_nf = wfgs.get_nf()
                for bs, cs in zip(x_predict, y_predict):
                    if cs[0] <= cs[1]:
                        nn_added += 1
                        bits = Bits(bytes=np.packbits(bs).tobytes(), length=gs_nf)
                        assert bits not in gs_dict, \
                            "Prediction calculated for state in prediag wf"
                        # FIXME: maybe figure out sth better?
                        gs_dict[bits] = max(cutoffAbs + 1e-8, 1e-8)

                wfgs._mark_python_ahead()
                wfgs.apply_cutoff(cutoffAbs)
                report["nn_added"] = nn_added
                report["ratio_nn_added_of_predicted"] = nn_added / report["nn_predicted"]
            elif take_by == 'amount':
                raise NotImplementedError(f"{nn_output=} and {take_by=}")
            else:
                raise ValueError(f"take_by must be cutoff or amount, not {take_by}")

        else:
            raise ValueError(f"nn_output must be logcoeff or categorical, not {nn_output}")

        nbasis_before_final_cutoff = len(wfgs)
        wfgs.unload()
        gc.collect()

        if diagonalize_after_every_nn:
            wfgs = H.eigh(wfgs, preserve=False)
            if cutoffAbs is not None:
                wfgs.apply_cutoff(cutoffAbs)
        else:
            wfgs.normalize()

        report["total_added"] = len(wfgs) - report["basis_size_start"]
        report["ratio_total_added_of_available"] = (report["total_added"] / report['available_unselected']
                                                    if report["available_unselected"] != 0
                                                    else 0.0)

        if cutoffAbs is not None:
            report["removed_by_final_cutoff"] = nbasis_before_final_cutoff - len(wfgs)
            report["ratio_final_cutoff_of_nnadded"] = (report["removed_by_final_cutoff"] / report["nn_added"]
                                                       if report["nn_added"] != 0
                                                       else 0.0)

        reports.append(report)
        log(report)

    if diagonalize_after_last_nn and not diagonalize_after_every_nn:
        wfgs = H.eigh(wfgs, preserve=False)
    return wfgs


def wf_filter_hdist(refbits, wf, hdistmin, hdistmax=None, log=lambda x: None):
    if hdistmax is None:
        hdistmax = hdistmin

    newwf = WaveFunction(
        wf.get_nf(),
        wfdict={k: v for k, v in wf.get_dict().items()
                if hdistmin <= hdist(refbits, k) <= hdistmax}
    )

    log(f"{refbits.bin=} {len(wf)=} {len(newwf)=}")

    return newwf


def extendloop_full_range(H, wf_init, optimal_per_prepstep, random_per_prepstep,
                          prepsteps_adjacent=None, prepsteps_full=2,
                          log=lambda x: None):
    wfgs, wf_init = wf_init, None

    if prepsteps_adjacent is None:
        prepsteps_adjacent = wfgs.get_nf()

    def get_max_contrib(wfdict):
        maxabscoeff = -np.inf
        basisstate = None
        for k, v in wfdict.items():
            if abs(v) > maxabscoeff:
                maxabscoeff = abs(v)
                basisstate = k
        return basisstate

    refbits = get_max_contrib(wfgs.get_dict())
    log(f"Reference state {refbits.bin=}")

    mindist = 1
    maxdist = 3
    for i in range(prepsteps_adjacent):
        log(f"Starting step {i} (take {optimal_per_prepstep} opt, {random_per_prepstep} random)")
        log(f"{len(wfgs)=} {mindist=} {maxdist=}")
        wfgs = step_extend_semirandom(H, wfgs,
                                      wf_filter_hdist(refbits, H @ wfgs,
                                                      mindist, maxdist,
                                                      log=log),
                                      optimal_per_prepstep,
                                      random_per_prepstep,
                                      log=log)
        mindist += 1
        maxdist += 1

    for i in range(prepsteps_full):
        log(f"Starting step {i} (take {optimal_per_prepstep} opt, {random_per_prepstep} random)")
        log(f"{len(wfgs)=}, no hdist restriction")
        wfgs = step_extend_semirandom(H, wfgs,
                                      H @ wfgs,
                                      optimal_per_prepstep,
                                      random_per_prepstep,
                                      log=log)

    log(f"{len(wfgs)=} before applying zero cutoff")
    wfgs.apply_cutoff(0)
    log(f"{len(wfgs)=} after applying zero cutoff")
    return wfgs


def nn_loop_experimental(model, fit_kwargs, H, wfgs,
                         amounts_nn_add, amounts_nn_prediag,
                         # for addition of states after prediction, take by
                         # 'cutoff' or by 'amount'
                         take_by='amount',
                         # nn output: 'logcoeff' for regression for
                         # log(coefficient), 'categorical' for >< cutoff
                         nn_output='logcoeff',
                         diagonalize_after_every_nn=False,
                         diagonalize_after_last_nn=True,
                         log=lambda x: None,
                         nn_extra_channels=()):
    try:
        wf_primary = wfgs.pop()  # calling code does not need to keep a reference
    except:
        wf_primary = wfgs

    histories = []
    for i, (amount_nn_add, amount_nn_prediag) in enumerate(zip(amounts_nn_add, amounts_nn_prediag)):
        log(f"Starting step {i} (NN, prediag {amount_nn_prediag}, add {amount_nn_add})")
        # Diagonalize with a random selection from the new basis
        # states obtained by applying H
        wf_extended = H @ wfgs
        log(f"Extended basis size {len(wf_extended)}")
        wf_prediag, extra_length = diagonalize_random_partial_extension(
            H, wfgs, wf_extended, amount_nn_prediag, log=log
        )

        # Get just ALL states for training and then just
        # new states (for cutoff condition / sorting)
        #  primary_keys = wf_primary.get_dict().keys()
        wfdict_nonprim = wf_prediag.get_dict().copy()
        gs_keys = wfgs.get_dict().keys()
        wfdict_new = {k: v for k, v in wfdict_nonprim.items()
                      if k not in gs_keys}
        del wfgs

        nonprim_bs, nonprim_cs = wfdict_to_arrays(wfdict_nonprim,
                                                  wf_prediag.get_nf(),
                                                  wf_prediag.is_complex(),
                                                  boolarrays=True)

        new_bs, new_cs = wfdict_to_arrays(wfdict_new,
                                          wf_prediag.get_nf(),
                                          wf_prediag.is_complex(),
                                          boolarrays=True)

        # Selecting according to the cutoff or categorical (below /
        # above cutoff) classification requires a cutoff, taking
        # exactly the requested amount according to log(|coeff|)
        # regression does not
        if take_by == 'cutoff' or nn_output == 'categorical':
            new_sort = np.argsort(np.abs(new_cs)**2)[::-1]
            new_bs = new_bs[new_sort]
            new_cs = new_cs[new_sort]

            target_fraction = amount_nn_add / extra_length
            target_of_new = min(int(target_fraction * new_cs.shape[0]), new_cs.shape[0] - 1)
            cutoffAbsSq = np.mean(np.abs(new_cs)[target_of_new : target_of_new + 1]**2)
            cutoffAbs = np.sqrt(cutoffAbsSq)
            log(f"Determined cutoff: {cutoffAbsSq=}, {cutoffAbs=}")
        else:
            cutoffAbsSq = None
            cutoffAbs = None

            # Add extra channels
        def aec(xdata):
            return (np.stack((xdata,
                              *[np.repeat(ch[np.newaxis, :],
                                          xdata.shape[0],
                                          axis=0)
                                for ch in nn_extra_channels]),
                             axis=-1)
                    if len(nn_extra_channels) > 0
                    else xdata)

        # Train on non-primary basis states
        x_train = aec(nonprim_bs)
        y_train = (-np.log(np.abs(nonprim_cs))
                   if nn_output == 'logcoeff'
                   else keras.utils.to_categorical(np.abs(nonprim_cs) >= cutoffAbs))

        log(f"Training set size: {nonprim_bs.shape[0]}")
        if cutoffAbs is not None:
            log(f"Training set elements above cutoff: {np.sum((np.abs(nonprim_cs) >= cutoffAbs))}")
        del nonprim_bs, nonprim_cs

        histories.append(model.fit(x_train, y_train, **fit_kwargs))

        # Predict for basis states after extension not included in
        # last diagonalization
        x_predict = bitsset_to_2darray(wf_extended.get_dict().keys()
                                       - wf_prediag.get_dict().keys(),
                                       nf=wf_extended.get_nf(), boolarrays=True)
        del wf_extended
        y_predict = model.predict(aec(x_predict), verbose=2)

        if nn_output == 'logcoeff':
            y_predict = np.reshape(y_predict, (y_predict.shape[0],))

            if take_by == 'cutoff':
                cutoffMinusLogAbs = -np.log(cutoffAbs)
                wfgs = wf_prediag.pysyncd()
                del wf_prediag

                gs_dict = wfgs.get_dict()
                gs_nf = wfgs.get_nf()
                for bs, cs in zip(x_predict, y_predict):
                    if cs <= cutoffMinusLogAbs:
                        bits = Bits(bytes=np.packbits(bs).tobytes(), length=gs_nf)
                        assert bits not in gs_dict, \
                            "Prediction calculated for state in prediag wf"
                        gs_dict[bits] = np.exp(-cs)

                wfgs._send_to_quanty(force=True)
                wfgs.apply_cutoff(cutoffAbs)
            elif take_by == 'amount':
                new_bs = np.concatenate((new_bs, x_predict), axis=0)
                new_cs = np.concatenate((new_cs, np.exp(-y_predict)), axis=0)
                new_sort = np.argsort(np.abs(new_cs))[::-1]
                new_bs = new_bs[new_sort]
                new_cs = new_cs[new_sort]

                wfgs = wf_prediag.pysyncd()
                del wf_prediag

                gs_dict = wfgs.get_dict()
                gs_nf = wfgs.get_nf()
                for i, (bs, cs) in enumerate(zip(new_bs, new_cs)):
                    bits = Bits(bytes=np.packbits(bs).tobytes(), length=gs_nf)
                    if i < amount_nn_add:
                        gs_dict[bits] = cs
                    elif bits in gs_dict:
                        del gs_dict[bits]

                wfgs._send_to_quanty(force=True)
            else:
                raise ValueError(f"take_by must be cutoff or amount, not {take_by}")

        elif nn_output == 'categorical':

            if take_by == 'cutoff':
                wfgs = wf_prediag.pysyncd()
                del wf_prediag

                gs_dict = wfgs.get_dict()
                gs_nf = wfgs.get_nf()
                for bs, cs in zip(x_predict, y_predict):
                    if cs[0] <= cs[1]:
                        bits = Bits(bytes=np.packbits(bs).tobytes(), length=gs_nf)
                        assert bits not in gs_dict, \
                            "Prediction calculated for state in prediag wf"
                        # FIXME: maybe figure out sth better?
                        gs_dict[bits] = max(cutoffAbs + 1e-8, 1e-8)

                wfgs._send_to_quanty(force=True)
                wfgs.apply_cutoff(cutoffAbs)
            elif take_by == 'amount':
                raise NotImplementedError(f"{nn_output=} and {take_by=}")
            else:
                raise ValueError(f"take_by must be cutoff or amount, not {take_by}")

        else:
            raise ValueError(f"nn_output must be logcoeff or categorical, not {nn_output}")

        if diagonalize_after_every_nn:
            wfgs = H.eigh(wfgs, preserve=False)
        else:
            wfgs.normalize()
    if diagonalize_after_last_nn and not diagonalize_after_every_nn:
        wfgs = H.eigh(wfgs, preserve=False)
    return wfgs


def gloc_from_hk(omega, hk, mu=0, sigma=0, eta=0):
    # input:
    # - omega(idx_o): frequency array (real / Matsubara)
    # - hk(idx_k, flavor, flavor): Hamiltonian in k-space
    # - mu: chemical potential
    # - sigma([idx_k], omega, flavor, flavor): self-energy
    # - eta: constant frequency shift
    return np.sum(np.linalg.inv(
        (omega[:, None, None] + mu + eta) * np.eye(sigma.shape[-1])
        - hk[:, np.newaxis, :, :]
        - sigma
    ), axis=0)/hk.size


def gloc_from_dos(omega, dos, E, hloc=0, mu=0, sigma=0, eta=0):
    # input:
    # - omega(idx_o): frequency array (real / Matsubara)
    # - dos(idx_E, flavor): density of states
    # - E(idx_E): DOS energy axis
    # - mu: chemical potential
    # - sigma(omega, flavor, flavor): self-energy
    # - eta: constant frequency shift
    return simpson(np.linalg.inv(
        ((omega[:, None, None] + mu + eta) * np.eye(sigma.shape[-1])
        - hloc - E - sigma) / dos[:, None, :, None] * np.eye(dos.shape[-1]),
    ), axis=0, x=E)


def gloc_from_bethe(omega, hloc, D, mu=0, sigma=0, eta=0):
    # input:
    # - omega(idx_o): frequency array (real / Matsubara)
    # - hloc(flavor, flavor): local Hamiltonian
    # - D(flavor): half-bandwidth
    # - mu: chemical potential
    # - sigma(omega, flavor, flavor): self-energy
    # - eta: constant frequency shift
    z = (omega[:, None, None] + mu + eta) * np.eye(sigma.shape[-1]) - hloc - sigma
    D2 = np.diagflat(D**2)
    return 2 * z / D2 * (1 - np.sqrt(1 - D2/z**2 + 0.0j))
