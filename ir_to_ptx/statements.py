# The dispatch: one IR statement -> its emitter. Also decides when the CSE tables are cleared.

from ir_to_ptx.cse import cse_clear
from ir_to_ptx.memory import emit_load, emit_write, emit_smem, emit_cp_async
from ir_to_ptx.elementwise import emit_assign, emit_let
from ir_to_ptx.reductions import emit_accum, emit_init, emit_warp_shuffle, emit_tree_halve
from ir_to_ptx.control_flow import emit_cond, emit_sync, emit_cp_async_commit, emit_cp_async_wait
from ir_to_ptx.loops import emit_strided_loop, emit_loop

# Straight-line statements. Every other kind clears the CSE tables before and after it.
CSE_SAFE_KINDS = ("Load", "Assign", "Write", "Let")

def emit_statements(lines, stmts, shapes):
    # Each call is a new scope ({ } block, loop copy, Cond body), so it starts and ends with empty tables.
    cse_clear()
    for stmt in stmts:
        kind = type(stmt).__name__
        if kind not in CSE_SAFE_KINDS:
            cse_clear()

        if kind == "Load":
            emit_load(lines, stmt, shapes)
        elif kind == "Assign":
            emit_assign(lines, stmt)
        elif kind == "Let":
            emit_let(lines, stmt)
        elif kind == "Write":
            emit_write(lines, stmt, shapes)
        elif kind == "Accum":
            emit_accum(lines, stmt)
        elif kind == "Init":
            emit_init(lines, stmt)
        elif kind == "StridedLoop":
            emit_strided_loop(lines, stmt, shapes)
        elif kind == "Loop":
            emit_loop(lines, stmt, shapes)
        elif kind == "WarpShuffle":
            emit_warp_shuffle(lines, stmt)
        elif kind == "Smem":
            emit_smem(lines, stmt, shapes)
        elif kind == "Cond":
            emit_cond(lines, stmt, shapes)
        elif kind == "Sync":
            emit_sync(lines, stmt)
        elif kind == "CpAsyncCopy":
            emit_cp_async(lines, stmt, shapes)
        elif kind == "CpAsyncCommit":
            emit_cp_async_commit(lines, stmt)
        elif kind == "CpAsyncWait":
            emit_cp_async_wait(lines, stmt)
        elif kind == "TreeHalve":
            emit_tree_halve(lines, stmt)
        else:
            raise NotImplementedError(f"no emitter for {kind}")

        if kind not in CSE_SAFE_KINDS:
            cse_clear()
    cse_clear()
