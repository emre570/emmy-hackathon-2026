# StridedLoop and Loop: unrolled main loop with induction pointers, then a tail loop.

from ir_to_ptx.registers import new_temp, new_label
from ir_to_ptx.indexing import emit_expr, index_stride
from ir_to_ptx.reductions import emit_accum_init
from ir_to_ptx import memory

# statements imports this file too, so it is imported as a module and used at run time.
from ir_to_ptx import statements

# Copies of the loop body in one pass of the main loop. The copies have no branch between them,
# so ptxas can move all their loads up and keep them in flight together. nvcc unrolls r02 by 16 too.
STRIDED_UNROLL = 16

# A Loop up to this extent gets one copy per iteration (a02: 32). a01 (64, 4 loops in one body) stays at 16,
# because 4 x 64 copies make the code big again (see the a01 code size fix in emit_counted_loop).
FULL_UNROLL_MAX = 32

def emit_loop_body_copy(lines, axis_name, step, body, shapes):
    # One iteration: the body, then a += step. The { } scope lets each copy declare its own %in0.
    lines.append("    {")
    statements.emit_statements(lines, body, shapes)
    lines.append(f"    add.s32 %{axis_name}, %{axis_name}, {step};")
    lines.append("    }")

def loop_accesses(body, shapes, axis_name):
    # Every Load and Write of a straight-line body: (buffer, index, shape, nbytes, stride).
    # None when the body is not straight-line, or an index does not grow by a fixed amount with the axis.
    accesses = []
    for stmt in body:
        kind = type(stmt).__name__
        if kind == "Load":
            buffer = stmt.input
            nbytes = memory.load_nbytes(stmt)
        elif kind == "Write":
            buffer = stmt.output
            nbytes = stmt.value_dtype.nbytes
        elif kind == "Assign" or kind == "Accum" or kind == "Let":
            continue
        else:
            return None

        if len(stmt.index) == 0:
            continue
        stride = index_stride(stmt.index, shapes[buffer], axis_name)
        if stride is None:
            return None
        accesses.append((buffer, stmt.index, shapes[buffer], nbytes, stride))
    return accesses

def emit_loop_pointers(lines, accesses):
    # One pointer per access, set to the address at the current axis value. Made before the loop,
    # outside every copy scope, so all copies can use it.
    for buffer, index, shape, nbytes, stride in accesses:
        key = (buffer, repr(index))
        if key in memory.loop_pointers:
            continue
        address = memory.emit_address(lines, buffer, index, shape, nbytes)
        pointer = new_temp(lines, "u64")
        lines.append(f"    mov.u64 {pointer}, {address};")
        memory.loop_pointers[key] = (pointer, stride * nbytes)

def emit_shifted_copy(lines, body, shapes, shift):
    # One iteration at axis + shift. The axis register and the pointers do not change.
    memory.copy_shift = shift
    lines.append("    {")
    statements.emit_statements(lines, body, shapes)
    lines.append("    }")
    memory.copy_shift = 0

def emit_strided_loop(lines, stmt, shapes):
    # for (a = start; a < extent; a += step): start is the thread's first element, step the thread count.
    if stmt.end is not None:
        raise NotImplementedError("StridedLoop with end")

    emit_accum_init(lines, stmt.body)
    start = emit_expr(lines, stmt.start)
    emit_counted_loop(lines, stmt.axis, start, stmt.step.value, stmt.body, shapes)

def emit_loop(lines, stmt, shapes):
    # for (a = 0; a < extent; a++): every thread runs all iterations. Same PTX as a StridedLoop.
    emit_accum_init(lines, stmt.body)
    if stmt.axis.extent.value <= FULL_UNROLL_MAX and not has_inner_loop(stmt.body):
        accesses = loop_accesses(stmt.body, shapes, stmt.axis.name)
        if accesses is not None:
            emit_full_unroll(lines, stmt.axis, stmt.body, shapes, accesses)
            return
    emit_counted_loop(lines, stmt.axis, "0", 1, stmt.body, shapes)

def emit_full_unroll(lines, axis, body, shapes, accesses):
    # One copy per iteration, no branch. Copy k reads at pointer + k * stride (an immediate offset).
    # a02: the a2 loop (32) runs once per a1 iteration, so the MAIN/TAIL checks cost on every pass.
    axis_name = axis.name
    lines.append("    {")
    lines.append(f"    .reg .s32 %{axis_name};")
    lines.append(f"    mov.s32 %{axis_name}, 0;")
    emit_loop_pointers(lines, accesses)
    for copy in range(axis.extent.value):
        emit_shifted_copy(lines, body, shapes, copy)
    memory.loop_pointers.clear()
    lines.append("    }")

def has_inner_loop(body):
    for stmt in body:
        kind = type(stmt).__name__
        if kind == "Loop" or kind == "StridedLoop":
            return True
    return False

def emit_counted_loop(lines, axis, start, step, body, shapes):
    # for (a = start; a < extent; a += step) { body }
    # Main loop: STRIDED_UNROLL iterations per pass, while the last one of them is still < extent.
    # Tail loop: the leftover iterations, one per pass.
    axis_name = axis.name
    extent = axis.extent.value

    # The axis register and the pointers live in the loop's own { } scope. a01 has 4 loops over a2
    # in one body, and each of them declares %a2. The Accum registers stay outside, set by the caller.
    lines.append("    {")
    lines.append(f"    .reg .s32 %{axis_name};")
    lines.append(f"    mov.s32 %{axis_name}, {start};")

    accesses = loop_accesses(body, shapes, axis_name)
    if accesses is not None:
        emit_loop_pointers(lines, accesses)

    main_label = new_label("MAIN")
    tail_label = new_label("TAIL")
    end_label = new_label("END")

    # A loop of loops gets no main loop: its copies only add code (a01: 183 KB of SASS), not loads in flight.
    if not has_inner_loop(body):
        lines.append(f"{main_label}:")
        last = new_temp(lines, "s32")
        lines.append(f"    add.s32 {last}, %{axis_name}, {(STRIDED_UNROLL - 1) * step};")
        main_done = new_temp(lines, "pred")
        lines.append(f"    setp.ge.s32 {main_done}, {last}, {extent};")
        lines.append(f"    @{main_done} bra {tail_label};")
        if accesses is not None:
            # The pointers replace the address math from the axis register. ptxas then has no reason to
            # reload the buffer base from constant memory at the start of every pass (measured: r02 10.8 -> 8.8 us).
            for copy in range(STRIDED_UNROLL):
                emit_shifted_copy(lines, body, shapes, copy * step)
            for key in memory.loop_pointers:
                pointer, byte_stride = memory.loop_pointers[key]
                lines.append(f"    add.s64 {pointer}, {pointer}, {STRIDED_UNROLL * step * byte_stride};")
            lines.append(f"    add.s32 %{axis_name}, %{axis_name}, {STRIDED_UNROLL * step};")
            # The tail loop computes its addresses from the axis register again.
            memory.loop_pointers.clear()
        else:
            for copy in range(STRIDED_UNROLL):
                emit_loop_body_copy(lines, axis_name, step, body, shapes)
        lines.append(f"    bra {main_label};")

    lines.append(f"{tail_label}:")
    tail_done = new_temp(lines, "pred")
    lines.append(f"    setp.ge.s32 {tail_done}, %{axis_name}, {extent};")
    lines.append(f"    @{tail_done} bra {end_label};")
    emit_loop_body_copy(lines, axis_name, step, body, shapes)
    lines.append(f"    bra {tail_label};")
    lines.append(f"{end_label}:")
    lines.append("    }")
