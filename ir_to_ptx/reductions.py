# Accumulators, and the reduction across threads: warp shuffle, then a tree in shared memory.

from ir_to_ptx.registers import new_temp, new_label
from ir_to_ptx.elementwise import OP_TABLE, assign_line, float_hex

# Start value of an accumulator: the value that does not change the result.
ACCUM_IDENTITY = {
    "add": "0f00000000",
    "maximum": "0fF149F2CA",
}

# Accumulators set by an Init statement. Their Accum lives in an inner loop, but the value must survive
# the outer loop too (matmul: acc runs over all K tiles), so emit_accum_init must not reset them.
init_names = set()

def emit_init(lines, stmt):
    # acc = identity, once, at the top of the Tile.
    lines.append(f"    .reg .f32 %{stmt.name};")
    lines.append(f"    mov.f32 %{stmt.name}, {float_hex(stmt.identity)};")
    init_names.add(stmt.name)

def emit_accum_init(lines, loop_body):
    # An Accum register lives across loop iterations, so it is declared and set before the loop.
    for stmt in loop_body:
        if type(stmt).__name__ == "Accum" and stmt.name not in init_names:
            op_name = stmt.op.name
            if op_name not in ACCUM_IDENTITY:
                raise NotImplementedError(f"no identity for accum op {op_name}")
            lines.append(f"    .reg .f32 %{stmt.name};")
            lines.append(f"    mov.f32 %{stmt.name}, {ACCUM_IDENTITY[op_name]};")

def emit_accum(lines, stmt):
    # acc <- op(acc, value): the same PTX line as an Assign, the result goes back into acc.
    # With a base (online softmax): acc <- op(base, value). The base already holds the rescaled acc.
    first = stmt.name
    if stmt.base is not None:
        first = stmt.base
    template = OP_TABLE[stmt.op.name]
    line = template.format(d=f"%{stmt.name}", a=f"%{first}", b=f"%{stmt.value}")
    lines.append(f"    {line}")

def emit_combine(lines, stmt):
    # The combine of a shuffle or tree step. A result that is not a state is a temp of this step,
    # so it is declared here, inside the step's { } scope. A state result writes the outer register.
    for combine in stmt.combine_states:
        if combine.name not in stmt.state:
            lines.append(f"    .reg .f32 %{combine.name};")
        lines.append(f"    {assign_line(combine)}")

def emit_warp_shuffle(lines, stmt):
    # Butterfly reduction inside a warp. Each step adds the value of the lane at (lane XOR offset).
    # After log2(length) steps every lane holds the full result.
    offset = stmt.length // 2
    while offset > 0:
        # A { } block gives the partner register its own scope, so each step can declare it again.
        lines.append("    {")
        for i, state in enumerate(stmt.state):
            partner = stmt.state_b[i]
            lines.append(f"    .reg .f32 %{partner};")
            lines.append(f"    shfl.sync.bfly.b32 %{partner}, %{state}, {offset}, 0x1f, 0xffffffff;")
        emit_combine(lines, stmt)
        lines.append("    }")
        offset = offset // 2

def emit_tree_halve(lines, stmt):
    # Tree reduction in smem: each step, the lower half adds the upper half (s = length/2, ..., 1).
    # At the end every thread reads the result from smem[0].
    if stmt.inner is not None or stmt.barrier_count is not None:
        raise NotImplementedError("TreeHalve with inner or barrier_count")

    tid = f"%{stmt.tid_var}"
    nbytes = stmt.dtype.nbytes

    # The address of smem[tid] is the same in every step, so compute it once for each buffer.
    addresses = []
    for buf in stmt.bufs:
        byte_offset = new_temp(lines, "s64")
        lines.append(f"    mul.wide.s32 {byte_offset}, {tid}, {nbytes};")
        address = new_temp(lines, "u64")
        lines.append(f"    add.s64 {address}, %{buf}, {byte_offset};")
        addresses.append(address)

    s = stmt.length // 2
    while s > 0:
        skip = new_temp(lines, "pred")
        skip_label = new_label("HALVE")
        lines.append(f"    setp.ge.s32 {skip}, {tid}, {s};")
        lines.append(f"    @{skip} bra {skip_label};")

        lines.append("    {")
        for i, state in enumerate(stmt.state):
            partner = stmt.state_b[i]
            lines.append(f"    .reg .f32 %{partner};")
            lines.append(f"    ld.shared.f32 %{state}, [{addresses[i]}];")
            lines.append(f"    ld.shared.f32 %{partner}, [{addresses[i]}+{s * nbytes}];")
        emit_combine(lines, stmt)
        for i, state in enumerate(stmt.state):
            lines.append(f"    st.shared.f32 [{addresses[i]}], %{state};")
        lines.append("    }")

        lines.append(f"{skip_label}:")
        lines.append(f"    bar.sync {stmt.barrier_id};")
        s = s // 2

    for i, state in enumerate(stmt.state):
        lines.append(f"    ld.shared.f32 %{state}, [%{stmt.bufs[i]}];")
