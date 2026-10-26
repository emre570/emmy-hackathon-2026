# Addresses, loads, stores, and shared memory.

from ir_to_ptx.registers import new_temp
from ir_to_ptx.indexing import emit_expr
from ir_to_ptx.cse import load_table, emit_copy
from emmy.compiler.ir.expr import BinaryExpr

# ponytail: global set, fine because one process compiles one kernel
smem_buffers = set()

# Set by compile.py from meta.json. When the launch gives dynamic shared memory (m01: 73728 bytes,
# more than the 48 KB static limit), every Smem is a slice of one extern pool, as in ref.cu.
dynamic_smem_bytes = 0
smem_pool_offset = 0

def state_space(buffer):
    # Smem buffers live in shared memory, all kernel arguments live in global memory.
    if buffer in smem_buffers:
        return "shared"
    return "global"

# Set by loops.emit_strided_loop while it prints the unrolled copies of the main loop.
# ponytail: globals, fine because one process compiles one kernel
loop_pointers = {}   # (buffer, index text) -> (pointer register, bytes per axis step)
copy_shift = 0       # how far this copy is ahead of the pointers, in units of the axis

def emit_address(lines, buffer, index, shape, nbytes):
    # A scalar buffer (index ()) is read at its base address.
    if len(index) == 0:
        return f"%{buffer}"

    # In the main loop: the pointer holds the address of copy 0, copy k adds its distance as an immediate.
    key = (buffer, repr(index))
    if key in loop_pointers:
        pointer, byte_stride = loop_pointers[key]
        byte_shift = copy_shift * byte_stride
        if byte_shift != 0:
            return f"{pointer}+{byte_shift}"
        return pointer

    # Row-major flatten: x[i, j] with shape (M, N) -> i * N + j.
    offset = emit_expr(lines, index[0])
    for axis in range(1, len(index)):
        position = emit_expr(lines, index[axis])
        flat = new_temp(lines, "s32")
        lines.append(f"    mad.lo.s32 {flat}, {offset}, {shape[axis]}, {position};")
        offset = flat

    byte_offset = new_temp(lines, "s64")
    lines.append(f"    mul.wide.s32 {byte_offset}, {offset}, {nbytes};")

    address = new_temp(lines, "u64")
    lines.append(f"    add.s64 {address}, %{buffer}, {byte_offset};")
    return address

def vector_operand(names):
    # One name -> %x. Several names -> {%x, %y}, the operand form of a vector load or store.
    if len(names) == 1:
        return f"%{names[0]}"

    operand = "{"
    for i, name in enumerate(names):
        if (i < len(names) - 1):
            operand += f"%{name}, "
        else:
            operand += f"%{name}"
    operand += "}"
    return operand

def vector_suffix(names):
    # ld.global.f32 for one value, ld.global.v2.f32 / .v4.f32 for a group.
    if len(names) == 1:
        return ""
    return f".v{len(names)}"

def load_nbytes(stmt):
    # A Load from an Smem buffer has no dtype in the IR. emit_smem allows only float, so it is 4 bytes.
    if stmt.dtype is None:
        return 4
    return stmt.dtype.nbytes

def emit_load(lines, stmt, shapes):
    key = (stmt.input, repr(stmt.index), len(stmt.names))
    if key in load_table:
        first_names = load_table[key]
        for i, name in enumerate(stmt.names):
            emit_copy(lines, name, first_names[i])
        return
    load_table[key] = stmt.names

    address = emit_address(lines, stmt.input, stmt.index, shapes[stmt.input], load_nbytes(stmt))

    for name in stmt.names:
        lines.append(f"    .reg .f32 %{name};")

    suffix = vector_suffix(stmt.names)
    operand = vector_operand(stmt.names)
    space = state_space(stmt.input)
    lines.append(f"    ld.{space}{suffix}.f32 {operand}, [{address}];")

def emit_write(lines, stmt, shapes):
    # The memory of this buffer can change, so old loads from it are not valid any more.
    for key in list(load_table.keys()):
        if key[0] == stmt.output:
            del load_table[key]

    address = emit_address(lines, stmt.output, stmt.index, shapes[stmt.output], stmt.value_dtype.nbytes)

    if stmt.atomic:
        # red = atomic add with no return value. PTX has no vector form for f32 here.
        if len(stmt.values) != 1:
            raise NotImplementedError("vector atomic write")
        lines.append(f"    red.global.add.f32 [{address}], %{stmt.values[0]};")
        return

    suffix = vector_suffix(stmt.values)
    operand = vector_operand(stmt.values)
    space = state_space(stmt.output)
    lines.append(f"    st.{space}{suffix}.f32 [{address}], {operand};")

def emit_smem_header(lines):
    # Module scope, before .entry. The pool has no size here: the launch gives it (meta.json smem_bytes).
    lines.append(".extern .shared .align 16 .b8 _smem_pool[];")
    lines.append("")

def emit_smem_slice(lines, name, nbytes, align):
    # The next free bytes of the pool, rounded up to align. m01: _a_smem at 0, _b_smem at 24576.
    global smem_pool_offset
    offset = (smem_pool_offset + align - 1) // align * align
    if offset + nbytes > dynamic_smem_bytes:
        raise ValueError(f"Smem {name} does not fit in {dynamic_smem_bytes} bytes")
    smem_pool_offset = offset + nbytes

    lines.append(f"    .reg .u64 %{name};")
    lines.append(f"    mov.u64 %{name}, _smem_pool;")
    lines.append(f"    add.u64 %{name}, %{name}, {offset};")

def emit_smem(lines, stmt, shapes):
    # A shared array, plus a u64 register with its address, so emit_address works the same as for global.
    if stmt.dtype != "float":
        raise NotImplementedError(f"Smem of type {stmt.dtype}")

    num_elements = 1
    for extent in stmt.extents:
        num_elements *= extent

    # cp.async of 16 bytes needs a 16-byte aligned target. The IR gives align=16 then, else 0.
    align = 4
    if stmt.align > align:
        align = stmt.align

    if dynamic_smem_bytes > 0:
        emit_smem_slice(lines, stmt.name, num_elements * 4, align)
    else:
        lines.append(f"    .shared .align {align} .f32 {stmt.name}[{num_elements}];")
        lines.append(f"    .reg .u64 %{stmt.name};")
        lines.append(f"    mov.u64 %{stmt.name}, {stmt.name};")

    smem_buffers.add(stmt.name)
    shapes[stmt.name] = list(stmt.extents)

def emit_cp_async(lines, stmt, shapes):
    # Copy nbytes from global to shared memory without a wait. The data is there only after
    # cp.async.wait_group and a bar.sync. Both index tuples count f32 elements, so nbytes = 4 for the address.
    if stmt.swizzle != "NONE":
        raise NotImplementedError(f"CpAsyncCopy with swizzle {stmt.swizzle}")

    # m01: _a_smem[0, 0] + lane(row, col) is the element _a_smem[0 + row, 0 + col].
    smem_index = stmt.smem_index
    if stmt.lane_index is not None:
        smem_index = []
        for i, position in enumerate(stmt.smem_index):
            smem_index.append(BinaryExpr(op="+", left=position, right=stmt.lane_index[i]))

    smem_address = emit_address(lines, stmt.smem, smem_index, shapes[stmt.smem], 4)
    src_address = emit_address(lines, stmt.src, stmt.src_index, shapes[stmt.src], 4)

    # .cg (L2 only) accepts only 16 bytes. 4 and 8 bytes need .ca (L1 and L2).
    cache = "ca"
    if stmt.nbytes == 16:
        cache = "cg"
    lines.append(f"    cp.async.{cache}.shared.global [{smem_address}], [{src_address}], {stmt.nbytes};")
