# Everything before the first statement of the Tile: header, signature, parameters, thread index.

from ir_to_ptx.registers import new_temp

def emit_header(lines):
    # The runner's ptxas (CUDA 12.9) accepts PTX ISA 8.8 at most.
    lines.append(".version 8.8")
    lines.append(".target sm_120")
    lines.append(".address_size 64")
    lines.append("")

def emit_signature(lines, kernel_name, arg_order):
    lines.append(f".visible .entry {kernel_name}(")
    for i, arg in enumerate(arg_order):
        if (i < len(arg_order) - 1):
            lines.append(f"    .param .u64 param_{arg},")
        else:
            lines.append(f"    .param .u64 param_{arg}")
    lines.append(")")
    lines.append("{")

def emit_params(lines, arg_order):
    for arg in arg_order:
        lines.append(f"    .reg .u64 %{arg};")
        lines.append(f"    ld.param.u64 %{arg}, [param_{arg}];")
        lines.append(f"    cvta.to.global.u64 %{arg}, %{arg};")
    lines.append("")

def emit_thread_index(lines):
    lines.append("    .reg .u32 %r_tid;")
    lines.append("    .reg .u32 %r_ntid;")
    lines.append("    .reg .u32 %r_ctaid;")
    lines.append("    .reg .s32 %gid;")
    lines.append("    mov.u32 %r_tid, %tid.x;")
    lines.append("    mov.u32 %r_ntid, %ntid.x;")
    lines.append("    mov.u32 %r_ctaid, %ctaid.x;")
    lines.append("    mad.lo.s32 %gid, %r_ctaid, %r_ntid, %r_tid;")

    # lane and warp are the names the IR uses in Cond and TreeHalve.
    lines.append("    .reg .s32 %lane;")
    lines.append("    .reg .s32 %warp;")
    lines.append("    and.b32 %lane, %r_tid, 31;")
    lines.append("    shr.u32 %warp, %r_tid, 5;")
    lines.append("")

def emit_guard(lines, num_cells):
    # The launch can have more threads than the Tile has cells. Extra threads skip to DONE.
    lines.append("    .reg .pred %p_guard;")
    lines.append(f"    setp.ge.s32 %p_guard, %gid, {num_cells};")
    lines.append("    @%p_guard bra DONE;")
    lines.append("")

def emit_tile_axes(lines, axes):
    # Split the flat thread id into one register per Tile axis. The last axis changes fastest (row-major).
    # Tile[a0, a1_co] with a1_co = 128: a1_co = gid % 128, a0 = gid / 128.
    rest = "%gid"
    for i in range(len(axes) - 1, -1, -1):
        axis = axes[i]
        lines.append(f"    .reg .s32 %{axis.name};")
        if i == 0:
            lines.append(f"    mov.s32 %{axis.name}, {rest};")
        else:
            lines.append(f"    rem.s32 %{axis.name}, {rest}, {axis.extent.value};")
            quotient = new_temp(lines, "s32")
            lines.append(f"    div.s32 {quotient}, {rest}, {axis.extent.value};")
            rest = quotient
    lines.append("")
