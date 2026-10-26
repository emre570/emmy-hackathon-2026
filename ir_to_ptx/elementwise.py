# Float ops on registers: one Assign = one PTX line.

import struct

from ir_to_ptx.cse import assign_table, canonical, emit_copy

# One PTX line per elementwise op. {d} = result register, {a} {b} = argument registers.
OP_TABLE = {
    "add":        "add.f32 {d}, {a}, {b};",
    "subtract":   "sub.f32 {d}, {a}, {b};",
    "multiply":   "mul.f32 {d}, {a}, {b};",
    "divide":     "div.rn.f32 {d}, {a}, {b};",
    "reciprocal": "rcp.rn.f32 {d}, {a};",
    "relu":       "max.f32 {d}, {a}, 0f00000000;",
    "sqrt":       "sqrt.rn.f32 {d}, {a};",
    "maximum":    "max.f32 {d}, {a}, {b};",
    "copy":       "mov.f32 {d}, {a};",
    # e^x = 2^(x * log2(e)). PTX has only ex2, so exp is two instructions: the multiply goes into d first.
    "exp":        "mul.f32 {d}, {a}, 0f3FB8AA3B;  ex2.approx.f32 {d}, {d};",
}

def float_hex(value):
    # PTX writes a float immediate as 0f + its 32 bits in hex: 1.0 -> 0f3F800000.
    bits = struct.unpack(">I", struct.pack(">f", value))[0]
    return f"0f{bits:08X}"

def emit_let(lines, stmt):
    # f32 one = 1.0: a register that holds a constant.
    if type(stmt.value).__name__ != "Literal":
        raise NotImplementedError("Let with a non-literal value")
    lines.append(f"    .reg .f32 %{stmt.name};")
    lines.append(f"    mov.f32 %{stmt.name}, {float_hex(stmt.value.value)};")

def assign_line(stmt):
    # The PTX line for one Assign, with no register declaration.
    op_name = stmt.op.name
    if op_name not in OP_TABLE:
        raise NotImplementedError(f"no PTX for op {op_name}")

    registers = []
    for arg in stmt.args:
        registers.append(f"%{arg}")

    template = OP_TABLE[op_name]
    if len(registers) == 1:
        return template.format(d=f"%{stmt.name}", a=registers[0])
    else:
        return template.format(d=f"%{stmt.name}", a=registers[0], b=registers[1])

def emit_assign(lines, stmt):
    # The key uses canonical args: reciprocal(in0__u1) matches reciprocal(in0__u0) when in0__u1 is a copy.
    canonical_args = []
    for arg in stmt.args:
        canonical_args.append(canonical(arg))
    key = (stmt.op.name, tuple(canonical_args))
    if key in assign_table:
        emit_copy(lines, stmt.name, assign_table[key])
        return
    assign_table[key] = stmt.name

    lines.append(f"    .reg .f32 %{stmt.name};")
    lines.append(f"    {assign_line(stmt)}")
