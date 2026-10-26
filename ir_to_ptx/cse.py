# CSE (common subexpression elimination): a repeat of the same load or the same Assign reuses the
# first register. nvcc's frontend does this for Emmy's CUDA path, ptxas does not do it for our PTX.
# memory.emit_load fills load_table, elementwise.emit_assign fills assign_table,
# statements.emit_statements decides when the tables are cleared.

# ponytail: global tables, fine because one process compiles one kernel
load_table = {}      # (buffer, index text, number of names) -> the register names of the first load
assign_table = {}    # (op, canonical args) -> the register name of the first result
alias = {}           # register name -> the register it is a copy of

def cse_clear():
    # Forget everything. Called at every point where an old register is not safe to reuse:
    # a new scope, a branch, a barrier, or a register that changes its value (Accum, shuffle).
    load_table.clear()
    assign_table.clear()
    alias.clear()

def canonical(name):
    if name in alias:
        return alias[name]
    return name

def emit_copy(lines, name, source):
    # The IR still uses the new name, so it gets a register. ptxas removes the mov.
    alias[name] = canonical(source)
    lines.append(f"    .reg .f32 %{name};")
    lines.append(f"    mov.f32 %{name}, %{source};")
