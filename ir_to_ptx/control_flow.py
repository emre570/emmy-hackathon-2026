# Branches (Cond) and barriers (Sync).

from ir_to_ptx.registers import new_temp, new_label
from ir_to_ptx.indexing import emit_expr

# statements imports this file too, so it is imported as a module and used at run time.
from ir_to_ptx import statements

# The condition to SKIP the body: the opposite of the IR condition.
SKIP_COMPARE = {
    "==": "ne",
}

def emit_cond(lines, stmt, shapes):
    # if (left op right) { body } else { else_body }
    cond = stmt.cond
    if cond.op not in SKIP_COMPARE:
        raise NotImplementedError(f"Cond with op {cond.op}")

    left = emit_expr(lines, cond.left)
    right = emit_expr(lines, cond.right)
    skip = new_temp(lines, "pred")
    else_label = new_label("ELSE")
    end_label = new_label("ENDIF")

    lines.append(f"    setp.{SKIP_COMPARE[cond.op]}.s32 {skip}, {left}, {right};")
    lines.append(f"    @{skip} bra {else_label};")
    statements.emit_statements(lines, stmt.body, shapes)
    lines.append(f"    bra {end_label};")
    lines.append(f"{else_label}:")
    statements.emit_statements(lines, stmt.else_body, shapes)
    lines.append(f"{end_label}:")

def emit_sync(lines, stmt):
    # __syncthreads(): every thread of the block waits here.
    if stmt.warp or stmt.count is not None:
        raise NotImplementedError("Sync with warp or count")
    lines.append(f"    bar.sync {stmt.barrier_id};")

def emit_cp_async_commit(lines, stmt):
    # Close the group of cp.async copies issued since the last commit.
    lines.append("    cp.async.commit_group;")

def emit_cp_async_wait(lines, stmt):
    # Wait until at most stmt.group groups are still in flight. The older groups are then in shared memory.
    lines.append(f"    cp.async.wait_group {stmt.group};")
