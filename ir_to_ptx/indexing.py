# Integer index expressions: emit_expr makes PTX for them, eval_index computes them on numbers.

from ir_to_ptx.registers import new_temp

def emit_expr(lines, expr):
    # Emit the PTX for one index expression and return the s32 register that holds it.
    # Children first, then the parent: the parent uses the register names its children return.
    kind = type(expr).__name__

    if kind == "Var":
        # The value is already in a register (the axis register, or an SSA name).
        return f"%{expr.name}"

    elif kind == "Literal":
        reg = new_temp(lines, "s32")
        lines.append(f"    mov.s32 {reg}, {expr.value};")
        return reg

    elif kind == "BinaryExpr":
        left = emit_expr(lines, expr.left)
        right = emit_expr(lines, expr.right)
        reg = new_temp(lines, "s32")
        if expr.op == "+":
            lines.append(f"    add.s32 {reg}, {left}, {right};")
        elif expr.op == "*":
            lines.append(f"    mul.lo.s32 {reg}, {left}, {right};")
        elif expr.op == "-":
            lines.append(f"    sub.s32 {reg}, {left}, {right};")
        elif expr.op == "/":
            # Index values are never negative, so C division (toward 0) and floor division agree.
            lines.append(f"    div.s32 {reg}, {left}, {right};")
        elif expr.op == "%":
            lines.append(f"    rem.s32 {reg}, {left}, {right};")
        else:
            raise NotImplementedError(f"index op {expr.op}")
        return reg

    elif kind == "TernaryExpr":
        # m01: (_ks + 128) < 512 ? (_ks + 128) : 448. setp makes the predicate, selp picks the value.
        cond = expr.cond
        if type(cond).__name__ != "BinaryExpr" or cond.op != "<":
            raise NotImplementedError(f"ternary condition {cond}")
        cond_left = emit_expr(lines, cond.left)
        cond_right = emit_expr(lines, cond.right)
        pred = new_temp(lines, "pred")
        lines.append(f"    setp.lt.s32 {pred}, {cond_left}, {cond_right};")
        if_true = emit_expr(lines, expr.if_true)
        if_false = emit_expr(lines, expr.if_false)
        reg = new_temp(lines, "s32")
        lines.append(f"    selp.s32 {reg}, {if_true}, {if_false}, {pred};")
        return reg

    else:
        raise NotImplementedError(f"index expression {kind}")

def eval_index(expr, values, default):
    # The integer value of an index expression. A Var not in values gets the default.
    kind = type(expr).__name__
    if kind == "Var":
        if expr.name in values:
            return values[expr.name]
        return default
    elif kind == "Literal":
        return expr.value
    elif kind == "BinaryExpr":
        left = eval_index(expr.left, values, default)
        right = eval_index(expr.right, values, default)
        if expr.op == "+":
            return left + right
        elif expr.op == "*":
            return left * right
        elif expr.op == "-":
            return left - right
        elif expr.op == "/":
            return left // right
        elif expr.op == "%":
            return left % right
        elif expr.op == "<":
            return left < right
    elif kind == "TernaryExpr":
        if eval_index(expr.cond, values, default):
            return eval_index(expr.if_true, values, default)
        return eval_index(expr.if_false, values, default)
    raise NotImplementedError(f"index expression {kind}")

def axis_under_div(expr, axis_name, inside):
    # True when axis_name is used under a / or % or in a ternary. Such an index is not linear in the axis,
    # even when the test points 0, 1, 2 look linear: (a * 4) % 64 is 0, 4, 8, then wraps at a = 16.
    kind = type(expr).__name__
    if kind == "Var":
        return inside and expr.name == axis_name
    elif kind == "TernaryExpr":
        if axis_under_div(expr.cond, axis_name, True):
            return True
        if axis_under_div(expr.if_true, axis_name, True):
            return True
        return axis_under_div(expr.if_false, axis_name, True)
    elif kind == "BinaryExpr":
        below = inside or expr.op == "/" or expr.op == "%"
        if axis_under_div(expr.left, axis_name, below):
            return True
        return axis_under_div(expr.right, axis_name, below)
    return False

def flat_index(index, shape, values, default):
    # The same row-major flatten as emit_address, but on numbers.
    flat = 0
    for axis in range(len(index)):
        flat = flat * shape[axis] + eval_index(index[axis], values, default)
    return flat

def index_stride(index, shape, axis_name):
    # How much the flat index grows when axis_name grows by 1. None when it is not a fixed amount.
    # We test axis = 0, 1, 2 with the other Vars at 0 and again at 1.
    for expr in index:
        if axis_under_div(expr, axis_name, False):
            return None
    stride = None
    for default in (0, 1):
        f0 = flat_index(index, shape, {axis_name: 0}, default)
        f1 = flat_index(index, shape, {axis_name: 1}, default)
        f2 = flat_index(index, shape, {axis_name: 2}, default)
        if f1 - f0 != f2 - f1:
            return None
        if stride is not None and f1 - f0 != stride:
            return None
        stride = f1 - f0
    return stride
