# python compile.py path/to/kernel.ir.json > out.ptx
# Reads one Kernel IR and its meta.json, prints PTX for sm_120. The stages live in ir_to_ptx/.

import os
import sys
import json
from emmy.compiler.graph import Graph

from ir_to_ptx.prologue import emit_header, emit_signature, emit_params, emit_thread_index, emit_guard, emit_tile_axes
from ir_to_ptx.statements import emit_statements
from ir_to_ptx import memory

def find_kernel_op(g):
    for node in g.nodes.values():
        if type(node.op).__name__ == 'KernelOp':
            return node.op

    raise ValueError("no KernelOp in IR")

def find_launch(meta, kernel_name):
    for launch in meta["kernels"]:
        if launch["entry"] == kernel_name:
            return launch

    raise ValueError(f"no launch for {kernel_name} in meta.json")

if __name__ == '__main__':
    ir_path = sys.argv[1]
    ir_file = open(ir_path)
    ir_out = json.load(ir_file)

    g = Graph.from_dict(ir_out)
    kernel_op = find_kernel_op(g)

    tile = kernel_op.body[0]
    num_cells = 1
    for axis in tile.axes:
        num_cells *= axis.extent.value

    meta_path = os.path.join(os.path.dirname(ir_path), "meta.json")
    meta = json.load(open(meta_path))

    launch = find_launch(meta, kernel_op.name)
    arg_order = launch["arg_order"]

    shapes = {}
    for arg in launch["args"]:
        shapes[arg["name"]] = arg["shape"]

    lines = []
    emit_header(lines)
    memory.dynamic_smem_bytes = launch["smem_bytes"]
    if memory.dynamic_smem_bytes > 0:
        memory.emit_smem_header(lines)
    emit_signature(lines, kernel_op.name, arg_order)
    emit_params(lines, arg_order)
    emit_thread_index(lines)
    emit_guard(lines, num_cells)
    emit_tile_axes(lines, tile.axes)
    emit_statements(lines, tile.body, shapes)
    lines.append("")
    lines.append("DONE:")
    lines.append("    ret;")
    lines.append("}")

    for line in lines:
        print(line)
