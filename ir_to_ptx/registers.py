# Fresh register and label names. One counter for both, so every name in a kernel is unique.

# ponytail: global counter, fine because one process compiles one kernel
temp_count = 0

def new_temp(lines, ptx_type):
    global temp_count
    name = f"%t{temp_count}"
    temp_count += 1
    lines.append(f"    .reg .{ptx_type} {name};")
    return name

def new_label(prefix):
    global temp_count
    name = f"{prefix}_{temp_count}"
    temp_count += 1
    return name
