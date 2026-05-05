BPF_HASH(syscall_count, u32, u64);
BPF_HASH(syscall_type_count, u64, u64);
BPF_HASH(ctx_switches, u32, u64);
BPF_HASH(page_faults, u32, u64);
BPF_HASH(disk_io, u32, u64);
BPF_HASH(net_send, u32, u64);
BPF_HASH(start, u32, u64);    // pid -> timestamp
BPF_HASH(cpu_time, u32, u64); // pid -> accumulated CPU time
BPF_ARRAY(total, u64, 1);     // total CPU time

// #include <bpf/bpf_helpers.h>
// #include <linux/types.h>
//
//// Use LRU maps for per-process state
// BPF_LRU_HASH(syscall_count, u32, u64);
// BPF_LRU_HASH(ctx_switches, u32, u64);
// BPF_LRU_HASH(page_faults, u32, u64);
// BPF_LRU_HASH(disk_io, u32, u64);
// BPF_LRU_HASH(net_send, u32, u64);
// BPF_LRU_HASH(cpu_time, u32, u64);
// BPF_LRU_HASH(last_seen, u32, u64);
//
//// syscall_type_count can remain a normal hash
// BPF_HASH(syscall_type_count, u64, u64);

TRACEPOINT_PROBE(raw_syscalls, sys_enter) {
  u32 pid = bpf_get_current_pid_tgid() >> 32;
  u64 zero = 0, *val;
  val = syscall_count.lookup_or_init(&pid, &zero);
  (*val)++;
  // Store syscall type and ID
  u64 key = ((u64)pid << 32) | args->id;
  u64 *typeval = syscall_type_count.lookup_or_init(&key, &zero);
  (*typeval)++;
  return 0;
}
// Node kernel does not support this tracepoint
// TRACEPOINT_PROBE(mm, do_page_fault) {
//     u32 pid = bpf_get_current_pid_tgid() >> 32;
//     u64 zero = 0, *val;
//     val = page_faults.lookup_or_init(&pid, &zero);
//     (*val)++;
//     return 0;
// }
TRACEPOINT_PROBE(block, block_rq_issue) {
  u32 pid = bpf_get_current_pid_tgid() >> 32;
  u64 bytes = args->bytes;
  u64 zero = 0, *val;
  val = disk_io.lookup_or_init(&pid, &zero);
  (*val) += bytes;
  return 0;
}
TRACEPOINT_PROBE(net, net_dev_queue) {
  u32 pid = bpf_get_current_pid_tgid() >> 32;
  u64 len = args->len;
  u64 zero = 0, *val;
  val = net_send.lookup_or_init(&pid, &zero);
  (*val) += len;
  return 0;
}

TRACEPOINT_PROBE(sched, sched_switch) {
  u64 ts = bpf_ktime_get_ns();
  u64 zero = 0;
  u64 *ctx_val;

  u32 prev_pid = args->prev_pid;
  if (prev_pid != 0) {
    ctx_val = ctx_switches.lookup_or_init(&prev_pid, &zero);
    if (ctx_val) {
      (*ctx_val)++;
    }

    u64 *start_ts = start.lookup(&prev_pid);
    if (start_ts) {
      u64 delta = ts - *start_ts;
      u64 *acc = cpu_time.lookup(&prev_pid);
      if (acc) {
        *acc += delta;
      } else {
        cpu_time.update(&prev_pid, &delta);
      }

      u32 index = 0;
      u64 *t = total.lookup(&index);
      if (t) {
        *t += delta;
      }
    }
    start.delete(&prev_pid);
  }

  // Start tracking the new process
  u32 next_pid = args->next_pid;
  if (next_pid != 0) {
    ctx_val = ctx_switches.lookup_or_init(&next_pid, &zero);
    if (ctx_val) {
      (*ctx_val)++;
    }
  }
  start.update(&next_pid, &ts);
  return 0;
}

// Remove all per-process entries on process exit
TRACEPOINT_PROBE(sched, sched_process_exit) {
  u32 pid = bpf_get_current_pid_tgid() >> 32;
  cpu_time.delete(&pid);
  start.delete(&pid);
  syscall_count.delete(&pid);
  ctx_switches.delete(&pid);
  disk_io.delete(&pid);
  net_send.delete(&pid);

  return 0;
}

// Fork handler: initialize per-process state for the child
TRACEPOINT_PROBE(sched, sched_process_fork) {
  u32 parent_pid = args->parent_pid;
  u32 child_pid = args->child_pid;
  u64 zero = 0;

  // Optionally initialize per-process maps for the child
  cpu_time.update(&child_pid, &zero);
  syscall_count.update(&child_pid, &zero);
  ctx_switches.update(&child_pid, &zero);
  disk_io.update(&child_pid, &zero);
  net_send.update(&child_pid, &zero);

  return 0;
}
