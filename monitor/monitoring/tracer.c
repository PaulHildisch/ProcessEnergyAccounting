#include <linux/sched.h>
#include <linux/nsproxy.h>
#include <linux/blkdev.h> // For struct request
#include <linux/skbuff.h> // For struct sk_buff
#include <linux/netdevice.h>

// Map definitions remain exactly the same
BPF_HASH(syscall_count, u32, u64);
BPF_HASH(syscall_type_count, u64, u64);
BPF_HASH(ctx_switches, u32, u64);
BPF_HASH(page_faults, u32, u64);
BPF_HASH(disk_io, u32, u64);
BPF_HASH(net_send, u32, u64);
BPF_HASH(start, u32, u64);
BPF_HASH(cpu_time, u32, u64);
BPF_ARRAY(total, u64, 1);
BPF_HASH(disk_read_bytes, u32, u64);   // VFS-level read bytes per process
BPF_HASH(disk_write_bytes, u32, u64);  // VFS-level write bytes per process
BPF_HASH(net_recv, u32, u64);          // network receive bytes per process
BPF_HASH(net_send_packets, u32, u64);  // network send packet count
BPF_HASH(net_recv_packets, u32, u64);  // network receive packet count

// 1. Syscall Entry (Raw Tracepoint)
RAW_TRACEPOINT_PROBE(sys_enter) {
    u32 pid = bpf_get_current_pid_tgid() >> 32;
    u64 id = ctx->args[1];
    u64 zero = 0, *val;

    val = syscall_count.lookup_or_init(&pid, &zero);
    (*val)++;

    u64 key = ((u64)pid << 32) | id;
    u64 *typeval = syscall_type_count.lookup_or_init(&key, &zero);
    (*typeval)++;
    return 0;
}

// 2. Disk I/O
// On modern kernels, we can safely use the 'nr_sector' or 'data_len'
// but we must ensure the compiler sees the struct.
RAW_TRACEPOINT_PROBE(block_rq_issue) {
    u32 pid = bpf_get_current_pid_tgid() >> 32;
    u64 zero = 0, *val;

    // Fallback: If the struct remains incomplete in your headers,
    // we increment by 1 (request count) to avoid compilation failure,
    // or try to access the size via a helper.
    val = disk_io.lookup_or_init(&pid, &zero);
    (*val) += 1;
    return 0;
}

// VFS-level read bytes: runs in process context, in-kernel file reads
// (includes page-cache hits, which still consume CPU/bus bandwidth).
int kretprobe__vfs_read(struct pt_regs *ctx) {
    int ret = PT_REGS_RC(ctx);
    if (ret <= 0) return 0;
    u32 pid = bpf_get_current_pid_tgid() >> 32;
    u64 zero = 0, *val;
    val = disk_read_bytes.lookup_or_init(&pid, &zero);
    (*val) += (u64)ret;
    return 0;
}

int kretprobe__vfs_write(struct pt_regs *ctx) {
    int ret = PT_REGS_RC(ctx);
    if (ret <= 0) return 0;
    u32 pid = bpf_get_current_pid_tgid() >> 32;
    u64 zero = 0, *val;
    val = disk_write_bytes.lookup_or_init(&pid, &zero);
    (*val) += (u64)ret;
    return 0;
}

// 3. Network Send
RAW_TRACEPOINT_PROBE(net_dev_queue) {
    u32 pid = bpf_get_current_pid_tgid() >> 32;
    struct sk_buff *skb = (struct sk_buff *)ctx->args[0];
    u32 len = 0;

    // Use the BCC helper to safely read the length
    bpf_probe_read_kernel(&len, sizeof(len), &skb->len);

    u64 zero = 0, *val;
    val = net_send.lookup_or_init(&pid, &zero);
    (*val) += len;
    // Track send packet count alongside bytes
    val = net_send_packets.lookup_or_init(&pid, &zero);
    (*val) += 1;
    return 0;
}

// Network receive bytes: runs in the context of the process calling recv/read
// on a socket. Covers TCP and UDP via the common sock_recvmsg path.
int kretprobe__sock_recvmsg(struct pt_regs *ctx) {
    int ret = PT_REGS_RC(ctx);
    if (ret <= 0) return 0;
    u32 pid = bpf_get_current_pid_tgid() >> 32;
    u64 zero = 0, *val;
    val = net_recv.lookup_or_init(&pid, &zero);
    (*val) += (u64)ret;
    val = net_recv_packets.lookup_or_init(&pid, &zero);
    (*val) += 1;
    return 0;
}

// 4. Scheduler Switch
RAW_TRACEPOINT_PROBE(sched_switch) {
    u64 ts = bpf_ktime_get_ns();
    struct task_struct *prev = (struct task_struct *)ctx->args[1];
    struct task_struct *next = (struct task_struct *)ctx->args[2];

    u32 prev_pid = 0, next_pid = 0;
    bpf_probe_read_kernel(&prev_pid, sizeof(prev_pid), &prev->pid);
    bpf_probe_read_kernel(&next_pid, sizeof(next_pid), &next->pid);

    if (prev_pid != 0) {
        u64 *start_ts = start.lookup(&prev_pid);
        if (start_ts) {
            u64 delta = ts - *start_ts;
            u64 *acc = cpu_time.lookup(&prev_pid);
            if (acc) { *acc += delta; }
            else { cpu_time.update(&prev_pid, &delta); }

            u32 index = 0;
            u64 *t = total.lookup(&index);
            if (t) { *t += delta; }
        }
        start.delete(&prev_pid);
    }
    start.update(&next_pid, &ts);
    return 0;
}

// 5. Process Exit
RAW_TRACEPOINT_PROBE(sched_process_exit) {
    u32 pid = bpf_get_current_pid_tgid() >> 32;
    cpu_time.delete(&pid);
    start.delete(&pid);
    syscall_count.delete(&pid);
    ctx_switches.delete(&pid);
    disk_io.delete(&pid);
    net_send.delete(&pid);
    disk_read_bytes.delete(&pid);
    disk_write_bytes.delete(&pid);
    net_recv.delete(&pid);
    net_send_packets.delete(&pid);
    net_recv_packets.delete(&pid);
    return 0;
}

// 6. Process Fork
RAW_TRACEPOINT_PROBE(sched_process_fork) {
    struct task_struct *child = (struct task_struct *)ctx->args[1];
    u32 child_pid = 0;
    bpf_probe_read_kernel(&child_pid, sizeof(child_pid), &child->pid);

    u64 zero = 0;
    cpu_time.update(&child_pid, &zero);
    syscall_count.update(&child_pid, &zero);
    ctx_switches.update(&child_pid, &zero);
    disk_io.update(&child_pid, &zero);
    net_send.update(&child_pid, &zero);
    disk_read_bytes.update(&child_pid, &zero);
    disk_write_bytes.update(&child_pid, &zero);
    net_recv.update(&child_pid, &zero);
    net_send_packets.update(&child_pid, &zero);
    net_recv_packets.update(&child_pid, &zero);
    return 0;
}