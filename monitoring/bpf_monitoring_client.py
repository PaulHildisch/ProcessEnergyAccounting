import ctypes
import os
import time

from bcc import BPF

from monitoring.syscall_classes import SYSCALL_CLASSES, SYSCALL_NAMES


class BPFMonitoringClient:
    def __init__(self):
        self.b = BPF(src_file="monitoring/tracer.c")
        self.cpu_time = self.b.get_table("cpu_time")
        self.start = self.b.get_table("start")
        self.syscall_count = self.b.get_table("syscall_count")
        self.ctx_switches = self.b.get_table("ctx_switches")
        self.total_cpu_time = self.b.get_table("total")
        # self.page_faults = self.b.get_table("page_faults")
        self.disk_io = self.b.get_table("disk_io")
        self.net_send = self.b.get_table("net_send")

    def get_process_list(self):
        pids = set()
        for table in [
            self.cpu_time,
            self.start,
            self.syscall_count,
            self.ctx_switches,
            self.disk_io,
            self.net_send,
        ]:
            pids.update(k.value for k in table.keys())
        print("PIDs before cleanup: ", len(pids))
        to_remove = set()
        for pid in pids:
            if not os.path.exists(f"/proc/{pid}"):
                to_remove.add(pid)
        pids -= to_remove
        for pid in to_remove:
            key = ctypes.c_int(pid)
            self.safe_remove_pid(self.cpu_time, key, "cpu_time")
            self.safe_remove_pid(self.start, key, "start")
            self.safe_remove_pid(self.syscall_count, key, "syscall_count")
            self.safe_remove_pid(self.ctx_switches, key, "ctx_switches")
            self.safe_remove_pid(self.disk_io, key, "disk_io")
            self.safe_remove_pid(self.net_send, key, "net_send")
        print(f"Found {len(pids)} unique PIDs in BPF tables (after cleanup)")
        print(
            "Items in Syscall Count Table: ",
            len(list(self.b.get_table("syscall_type_count"))),
        )
        syscall_type_count = self.b.get_table("syscall_type_count")
        for k in list(syscall_type_count.keys()):
            key = k.value
            pid = key >> 32
            if pid not in pids:
                if k in syscall_type_count:
                    syscall_type_count.pop(k)
        print(
            "Items in Syscall Count Table after cleanup: ",
            len(list(self.b.get_table("syscall_type_count"))),
        )
        start_time = time.time()
        process_list = []
        total_cpu_time_bft = self.total_cpu_time[ctypes.c_int(0)].value
        total_cpu_time_sum = sum(val.value for val in self.cpu_time.values())
        print("Total CPU time (sum): ", total_cpu_time_sum)
        print("Total CPU time (bpf): ", total_cpu_time_bft)
        syscall_classes = self.classify_syscalls()
        for pid in pids:
            key = ctypes.c_int(pid)
            proc = {
                "pid": pid,
                "cpu_time_ns": self.safe_get_bpf_table(self.cpu_time, key),
                "last_scheduled_in_ns": self.safe_get_bpf_table(self.start, key),
                "syscall_count": self.safe_get_bpf_table(self.syscall_count, key),
                "context_switches": self.safe_get_bpf_table(self.ctx_switches, key),
                "disk_io_bytes": self.safe_get_bpf_table(self.disk_io, key),
                "net_send_bytes": self.safe_get_bpf_table(self.net_send, key),
                "syscall_classes": syscall_classes.get(pid, {}),
                "total": total_cpu_time_bft,
            }
            process_list.append(proc)
        end_time = time.time()
        print(f"Process list generated in {end_time - start_time:.2f} seconds")
        for proc in process_list:
            try:
                with open(f"/proc/{proc['pid']}/comm", "r") as f:
                    proc["name"] = f.read().strip()
            except (FileNotFoundError, ProcessLookupError, OSError):
                proc["name"] = "N/A"
        return process_list

    def classify_syscalls(self):
        syscall_type_count = self.b.get_table("syscall_type_count")
        per_pid_class = {}
        for k, v in syscall_type_count.items():
            key = k.value
            pid = key >> 32
            syscall_nr = key & 0xFFFFFFFF
            name = SYSCALL_NAMES.get(syscall_nr, "unknown")
            cls = SYSCALL_CLASSES.get(name, "other")
            if pid not in per_pid_class:
                per_pid_class[pid] = {}
            per_pid_class[pid][cls] = per_pid_class[pid].get(cls, 0) + v.value
        return per_pid_class

    def safe_remove_pid(self, list, key, list_name):
        if key in list:
            try:
                list.pop(key)
            except KeyError:
                print(f"KeyError: {key} not found in {list_name}")
                pass

    def safe_get_bpf_table(self, table, key, default=0):
        try:
            return table[key].value
        except KeyError:
            return default
