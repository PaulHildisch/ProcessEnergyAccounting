#Util file to find the intersection of features across multiple automatic selection results
#-> We used this to form a generalized feature set by extracting a minimum subset that is present in all workflows with and without stressng 


#ampliseq Train:
aseq = set(['delta_cpu_ns', 'delta_io_bytes', 'delta_net_send_bytes', 'context_switches', 'syscall_count', 'syscall_class_network', 'syscall_class_memory', 'syscall_class_sched'])


#sarek Train:
sarek = set( ['delta_cpu_ns', 'delta_io_bytes', 'delta_net_send_bytes', 'context_switches', 'syscall_count', 'delta_rss_memory', 'delta_cpu_time_proc', 'syscall_class_file', 'syscall_class_network', 'syscall_class_memory', 'syscall_class_process', 'syscall_class_other'])



#mixed unseen type:
mu = set(['delta_cpu_ns', 'delta_io_bytes', 'delta_net_send_bytes', 'context_switches', 'syscall_count', 'delta_rss_memory', 'delta_cpu_time_proc', 'syscall_class_file', 'syscall_class_network', 'syscall_class_memory', 'syscall_class_process', 'syscall_class_other', 'syscall_class_sched'])

#mixed unseen type2:
mu2 = set(['delta_cpu_ns', 'delta_io_bytes', 'delta_net_send_bytes', 'context_switches', 'syscall_count', 'delta_rss_memory', 'delta_cpu_time_proc', 'syscall_class_file', 'syscall_class_network', 'syscall_class_memory', 'syscall_class_process', 'syscall_class_other'])


#stress ng:
stressng = set( ['delta_cpu_ns', 'delta_io_bytes', 'delta_net_send_bytes', 'context_switches', 'syscall_count', 'delta_rss_memory', 'syscall_class_process', 'syscall_class_signal'])


generalizing_features = set.intersection(aseq, mu, sarek, mu2, stressng)
print("Features that all workflows and benchmarks have in common after auto selection with stressng")
print(generalizing_features)

print("Features that all workflows and benchmarks have in common after auto selection without stressng")
print(set.intersection(aseq, mu, sarek, mu2))