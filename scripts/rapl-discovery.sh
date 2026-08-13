#!/bin/bash

declare -A SYSTEM_INFO=(
    [hostname]="\${HOSTNAME_PLACEHOLDER}"
    [os_name]="\${OS_NAME_PLACEHOLDER}"
    [os_version]="\${OS_VERSION_PLACEHOLDER}"
    [kernel_version]="\${KERNEL_VERSION_PLACEHOLDER}"
    [architecture]="\${ARCHITECTURE_PLACEHOLDER}"
    [cpu_model]="\${CPU_MODEL_PLACEHOLDER}"
    [cpu_vendor]="\${CPU_VENDOR_PLACEHOLDER}"
    [cpu_family]="\${CPU_FAMILY_PLACEHOLDER}"
    [cpu_model_number]="\${CPU_MODEL_NUMBER_PLACEHOLDER}"
    [cpu_stepping]="\${CPU_STEPPING_PLACEHOLDER}"
    [cpu_microcode]="\${CPU_MICROCODE_PLACEHOLDER}"
    [cpu_cores]="\${CPU_CORES_PLACEHOLDER}"
    [cpu_threads]="\${CPU_THREADS_PLACEHOLDER}"
    [cpu_sockets]="\${CPU_SOCKETS_PLACEHOLDER}"
    [cpus_onboard]="\${CPUS_ONBOARD_PLACEHOLDER}"
    [cpu_bogo_mips]="\${CPU_BOGO_MIPS_PLACEHOLDER}"
    [cpu_age_estimate]="\${CPU_AGE_ESTIMATE_PLACEHOLDER}"
    [memory_total]="\${MEMORY_TOTAL_PLACEHOLDER}"
    [memory_type]="\${MEMORY_TYPE_PLACEHOLDER}"
    [memory_speed]="\${MEMORY_SPEED_PLACEHOLDER}"
    [memory_slots]="\${MEMORY_SLOTS_PLACEHOLDER}"
    [disk_total]="\${DISK_TOTAL_PLACEHOLDER}"
    [disk_model]="\${DISK_MODEL_PLACEHOLDER}"
    [disk_type]="\${DISK_TYPE_PLACEHOLDER}"
    [disk_rotational]="\${DISK_ROTATIONAL_PLACEHOLDER}"
    [disk_age_estimate]="\${DISK_AGE_ESTIMATE_PLACEHOLDER}"
    [ip_address]="\${IP_ADDRESS_PLACEHOLDER}"
    [uptime]="\${UPTIME_PLACEHOLDER}"
)

declare -A POWERCAP_INFO=(
    [control_type]="\${POWERCAP_CONTROL_TYPE_PLACEHOLDER}"
    [enabled]="\${POWERCAP_ENABLED_PLACEHOLDER}"
    [zones]="\${POWERCAP_ZONES_PLACEHOLDER}"
)

declare -A RAPL_INFO=(
    [control_type]="${RAPL_CONTROL_TYPE_PLACEHOLDER}"
    [enabled]="${RAPL_ENABLED_PLACEHOLDER}"
    [zones]="${RAPL_ZONES_PLACEHOLDER}"
    [available_domains]="${RAPL_AVAILABLE_DOMAINS_PLACEHOLDER}"
    [modprobe_status]="${RAPL_MODPROBE_STATUS_PLACEHOLDER}"
)

format_microwatts() {
    local _value="$1"
    if [[ "${_value}" =~ ^[0-9]+$ ]]; then
        awk -v v="${_value}" 'BEGIN {printf "%.2f W", v/1000000}'
    else
        printf '%s' "unknown"
    fi
}

format_microjoules() {
    local _value="$1"
    if [[ "${_value}" =~ ^[0-9]+$ ]]; then
        awk -v v="${_value}" 'BEGIN {printf "%.2f J", v/1000000}'
    else
        printf '%s' "unknown"
    fi
}

read_first_line_or_unknown() {
    local _path="$1"
    if [[ -r "${_path}" ]]; then
        head -n 1 "${_path}" 2>/dev/null
    else
        printf '%s' "unknown"
    fi
}

fill_system_info() {
    local _hostname _os_name _os_version _kernel_version _architecture
    local _cpu_model _cpu_vendor _cpu_family _cpu_model_number _cpu_stepping _cpu_microcode
    local _cpu_cores _cpu_threads _memory_total _memory_type _memory_speed _memory_slots
    local _disk_total _disk_model _disk_type _disk_rotational _ip_address _uptime
    local _cpu_sockets _cpus_onboard _cpu_bogo_mips _cpu_age_estimate _disk_age_estimate

    _hostname="$(hostname 2>/dev/null || echo "unknown")"

    if [[ -r /etc/os-release ]]; then
        _os_name="$(. /etc/os-release && echo "${NAME:-unknown}")"
        _os_version="$(. /etc/os-release && echo "${VERSION_ID:-${VERSION:-unknown}}")"
    else
        _os_name="$(uname -s 2>/dev/null || echo "unknown")"
        _os_version="$(uname -r 2>/dev/null || echo "unknown")"
    fi

    _kernel_version="$(uname -r 2>/dev/null || echo "unknown")"
    _architecture="$(uname -m 2>/dev/null || echo "unknown")"

    _cpu_model="$(awk -F': ' '/model name/ {print $2; exit}' /proc/cpuinfo 2>/dev/null)"
    [[ -n "${_cpu_model}" ]] || _cpu_model="$(awk -F': ' '/Hardware/ {print $2; exit}' /proc/cpuinfo 2>/dev/null)"
    [[ -n "${_cpu_model}" ]] || _cpu_model="unknown"

    _cpu_vendor="$(awk -F': ' '/vendor_id/ {print $2; exit}' /proc/cpuinfo 2>/dev/null)"
    [[ -n "${_cpu_vendor}" ]] || _cpu_vendor="unknown"

    _cpu_family="$(awk -F': ' '/cpu family/ {print $2; exit}' /proc/cpuinfo 2>/dev/null)"
    [[ -n "${_cpu_family}" ]] || _cpu_family="unknown"

    _cpu_model_number="$(awk -F': ' '/model[[:space:]]*: / {print $2; exit}' /proc/cpuinfo 2>/dev/null)"
    [[ -n "${_cpu_model_number}" ]] || _cpu_model_number="unknown"

    _cpu_stepping="$(awk -F': ' '/stepping/ {print $2; exit}' /proc/cpuinfo 2>/dev/null)"
    [[ -n "${_cpu_stepping}" ]] || _cpu_stepping="unknown"

    _cpu_microcode="$(awk -F': ' '/microcode/ {print $2; exit}' /proc/cpuinfo 2>/dev/null)"
    [[ -n "${_cpu_microcode}" ]] || _cpu_microcode="unknown"

    _cpu_bogo_mips="$(awk -F': ' '/bogomips/ {print $2; exit}' /proc/cpuinfo 2>/dev/null)"
    [[ -n "${_cpu_bogo_mips}" ]] || _cpu_bogo_mips="unknown"

    _cpu_cores="$(getconf _NPROCESSORS_ONLN 2>/dev/null || nproc 2>/dev/null || echo "0")"
    _cpu_threads="$(awk '/^processor[[:space:]]*:/{n++} END {print (n>0 ? n : 0)}' /proc/cpuinfo 2>/dev/null)"
    [[ -n "${_cpu_threads}" ]] || _cpu_threads="${_cpu_cores}"

    _memory_total="$(awk '/MemTotal/ {printf "%.2f GiB", $2/1024/1024; exit}' /proc/meminfo 2>/dev/null)"
    [[ -n "${_memory_total}" ]] || _memory_total="unknown"

    _memory_type="$(which dmidecode >/dev/null 2>&1 && dmidecode -t memory 2>/dev/null | awk -F: '/Memory Type:/ {gsub(/^[[:space:]]+/, "", $2); print $2; exit}')"
    [[ -n "${_memory_type}" ]] || _memory_type="unknown"

    _memory_speed="$(awk -F': ' '/MemSpeed|Speed/ {print $2; exit}' <(lsblk -dn -o NAME 2>/dev/null | head -n 1 >/dev/null; echo "") 2>/dev/null)"
    [[ -n "${_memory_speed}" ]] || _memory_speed="unknown"

    _memory_slots="$(grep -c '^processor' /proc/cpuinfo 2>/dev/null)"
    [[ -n "${_memory_slots}" ]] || _memory_slots="unknown"

    _disk_total="$(df -h / 2>/dev/null | awk 'NR==2 {print $2}')"
    [[ -n "${_disk_total}" ]] || _disk_total="unknown"

    _disk_model="$(lsblk -dn -o MODEL 2>/dev/null | awk 'NF {print; exit}')"
    [[ -n "${_disk_model}" ]] || _disk_model="unknown"

    _disk_type="$(lsblk -dn -o TYPE 2>/dev/null | awk 'NF {print; exit}')"
    [[ -n "${_disk_type}" ]] || _disk_type="unknown"

    _disk_rotational="$(lsblk -dn -o ROTA 2>/dev/null | awk 'NF {print; exit}')"
    [[ -n "${_disk_rotational}" ]] || _disk_rotational="unknown"

    _disk_age_estimate="$(lsblk -dn -o MODEL,TYPE 2>/dev/null | awk 'NF {print "unknown"; exit}')"
    [[ -n "${_disk_age_estimate}" ]] || _disk_age_estimate="unknown"

    _ip_address="$(hostname -I 2>/dev/null | awk '{print $1}')"
    [[ -n "${_ip_address}" ]] || _ip_address="$(ip route get 1 2>/dev/null | awk '/src/ {for (i=1; i<=NF; i++) if ($i=="src") {print $(i+1); exit}}')"
    [[ -n "${_ip_address}" ]] || _ip_address="unknown"

    _uptime="$(uptime -p 2>/dev/null)"
    [[ -n "${_uptime}" ]] || _uptime="$(awk '{print int($1)" seconds"}' /proc/uptime 2>/dev/null)"
    [[ -n "${_uptime}" ]] || _uptime="unknown"

    _cpu_sockets="$(awk -F': ' '
        /physical id/ { ids[$2]=1; found=1 }
        END {
            if (found) {
                print length(ids)
            } else {
                print 1
            }
        }' /proc/cpuinfo 2>/dev/null)"
    [[ -n "${_cpu_sockets}" && "${_cpu_sockets}" -gt 0 ]] 2>/dev/null || _cpu_sockets="1"

    _cpus_onboard="${_cpu_cores}"
    _cpu_age_estimate="unknown"
    _disk_age_estimate="unknown"

    SYSTEM_INFO[hostname]="${_hostname}"
    SYSTEM_INFO[os_name]="${_os_name}"
    SYSTEM_INFO[os_version]="${_os_version}"
    SYSTEM_INFO[kernel_version]="${_kernel_version}"
    SYSTEM_INFO[architecture]="${_architecture}"
    SYSTEM_INFO[cpu_model]="${_cpu_model}"
    SYSTEM_INFO[cpu_vendor]="${_cpu_vendor}"
    SYSTEM_INFO[cpu_family]="${_cpu_family}"
    SYSTEM_INFO[cpu_model_number]="${_cpu_model_number}"
    SYSTEM_INFO[cpu_stepping]="${_cpu_stepping}"
    SYSTEM_INFO[cpu_microcode]="${_cpu_microcode}"
    SYSTEM_INFO[cpu_cores]="${_cpu_cores}"
    SYSTEM_INFO[cpu_threads]="${_cpu_threads}"
    SYSTEM_INFO[cpu_sockets]="${_cpu_sockets}"
    SYSTEM_INFO[cpus_onboard]="${_cpus_onboard}"
    SYSTEM_INFO[cpu_bogo_mips]="${_cpu_bogo_mips}"
    SYSTEM_INFO[cpu_age_estimate]="${_cpu_age_estimate}"
    SYSTEM_INFO[memory_total]="${_memory_total}"
    SYSTEM_INFO[memory_type]="${_memory_type}"
    SYSTEM_INFO[memory_speed]="${_memory_speed}"
    SYSTEM_INFO[memory_slots]="${_memory_slots}"
    SYSTEM_INFO[disk_total]="${_disk_total}"
    SYSTEM_INFO[disk_model]="${_disk_model}"
    SYSTEM_INFO[disk_type]="${_disk_type}"
    SYSTEM_INFO[disk_rotational]="${_disk_rotational}"
    SYSTEM_INFO[disk_age_estimate]="${_disk_age_estimate}"
    SYSTEM_INFO[ip_address]="${_ip_address}"
    SYSTEM_INFO[uptime]="${_uptime}"

    printf '\n'
    printf '========================================\n'
    printf '           System Information           \n'
    printf '========================================\n'
    printf '%-16s : %s\n' "Hostname" "${SYSTEM_INFO[hostname]}"
    printf '%-16s : %s\n' "OS Name" "${SYSTEM_INFO[os_name]}"
    printf '%-16s : %s\n' "OS Version" "${SYSTEM_INFO[os_version]}"
    printf '%-16s : %s\n' "Kernel Version" "${SYSTEM_INFO[kernel_version]}"
    printf '%-16s : %s\n' "Architecture" "${SYSTEM_INFO[architecture]}"
    printf '%-16s : %s\n' "CPU Model" "${SYSTEM_INFO[cpu_model]}"
    printf '%-16s : %s\n' "CPU Vendor" "${SYSTEM_INFO[cpu_vendor]}"
    printf '%-16s : %s\n' "CPU Family" "${SYSTEM_INFO[cpu_family]}"
    printf '%-16s : %s\n' "CPU Model #" "${SYSTEM_INFO[cpu_model_number]}"
    printf '%-16s : %s\n' "CPU Stepping" "${SYSTEM_INFO[cpu_stepping]}"
    printf '%-16s : %s\n' "CPU Microcode" "${SYSTEM_INFO[cpu_microcode]}"
    printf '%-16s : %s\n' "CPU Cores" "${SYSTEM_INFO[cpu_cores]}"
    printf '%-16s : %s\n' "CPU Threads" "${SYSTEM_INFO[cpu_threads]}"
    printf '%-16s : %s\n' "CPU Sockets" "${SYSTEM_INFO[cpu_sockets]}"
    printf '%-16s : %s\n' "CPUs Onboard" "${SYSTEM_INFO[cpus_onboard]}"
    printf '%-16s : %s\n' "Bogo MIPS" "${SYSTEM_INFO[cpu_bogo_mips]}"
    printf '%-16s : %s\n' "CPU Age" "${SYSTEM_INFO[cpu_age_estimate]}"
    printf '%-16s : %s\n' "Memory Total" "${SYSTEM_INFO[memory_total]}"
    printf '%-16s : %s\n' "Memory Type" "${SYSTEM_INFO[memory_type]}"
    printf '%-16s : %s\n' "Memory Speed" "${SYSTEM_INFO[memory_speed]}"
    printf '%-16s : %s\n' "Memory Slots" "${SYSTEM_INFO[memory_slots]}"
    printf '%-16s : %s\n' "Disk Total" "${SYSTEM_INFO[disk_total]}"
    printf '%-16s : %s\n' "Disk Model" "${SYSTEM_INFO[disk_model]}"
    printf '%-16s : %s\n' "Disk Type" "${SYSTEM_INFO[disk_type]}"
    printf '%-16s : %s\n' "Disk Rotational" "${SYSTEM_INFO[disk_rotational]}"
    printf '%-16s : %s\n' "Disk Age" "${SYSTEM_INFO[disk_age_estimate]}"
    printf '%-16s : %s\n' "IP Address" "${SYSTEM_INFO[ip_address]}"
    printf '%-16s : %s\n' "Uptime" "${SYSTEM_INFO[uptime]}"
}

fill_powercap_info() {
    local _base_dir
    local _control_type _enabled _zones
    local _socket_index _socket_count
    local _package_path _package_name _package_enabled _package_energy_uj _package_max_energy_range_uj
    local _constraint_name _constraint_power_limit_uw _constraint_time_window_us
    local _zone_counter

    _base_dir=""
    _control_type="unsupported"
    _enabled="unknown"
    _zones="0"
    _zone_counter=0

    for _base_dir in /sys/class/powercap /sys/devices/virtual/powercap; do
        if [[ -d "${_base_dir}" ]]; then
            if compgen -G "${_base_dir}/intel-rapl:*" > /dev/null; then
                _control_type="sysfs"
                break
            fi
        fi
    done

    if [[ "${_control_type}" == "sysfs" ]]; then
        _enabled="1"
        _socket_count="${SYSTEM_INFO[cpu_sockets]}"
        [[ "${_socket_count}" =~ ^[0-9]+$ ]] || _socket_count=1

        for (( _socket_index=0; _socket_index<_socket_count; _socket_index++ )); do
            _package_path="${_base_dir}/intel-rapl:${_socket_index}"

            if [[ ! -d "${_package_path}" ]]; then
                POWERCAP_INFO["package_${_socket_index}_present"]="0"
                POWERCAP_INFO["package_${_socket_index}_path"]="missing"
                continue
            fi

            _zone_counter=$((_zone_counter + 1))

            _package_name="$(read_first_line_or_unknown "${_package_path}/name")"
            _package_enabled="$(read_first_line_or_unknown "${_package_path}/enabled")"
            _package_energy_uj="$(read_first_line_or_unknown "${_package_path}/energy_uj")"
            _package_max_energy_range_uj="$(read_first_line_or_unknown "${_package_path}/max_energy_range_uj")"

            POWERCAP_INFO["package_${_socket_index}_present"]="1"
            POWERCAP_INFO["package_${_socket_index}_path"]="${_package_path}"
            POWERCAP_INFO["package_${_socket_index}_name"]="${_package_name}"
            POWERCAP_INFO["package_${_socket_index}_enabled"]="${_package_enabled}"
            POWERCAP_INFO["package_${_socket_index}_energy_uj"]="${_package_energy_uj}"
            POWERCAP_INFO["package_${_socket_index}_max_energy_range_uj"]="${_package_max_energy_range_uj}"

            for _constraint_name_path in "${_package_path}"/constraint_*_name; do
                [[ -e "${_constraint_name_path}" ]] || continue

                _constraint_idx="${_constraint_name_path##*/constraint_}"
                _constraint_idx="${_constraint_idx%_name}"

                _constraint_name="$(read_first_line_or_unknown "${_package_path}/constraint_${_constraint_idx}_name")"
                _constraint_power_limit_uw="$(read_first_line_or_unknown "${_package_path}/constraint_${_constraint_idx}_power_limit_uw")"
                _constraint_time_window_us="$(read_first_line_or_unknown "${_package_path}/constraint_${_constraint_idx}_time_window_us")"

                POWERCAP_INFO["package_${_socket_index}_constraint_${_constraint_idx}_name"]="${_constraint_name}"
                POWERCAP_INFO["package_${_socket_index}_constraint_${_constraint_idx}_power_limit_uw"]="${_constraint_power_limit_uw}"
                POWERCAP_INFO["package_${_socket_index}_constraint_${_constraint_idx}_time_window_us"]="${_constraint_time_window_us}"
            done
        done

        _zones="${_zone_counter}"
    fi

    POWERCAP_INFO[control_type]="${_control_type}"
    POWERCAP_INFO[enabled]="${_enabled}"
    POWERCAP_INFO[zones]="${_zones}"

    printf '\n'
    printf '========================================\n'
    printf '        Powercap Information            \n'
    printf '========================================\n'
    printf '%-16s : %s\n' "Control Type" "${POWERCAP_INFO[control_type]}"
    printf '%-16s : %s\n' "Enabled" "${POWERCAP_INFO[enabled]}"
    printf '%-16s : %s\n' "Zones" "${POWERCAP_INFO[zones]}"

    _socket_count="${SYSTEM_INFO[cpu_sockets]}"
    [[ "${_socket_count}" =~ ^[0-9]+$ ]] || _socket_count=1

    for (( _socket_index=0; _socket_index<_socket_count; _socket_index++ )); do
        printf '\n'
        printf '%s\n' "Package ${_socket_index}:"
        printf '%-16s : %s\n' "Present" "${POWERCAP_INFO[package_${_socket_index}_present]:-0}"
        printf '%-16s : %s\n' "Name" "${POWERCAP_INFO[package_${_socket_index}_name]:-unknown}"
        printf '%-16s : %s\n' "Enabled" "${POWERCAP_INFO[package_${_socket_index}_enabled]:-unknown}"
        printf '%-16s : %s\n' "Energy (uJ)" "${POWERCAP_INFO[package_${_socket_index}_energy_uj]:-unknown}"
        printf '%-16s : %s\n' "Max Range" "$(format_microjoules "${POWERCAP_INFO[package_${_socket_index}_max_energy_range_uj]:-unknown}")"
        printf '%-16s : %s\n' "Path" "${POWERCAP_INFO[package_${_socket_index}_path]:-unknown}"

        for _constraint_idx in 0 1 2 3; do
            if [[ -n "${POWERCAP_INFO[package_${_socket_index}_constraint_${_constraint_idx}_name]:-}" ]]; then
                printf '%-16s : %s\n' "Constraint ${_constraint_idx}" "${POWERCAP_INFO[package_${_socket_index}_constraint_${_constraint_idx}_name]}"
                printf '%-16s : %s\n' "Limit ${_constraint_idx}" "$(format_microwatts "${POWERCAP_INFO[package_${_socket_index}_constraint_${_constraint_idx}_power_limit_uw]}")"
                printf '%-16s : %s\n' "Window ${_constraint_idx}" "${POWERCAP_INFO[package_${_socket_index}_constraint_${_constraint_idx}_time_window_us]:-unknown} us"
            fi
        done
    done
}

fill_rapl_info() {
    local _base_dir
    local _control_type _enabled _zones _modprobe_status _available_domains
    local _socket_index _socket_count
    local _package_path _package_name _package_enabled _package_energy_uj _package_max_energy_range_uj
    local _sub_path _sub_name _sub_enabled _sub_energy_uj _sub_max_energy_range_uj
    local _constraint_name _constraint_power_limit_uw _constraint_time_window_us
    local _zone_counter
    local _module _module_line
    local _constraint_idx _sub_key _domain_list

    _base_dir=""
    _control_type="unsupported"
    _enabled="unknown"
    _zones="0"
    _modprobe_status=""
    _available_domains=""
    _zone_counter=0

    for _module in intel_rapl_common intel_rapl_msr intel_rapl; do
        if modprobe -n -v "${_module}" >/dev/null 2>&1; then
            _module_line="$(
                modprobe -n -v "${_module}" 2>/dev/null | paste -sd ' ; ' - \
                || echo "loadable"
            )"
            [[ -n "${_module_line}" ]] || _module_line="loadable"
            _modprobe_status+="${_module}: ${_module_line}"$'\n'
        else
            _module_line="$(modprobe -n -v "${_module}" 2>&1 | tr '\n' ' ' | sed 's/[[:space:]]\+/ /g; s/[[:space:]]*$//')"
            [[ -n "${_module_line}" ]] || _module_line="not available"
            _modprobe_status+="${_module}: ${_module_line}"$'\n'
        fi
    done
    _modprobe_status="${_modprobe_status%$'\n'}"

    for _base_dir in /sys/class/powercap /sys/devices/virtual/powercap; do
        if [[ -d "${_base_dir}" ]]; then
            if compgen -G "${_base_dir}/intel-rapl:*" > /dev/null; then
                _control_type="sysfs"
                break
            fi
        fi
    done

    if [[ "${_control_type}" == "sysfs" ]]; then
        _enabled="1"
        _socket_count="${SYSTEM_INFO[cpu_sockets]}"
        [[ "${_socket_count}" =~ ^[0-9]+$ ]] || _socket_count=1

        for (( _socket_index=0; _socket_index<_socket_count; _socket_index++ )); do
            _package_path="${_base_dir}/intel-rapl:${_socket_index}"
            _domain_list="package"

            if [[ ! -d "${_package_path}" ]]; then
                RAPL_INFO["package_${_socket_index}_present"]="0"
                RAPL_INFO["package_${_socket_index}_path"]="missing"
                RAPL_INFO["package_${_socket_index}_available_domains"]="none"
                continue
            fi

            _zone_counter=$((_zone_counter + 1))

            _package_name="$(read_first_line_or_unknown "${_package_path}/name")"
            _package_enabled="$(read_first_line_or_unknown "${_package_path}/enabled")"
            _package_energy_uj="$(read_first_line_or_unknown "${_package_path}/energy_uj")"
            _package_max_energy_range_uj="$(read_first_line_or_unknown "${_package_path}/max_energy_range_uj")"

            RAPL_INFO["package_${_socket_index}_present"]="1"
            RAPL_INFO["package_${_socket_index}_path"]="${_package_path}"
            RAPL_INFO["package_${_socket_index}_name"]="${_package_name}"
            RAPL_INFO["package_${_socket_index}_enabled"]="${_package_enabled}"
            RAPL_INFO["package_${_socket_index}_energy_uj"]="${_package_energy_uj}"
            RAPL_INFO["package_${_socket_index}_max_energy_range_uj"]="${_package_max_energy_range_uj}"

            for _constraint_name_path in "${_package_path}"/constraint_*_name; do
                [[ -e "${_constraint_name_path}" ]] || continue

                _constraint_idx="${_constraint_name_path##*/constraint_}"
                _constraint_idx="${_constraint_idx%_name}"

                _constraint_name="$(read_first_line_or_unknown "${_package_path}/constraint_${_constraint_idx}_name")"
                _constraint_power_limit_uw="$(read_first_line_or_unknown "${_package_path}/constraint_${_constraint_idx}_power_limit_uw")"
                _constraint_time_window_us="$(read_first_line_or_unknown "${_package_path}/constraint_${_constraint_idx}_time_window_us")"

                RAPL_INFO["package_${_socket_index}_constraint_${_constraint_idx}_name"]="${_constraint_name}"
                RAPL_INFO["package_${_socket_index}_constraint_${_constraint_idx}_power_limit_uw"]="${_constraint_power_limit_uw}"
                RAPL_INFO["package_${_socket_index}_constraint_${_constraint_idx}_time_window_us"]="${_constraint_time_window_us}"
            done

            for _sub_path in "${_package_path}"/intel-rapl:*; do
                [[ -d "${_sub_path}" ]] || continue

                _zone_counter=$((_zone_counter + 1))
                _sub_name="$(read_first_line_or_unknown "${_sub_path}/name")"
                _sub_enabled="$(read_first_line_or_unknown "${_sub_path}/enabled")"
                _sub_energy_uj="$(read_first_line_or_unknown "${_sub_path}/energy_uj")"
                _sub_max_energy_range_uj="$(read_first_line_or_unknown "${_sub_path}/max_energy_range_uj")"

                case "${_sub_name}" in
                    core|cores)
                        _sub_key="core"
                        ;;
                    uncore)
                        _sub_key="uncore"
                        ;;
                    dram)
                        _sub_key="dram"
                        ;;
                    psys|platform)
                        _sub_key="platform"
                        ;;
                    gpu)
                        _sub_key="gpu"
                        ;;
                    *)
                        _sub_key="$(basename "${_sub_path}" | tr ':-' '__')"
                        ;;
                esac

                case ",${_domain_list}," in
                    *,"${_sub_key}",*)
                        ;;
                    *)
                        _domain_list="${_domain_list},${_sub_key}"
                        ;;
                esac

                RAPL_INFO["package_${_socket_index}_${_sub_key}_name"]="${_sub_name}"
                RAPL_INFO["package_${_socket_index}_${_sub_key}_enabled"]="${_sub_enabled}"
                RAPL_INFO["package_${_socket_index}_${_sub_key}_energy_uj"]="${_sub_energy_uj}"
                RAPL_INFO["package_${_socket_index}_${_sub_key}_max_energy_range_uj"]="${_sub_max_energy_range_uj}"
                RAPL_INFO["package_${_socket_index}_${_sub_key}_path"]="${_sub_path}"
            done

            RAPL_INFO["package_${_socket_index}_available_domains"]="${_domain_list}"

            if [[ -n "${_available_domains}" ]]; then
                _available_domains="${_available_domains}; package_${_socket_index}=${_domain_list}"
            else
                _available_domains="package_${_socket_index}=${_domain_list}"
            fi
        done

        _zones="${_zone_counter}"
    fi

    [[ -n "${_available_domains}" ]] || _available_domains="none"

    RAPL_INFO[control_type]="${_control_type}"
    RAPL_INFO[enabled]="${_enabled}"
    RAPL_INFO[zones]="${_zones}"
    RAPL_INFO[available_domains]="${_available_domains}"
    RAPL_INFO[modprobe_status]="${_modprobe_status}"

    printf '\n'
    printf '========================================\n'
    printf '          RAPL Information              \n'
    printf '========================================\n'
    printf '%-16s : %s\n' "Control Type" "${RAPL_INFO[control_type]}"
    printf '%-16s : %s\n' "Enabled" "${RAPL_INFO[enabled]}"
    printf '%-16s : %s\n' "Zones" "${RAPL_INFO[zones]}"
    printf '%-16s : %s\n' "Domains" "${RAPL_INFO[available_domains]}"
    printf '%-16s : %s\n' "Modprobe" ""

    while IFS= read -r _module_line; do
        printf '  %s\n' "${_module_line}"
    done <<< "${RAPL_INFO[modprobe_status]}"

    _socket_count="${SYSTEM_INFO[cpu_sockets]}"
    [[ "${_socket_count}" =~ ^[0-9]+$ ]] || _socket_count=1

    for (( _socket_index=0; _socket_index<_socket_count; _socket_index++ )); do
        printf '\n'
        printf '%s\n' "Package ${_socket_index}:"
        printf '%-16s : %s\n' "Present" "${RAPL_INFO[package_${_socket_index}_present]:-0}"
        printf '%-16s : %s\n' "Name" "${RAPL_INFO[package_${_socket_index}_name]:-unknown}"
        printf '%-16s : %s\n' "Enabled" "${RAPL_INFO[package_${_socket_index}_enabled]:-unknown}"
        printf '%-16s : %s\n' "Domains" "${RAPL_INFO[package_${_socket_index}_available_domains]:-none}"
        printf '%-16s : %s\n' "Energy (uJ)" "${RAPL_INFO[package_${_socket_index}_energy_uj]:-unknown}"
        printf '%-16s : %s\n' "Max Range" "$(format_microjoules "${RAPL_INFO[package_${_socket_index}_max_energy_range_uj]:-unknown}")"
        printf '%-16s : %s\n' "Path" "${RAPL_INFO[package_${_socket_index}_path]:-unknown}"

        for _constraint_idx in 0 1 2 3; do
            if [[ -n "${RAPL_INFO[package_${_socket_index}_constraint_${_constraint_idx}_name]:-}" ]]; then
                printf '%-16s : %s\n' "Constraint ${_constraint_idx}" "${RAPL_INFO[package_${_socket_index}_constraint_${_constraint_idx}_name]}"
                printf '%-16s : %s\n' "Limit ${_constraint_idx}" "$(format_microwatts "${RAPL_INFO[package_${_socket_index}_constraint_${_constraint_idx}_power_limit_uw]}")"
                printf '%-16s : %s\n' "Window ${_constraint_idx}" "${RAPL_INFO[package_${_socket_index}_constraint_${_constraint_idx}_time_window_us]:-unknown} us"
            fi
        done

        for _sub_key in core uncore dram gpu platform; do
            if [[ -n "${RAPL_INFO[package_${_socket_index}_${_sub_key}_name]:-}" ]]; then
                printf '%-16s : %s\n' "${_sub_key^} Name" "${RAPL_INFO[package_${_socket_index}_${_sub_key}_name]}"
                printf '%-16s : %s\n' "${_sub_key^} Enabled" "${RAPL_INFO[package_${_socket_index}_${_sub_key}_enabled]:-unknown}"
                printf '%-16s : %s\n' "${_sub_key^} Energy" "${RAPL_INFO[package_${_socket_index}_${_sub_key}_energy_uj]:-unknown}"
                printf '%-16s : %s\n' "${_sub_key^} Range" "$(format_microjoules "${RAPL_INFO[package_${_socket_index}_${_sub_key}_max_energy_range_uj]:-unknown}")"
                printf '%-16s : %s\n' "${_sub_key^} Path" "${RAPL_INFO[package_${_socket_index}_${_sub_key}_path]:-unknown}"
            fi
        done
    done
}

enable_supported_rapl_modules_interactive() {
    local _answer _module_line _module _status_line
    local _base_dir _socket_count _socket_index
    local _package_present _package_path _package_name _package_enabled
    local _had_change _changed_any _writable_found
    local _sub_path _sub_name _sub_enabled
    local _enable_file
    local _domain_list _domain _key

    printf '\n'
    printf '========================================\n'
    printf '   Enable Supported RAPL Modules        \n'
    printf '========================================\n'

    printf '%s\n' "Using previously collected RAPL and powercap information."

    printf '\n'
    printf '%s\n' "Detected control type summary:"
    printf '  Powercap : %s\n' "${POWERCAP_INFO[control_type]}"
    printf '  RAPL     : %s\n' "${RAPL_INFO[control_type]}"
    printf '  Zones    : %s\n' "${RAPL_INFO[zones]}"
    printf '  Domains  : %s\n' "${RAPL_INFO[available_domains]}"

    printf '\n'
    printf '%s\n' "Previously detected modprobe status:"
    while IFS= read -r _module_line; do
        [[ -n "${_module_line}" ]] || continue
        printf '  %s\n' "${_module_line}"
    done <<< "${RAPL_INFO[modprobe_status]}"

    if [[ "${RAPL_INFO[control_type]}" == "sysfs" || "${POWERCAP_INFO[control_type]}" == "sysfs" ]]; then
        printf '\n'
        printf '%s\n' "RAPL sysfs support is already present."
    else
        printf '\n'
        printf '%s\n' "RAPL sysfs support is not currently visible."
        printf '%s' "Would you like to attempt loading available RAPL-related kernel modules? [y/N]: "
        read -r _answer

        if [[ "${_answer}" =~ ^[Yy]([Ee][Ss])?$ ]]; then
            while IFS= read -r _module_line; do
                [[ -n "${_module_line}" ]] || continue
                _module="${_module_line%%:*}"
                _status_line="${_module_line#*: }"

                printf '\n'
                printf 'Module: %s\n' "${_module}"
                printf 'Status: %s\n' "${_status_line}"

                if [[ "${_status_line}" == *"not available"* ]]; then
                    printf '%s\n' "Skipping: module is not available."
                    continue
                fi

                printf '%s' "Attempt to load ${_module}? [y/N]: "
                read -r _answer
                if [[ "${_answer}" =~ ^[Yy]([Ee][Ss])?$ ]]; then
                    if modprobe "${_module}" 2>/dev/null; then
                        printf '%s\n' "Loaded ${_module} successfully."
                    else
                        printf '%s\n' "Failed to load ${_module}. You may need elevated privileges."
                    fi
                else
                    printf '%s\n' "Skipped ${_module}."
                fi
            done <<< "${RAPL_INFO[modprobe_status]}"

            printf '\n'
            printf '%s\n' "Module load attempts completed."
            printf '%s\n' "Note: this function intentionally reuses existing discovery data and does not rescan sysfs."
        else
            printf '%s\n' "Module loading skipped by user."
        fi
    fi

    _base_dir=""
    for _key in /sys/class/powercap /sys/devices/virtual/powercap; do
        if [[ -d "${_key}" ]]; then
            _base_dir="${_key}"
            break
        fi
    done

    if [[ -z "${_base_dir}" ]]; then
        printf '\n'
        printf '%s\n' "No powercap base directory is present on this system."
        return 0
    fi

    _socket_count="${SYSTEM_INFO[cpu_sockets]}"
    [[ "${_socket_count}" =~ ^[0-9]+$ ]] || _socket_count=1

    printf '\n'
    printf '%s\n' "Package/domain enablement plan from cached RAPL data:"
    for (( _socket_index=0; _socket_index<_socket_count; _socket_index++ )); do
        _package_present="${RAPL_INFO[package_${_socket_index}_present]:-0}"
        _package_path="${RAPL_INFO[package_${_socket_index}_path]:-missing}"
        _package_name="${RAPL_INFO[package_${_socket_index}_name]:-unknown}"
        _package_enabled="${RAPL_INFO[package_${_socket_index}_enabled]:-unknown}"
        _domain_list="${RAPL_INFO[package_${_socket_index}_available_domains]:-none}"

        printf '  Package %s\n' "${_socket_index}"
        printf '    Present : %s\n' "${_package_present}"
        printf '    Name    : %s\n' "${_package_name}"
        printf '    Enabled : %s\n' "${_package_enabled}"
        printf '    Domains : %s\n' "${_domain_list}"
        printf '    Path    : %s\n' "${_package_path}"
    done

    printf '\n'
    printf '%s' "Would you like to interactively enable package and domain zones that support writes to 'enabled'? [y/N]: "
    read -r _answer
    if [[ ! "${_answer}" =~ ^[Yy]([Ee][Ss])?$ ]]; then
        printf '%s\n' "Interactive enablement skipped by user."
        return 0
    fi

    _changed_any=0

    for (( _socket_index=0; _socket_index<_socket_count; _socket_index++ )); do
        _package_present="${RAPL_INFO[package_${_socket_index}_present]:-0}"
        _package_path="${RAPL_INFO[package_${_socket_index}_path]:-missing}"
        _package_name="${RAPL_INFO[package_${_socket_index}_name]:-unknown}"
        _package_enabled="${RAPL_INFO[package_${_socket_index}_enabled]:-unknown}"

        printf '\n'
        printf '%s\n' "----------------------------------------"
        printf 'Package %s\n' "${_socket_index}"
        printf '%s\n' "----------------------------------------"

        if [[ "${_package_present}" != "1" ]]; then
            printf '%s\n' "Package is not present in cached RAPL data. Skipping."
            continue
        fi

        printf 'Name    : %s\n' "${_package_name}"
        printf 'Enabled : %s\n' "${_package_enabled}"
        printf 'Path    : %s\n' "${_package_path}"

        _had_change=0
        _writable_found=0

        _enable_file="${_package_path}/enabled"
        if [[ -e "${_enable_file}" ]]; then
            if [[ -w "${_enable_file}" ]]; then
                _writable_found=1
                if [[ "${_package_enabled}" == "1" ]]; then
                    printf '%s\n' "Package zone already enabled."
                else
                    printf '%s' "Enable package zone '${_package_name}'? [y/N]: "
                    read -r _answer
                    if [[ "${_answer}" =~ ^[Yy]([Ee][Ss])?$ ]]; then
                        if printf '1\n' > "${_enable_file}" 2>/dev/null; then
                            printf '%s\n' "Package zone enabled."
                            _had_change=1
                            _changed_any=1
                        else
                            printf '%s\n' "Failed to enable package zone. You may need elevated privileges."
                        fi
                    else
                        printf '%s\n' "Package zone skipped."
                    fi
                fi
            else
                printf '%s\n' "Package zone has an enabled file but it is not writable."
            fi
        else
            printf '%s\n' "Package zone does not expose an enabled file."
        fi

        for _domain in package core uncore dram gpu platform; do
            [[ "${_domain}" != "package" ]] || continue

            if [[ -z "${RAPL_INFO[package_${_socket_index}_${_domain}_path]:-}" ]]; then
                continue
            fi

            _sub_path="${RAPL_INFO[package_${_socket_index}_${_domain}_path]}"
            _sub_name="${RAPL_INFO[package_${_socket_index}_${_domain}_name]:-${_domain}}"
            _sub_enabled="${RAPL_INFO[package_${_socket_index}_${_domain}_enabled]:-unknown}"
            _enable_file="${_sub_path}/enabled"

            printf '\n'
            printf 'Domain  : %s\n' "${_sub_name}"
            printf 'Enabled : %s\n' "${_sub_enabled}"
            printf 'Path    : %s\n' "${_sub_path}"

            if [[ ! -e "${_enable_file}" ]]; then
                printf '%s\n' "This domain does not expose an enabled file."
                continue
            fi

            if [[ ! -w "${_enable_file}" ]]; then
                printf '%s\n' "Enabled file is present but not writable."
                continue
            fi

            _writable_found=1

            if [[ "${_sub_enabled}" == "1" ]]; then
                printf '%s\n' "Domain already enabled."
                continue
            fi

            printf '%s' "Enable domain '${_sub_name}'? [y/N]: "
            read -r _answer
            if [[ "${_answer}" =~ ^[Yy]([Ee][Ss])?$ ]]; then
                if printf '1\n' > "${_enable_file}" 2>/dev/null; then
                    printf '%s\n' "Domain enabled."
                    _had_change=1
                    _changed_any=1
                else
                    printf '%s\n' "Failed to enable domain. You may need elevated privileges."
                fi
            else
                printf '%s\n' "Domain skipped."
            fi
        done

        if [[ "${_writable_found}" -eq 0 ]]; then
            printf '%s\n' "No writable enable controls were found for this package."
        elif [[ "${_had_change}" -eq 0 ]]; then
            printf '%s\n' "No changes were made for this package."
        fi
    done

    printf '\n'
    printf '========================================\n'
    printf '             Summary                    \n'
    printf '========================================\n'
    if [[ "${_changed_any}" -eq 1 ]]; then
        printf '%s\n' "One or more RAPL package/domain zones were enabled."
        printf '%s\n' "Cached data structures were intentionally reused and may not reflect live post-change state until the discovery functions are run again."
    else
        printf '%s\n' "No RAPL enablement changes were made."
    fi
}

fill_system_info
fill_powercap_info
fill_rapl_info
enable_supported_rapl_modules_interactive
