from accumulation.cgroups import CgroupV2
from accumulation.docker import DockerManager
from accumulation.k8s import K8sManager
from config import env_flag
from database.DBClient import DBClient
from exporter.exporter import PrometheusExporter
from inference.api import InferenceRequest
from smart_meter_api_wrapper.smart_meter import SmartMeterAPIClient


def create_db_client(args, env_path):
    if not args.use_influxdb:
        return None

    missing_influx = [
        name
        for name, value in {
            "influx_url": args.influx_url,
            "influx_token": args.influx_token,
            "influx_org": args.influx_org,
            "influx_bucket": args.influx_bucket,
        }.items()
        if value is None or (isinstance(value, str) and value.strip() == "")
    ]
    if missing_influx:
        raise ValueError(
            "Missing InfluxDB settings: "
            + ", ".join(missing_influx)
            + f". Provide CLI args or set them in {env_path}."
        )

    print(
        f"Influx: {args.influx_url} (org={args.influx_org}, bucket={args.influx_bucket})"
    )
    return DBClient(
        args.influx_url,
        args.influx_token,
        args.influx_org,
        args.influx_bucket,
    )


def create_meter_client(args, env_path):
    meter_sensor_id = args.meter_sensor_id
    if not args.use_meter:
        print("Smart meter integration disabled")
        return None, meter_sensor_id

    missing_meter = [
        name
        for name, value in {
            "meter_host": args.meter_host,
            "meter_user": args.meter_user,
            "meter_password": args.meter_password,
        }.items()
        if value is None or (isinstance(value, str) and value.strip() == "")
    ]
    if missing_meter:
        raise ValueError(
            "Missing Smart Meter settings: "
            + ", ".join(missing_meter)
            + f". Provide CLI args or set them in {env_path}."
        )

    meter_sensor_id = args.meter_sensor_id
    meter_ssl = args.meter_ssl or env_flag("SMARTMETER_SSL", default=False)
    meter_client = SmartMeterAPIClient(
        host=args.meter_host,
        ssl=meter_ssl,
        username=args.meter_user,
        password=args.meter_password,
    )
    return meter_client, meter_sensor_id


def create_exporter(args):
    if not args.use_prometheus_exporter:
        return None

    if not args.exporter_addr or not args.exporter_port:
        raise ValueError(
            "--exporter-addr and --exporter-port is required when --use-prometheus-exporter is enabled"
        )
    return PrometheusExporter(
        node="localhost",
        addr=args.exporter_addr,
        port=int(args.exporter_port),
        mode=args.exporter_mode,
    )


def create_cgroups_manager():
    return CgroupV2()


def create_docker_manager(args, cgroups_manager):
    if not args.docker_integration:
        return None

    print("Docker integration enabled.")
    docker_manager = DockerManager(cgroups_manager)
    # Pass the callback to DockerManager so cgroups_manager receives container events
    docker_manager.run(callback=cgroups_manager.handle_container_event)
    return docker_manager


def create_online_estimator(args):
    online_estimator = None
    if args.model_pkl:
        print(f"Model provided; enabling inference with model: {args.model_pkl}")
        online_estimator = InferenceRequest(args.model_pkl)

    if args.online_energy_estimation:
        # Placeholder: online energy estimation logic to be implemented later.
        print("Online energy estimation flag set (placeholder; not yet implemented).")

    return online_estimator


def create_k8s_manager(args, cgroups_manager):
    if not args.kubernetes_integration:
        return None

    if not args.kubeconfig:
        raise ValueError(
            "--kubeconfig is required when --kubernetes-integration is enabled"
        )
    print("Kubernetes integration enabled.")
    k8s_manager = K8sManager(
        args.kubeconfig,
        cgroups_manager,
        use_pod_regex=args.use_pod_regex,
    )
    # k8s_manager.run(callback=cgroups_manager.handle_pod_container_event)
    k8s_manager.run(callback=cgroups_manager.handle_pod_container_event)
    return k8s_manager
