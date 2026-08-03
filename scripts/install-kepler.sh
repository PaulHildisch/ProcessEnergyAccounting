# 1. Install cert-manager (required for operator webhooks)
# For the latest version, see: https://cert-manager.io/docs/installation/
kubectl apply -f https://github.com/cert-manager/cert-manager/releases/download/v1.18.2/cert-manager.yaml

# Wait for cert-manager to be ready
kubectl wait --for=condition=available --timeout=120s deployment -n cert-manager --all

# 2. Install Prometheus Operator (required for ServiceMonitor support)
# This installs prometheus-operator + Prometheus + Grafana
# If you only need prometheus-operator, see the monitoring stack guide
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm install prometheus prometheus-community/kube-prometheus-stack \
  --namespace monitoring \
  --create-namespace \
  --set prometheus.prometheusSpec.serviceMonitorSelectorNilUsesHelmValues=false

# Wait for monitoring stack to be ready
kubectl wait --for=condition=ready --timeout=180s pod -n monitoring --all

# 3. Install Kepler Operator
helm install kepler-operator \
  oci://quay.io/sustainable_computing_io/charts/kepler-operator \
  --namespace kepler-operator \
  --create-namespace

# Wait for operator to be ready
kubectl wait --for=condition=available --timeout=120s deployment -n kepler-operator --all

# 4. Deploy Kepler
# Note: PowerMonitor must be named "power-monitor" (enforced by operator)
kubectl apply -f https://raw.githubusercontent.com/sustainable-computing-io/kepler-operator/main/config/samples/kepler.system_v1alpha1_powermonitor.yaml

# Wait for Kepler pods to be running
kubectl wait --for=condition=ready --timeout=120s pod -n power-monitor --all

# 5. Verify installation
kubectl get pods -n power-monitor
